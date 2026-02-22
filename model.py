# GPU prealloc false for xla 
# import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(0.8)

# imports
import os; os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax 
import flax 
import jax.numpy as jnp
from flax import nnx as nn 
# data pipeline and loaders importing
from data import run_data_config_pipeline

# data_iterator = run_data_config_pipeline(data_iterator=True)
# assert data_iterator is not None, "Data iterator should not be None when data_iterator=True"
# print(next(data_iterator).shape)


###########################################################################
#                     MODEL BLOCKS
###########################################################################

class PositionalEncoding(nn.Module):
    """
    Injects temporal inductive bias into the embedding vectors via deterministic multi scale fourier frequencies 
    Pre-computes the encoding matrix to eliminate forward pass trigonometric FLOP overhead 
    """
    def __init__(self, d_model:int, max_len:int=5_000, dtype:jnp.dtype=jnp.bfloat16) -> None:
        super().__init__()

        # precomputes the structural tensor in float32 for high precision math 
        pe = jnp.zeros((max_len, d_model), dtype=jnp.float32)  
        # positional indices : (max_len, 1)
        position = jnp.arange(0, max_len, dtype=jnp.float32)[:,jnp.newaxis] 
        # Frequency Exponential scaling 
        div_term = jnp.exp(jnp.arange(0, d_model, 2, dtype=jnp.float32) * -(jnp.log(10000.0)/d_model))
        # Apply sin to even indices and cos to odd indices using JAX .at sytax 
        pe = pe.at[:, 0::2].set(jnp.sin(position*div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position*div_term))
        # Expand batch dimension for XLA broadcasting then cast down to bfloat16 to save bandwidth 
        self.pe_cache = pe[jnp.newaxis,:, :].astype(dtype=dtype) # (1, max_len, d_model)

    def __call__(self, x:jax.Array) -> jax.Array:
        # x shape: (batch, seqLength, d_model) 
        seq_len=x.shape[1]
        # slice the cache upto seq_len and brodcast 
        return x + self.pe_cache[:, :seq_len, :]

class TransformerEmbedding(nn.Module): 
    """
    Combines discrete token lookup with continuos positional embedding using embedding lookup table
    """
    def __init__(self, vocab_size:int, d_model:int, max_seq_len:int, dropout_rate:float, rngs:nn.Rngs, dtype:jnp.dtype=jnp.bfloat16): 
        super().__init__() 
        self.d_model = d_model 

        self.token_embed = nn.Embed(num_embeddings=vocab_size, features=d_model, param_dtype=dtype, dtype=dtype, rngs=rngs) # our lookup table 
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_seq_len, dtype=dtype)
        self.dropout = nn.Dropout(rate=dropout_rate, rngs=rngs) 
    
    def __call__(self, x: jax.Array, deterministic:bool = False) -> jax.Array:
        """
        x : Integer array of shape (batch, seq_len)
        """
        # lookup tokens 
        x_emb = self.token_embed(inputs=x)
        # scale up token embedding to match variance of pos encodings
        # without this, pos signal dominates the semantic signal 
        x_emb = x_emb * jnp.sqrt(jnp.array(self.d_model, dtype=x_emb.dtype)) 
        # Inject temporal inductive bias of positonal trigonometric waves 
        x_pe = self.positional_encoding(x_emb) 
        # Apply stochastic regularisation 
        return self.dropout(x_pe, deterministic=deterministic)

class MultiHeadSelfAttention(nn.Module):
    """
    RTX 3060 optimised multi-head self attention module.
    Works best with ampere gpus 
    Functionality:
        - Uses fused QKV projections for better performance 
        - Has mixed precision softmax for improved numerical stability
    """
    def __init__(self, 
                 d_model:int, 
                 num_heads:int, 
                 rngs:nn.Rngs, 
                 dropout_rate:float=0.0,
                 dtype:jnp.dtype=jnp.bfloat16 ):
        super().__init__()
        # ensure tensor dimensions align properly 
        if d_model % num_heads != 0: raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model 
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dtype=dtype
        
        # scale factor for dot product attention : scale = 1/sqrt(d_k) 
        # normalises the variance of the dot product to prevent it from growing too large,
        # which can lead to vanishing gradients during training
        self.scale = jax.lax.rsqrt(jnp.array(self.d_k, dtype=self.dtype))

        # FUSED PROJECTION: Computes Q, K, V in single matrix multiplication 
        # matrix shape: (d_model, 3 * d_model) to produce concatenated QKV
        # this is more efficient than separate projections for Q, K, V , 
        # since we can fuse a single GEMM operation which is highly optimized on modern hardware
        self.qkv_proj = nn.Linear(
            in_features=self.d_model,
            out_features=3 * self.d_model,
            use_bias=False, # Bias in QKV is generally unnecessary and slows down compute
            dtype=self.dtype,
            param_dtype=self.dtype, 
            rngs=rngs
        )   

        # output projection to combine multi-head outputs back to d_model dimensions
        self.out_proj = nn.Linear(
            in_features=self.d_model, 
            out_features=self.d_model, 
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.dtype,
            rngs=rngs
        )
        self.dropout = nn.Dropout(rate=dropout_rate, rngs=rngs)

    # forward pass for multi-head self attention
    def __call__(self, x:jax.Array, mask:jax.Array|None=None, deterministic:bool=False) -> jax.Array:
        """
        x: Input tensor of shape (batch_size, seq_length, d_model)
        mask: Optional boolean mask of shape (batch_size, 1, seq_length, seq_length) 
              for masking out padding tokens or future tokens in decoder
        deterministic: If True, disable dropout
        """
        B, L, D = x.shape
        assert D == self.d_model, f"Input feature dimension ({D}) must match model dimension ({self.d_model})"

        # fused projection 
        qkv = self.qkv_proj(x) # shape: (B, L, 3 * d_model)

        # reshape and split into Q, K, V
        # Target shape before split : (B, L, 3, num_heads, d_k)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.d_k) 

        # transpose to isolate Q, K, V across 3rd dimension 
        # shape : (3, B, num_heads, L, d_k)
        qkv = jnp.transpose(qkv, (2,0,3,1,4)) # shape : (3, B, num_heads, L, d_K)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        # Scaled dot product attention 
        # Einsum handles the batched multi head multiplication
        # Q @ K^T. 
        logits = jnp.einsum('bhqd, bhkd -> bhqk', q, k) * self.scale

        # apply mask if provided, casual masking for decoder 
        if mask is not None:
            # we use a very large negative number to drive the softmax probability to zero 
            # dtype must match logits 
            neg_inf = jnp.array(-1e9, dtype=logits.dtype)
            logits = jnp.where(mask, logits, neg_inf)
        
        # for numerical stability we cast to fp32 for softmax 
        # then back to bfloat16 
        attention_weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(self.dtype)
        
        # Apply stochastic dropout on attention weights
        attention_weights = self.dropout(attention_weights, deterministic=deterministic)

        # multiplying attention weights to V 
        # v : (B, num_heads, L, d_k) 
        # attention_weights : (B, num_heads, L, L)
        # attention_weights @ v^T shape: (B, num_heads, L, d_k) 
        context = jnp.einsum('bhqk, bhkd -> bhqd',attention_weights, v) 

        # recombining heads and flattening the last two dimensions 
        context = jnp.transpose(context, (0,2,1,3)).reshape(B, L, self.d_model)

        # final linear projections to get maximum context
        return self.out_proj(context)

class CausalMultiHeadAttention(nn.Module): 
    """ 
    Multi-head causal self attention used in the decoder
    This forces autoregressive (left-to-right) generation by masking future positions. 

    Compared to encoder attention:
    - Always applies a causal (lower-triangular mask)
    - Usually called with same sequence as Q=K=V (self-attention)
    """

    def __init__(self, d_model:int, num_heads:int, rngs:nn.Rngs, dropout_rate:float=0.0, dtype: jnp.dtype= jnp.bfloat16):
        super().__init__()

        self.mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, rngs=rngs, dropout_rate=dropout_rate, dtype=dtype)
        self.num_heads=num_heads
        self.dtype=dtype

    def __call__(self, x: jax.Array, padding_mask:jax.Array|None=None, deterministic:bool=False) -> jax.Array:
        """ 
        Args:
            x : Query, key and value all come from the same tensor of shape(batch, target_seq_len, d_model) 
            padding_mask: Optinal padding mask (usually from target side) of shape(batch, 1, 1, target_seq_len) - true = attend 
            deterministic: If True, disable dropout

        Returns: 
            Output of shape (batch, target_seq_Len, d_model)
        """

        B, L_dec, d_model = x.shape
        causal_mask = jnp.tril(jnp.ones((L_dec, L_dec), dtype=jnp.bool_))

        # reshape for broadcast -> (1, 1, L_dec, L_dec)
        causal_mask = causal_mask[None, None, :, :] # this adds two extra dims 

        # combine padding mask and causal mask 
        if padding_mask is not None:
            # padding mask : (B, 1, 1 , L_dec) 
            # causal mask : (1, 1, L_dec, L_dec) 
            # combined shape : (B, 1, L_dec, L_dec) 

            # True -> can attend 
            # False -> Cannot attend 
            combined_mask = padding_mask & causal_mask # logical AND operation 

        else: combined_mask = causal_mask 

        # use mha to pass in mask as a parameter 
        return self.mha(x=x, mask=combined_mask, deterministic=deterministic) 

class CrossAttention(nn.Module):
    """
    Cross-attention module for the Transformer decoder.
    Allows the decoder to attend to the encoder's output for alignment.
    """
    def __init__(self, d_model:int, num_heads:int, rngs:nn.Rngs, dropout_rate:float=0.0, dtype:jnp.dtype=jnp.bfloat16):
        super().__init__()
        if d_model % num_heads != 0: raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dtype = dtype
        self.scale = jax.lax.rsqrt(jnp.array(self.d_k, dtype=self.dtype))

        # Separate projections for Queries (Decoder) and Keys/Values (Encoder)
        self.q_proj = nn.Linear(in_features=self.d_model, out_features=self.d_model, use_bias=False, dtype=self.dtype, param_dtype=self.dtype, rngs=rngs)
        self.kv_proj = nn.Linear(in_features=self.d_model, out_features=2 * self.d_model, use_bias=False, dtype=self.dtype, param_dtype=self.dtype, rngs=rngs)
        self.out_proj = nn.Linear(in_features=self.d_model, out_features=self.d_model, use_bias=False, dtype=self.dtype, param_dtype=self.dtype, rngs=rngs)
        self.dropout = nn.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x:jax.Array, context:jax.Array, mask:jax.Array|None=None, deterministic:bool=False) -> jax.Array:
        """
        x: Decoder queries, shape (B, target_seq_len, d_model)
        context: Encoder output, shape (B, source_seq_len, d_model)
        mask: Optional padding mask from encoder, shape (B, 1, 1, source_seq_len)
        """
        B, L_q, _ = x.shape
        _, L_kv, _ = context.shape

        q = self.q_proj(x) # (B, L_q, d_model)
        kv = self.kv_proj(context) # (B, L_kv, 2 * d_model)

        q = q.reshape(B, L_q, self.num_heads, self.d_k)
        kv = kv.reshape(B, L_kv, 2, self.num_heads, self.d_k)

        # transpose: (B, num_heads, L_q, d_k)
        q = jnp.transpose(q, (0, 2, 1, 3))
        # isolate k, v, transpose: (B, num_heads, L_kv, d_k)
        kv = jnp.transpose(kv, (2, 0, 3, 1, 4))
        k, v = kv[0], kv[1]

        # Scaled dot product attention
        logits = jnp.einsum('bhqd, bhkd -> bhqk', q, k) * self.scale

        if mask is not None:
            neg_inf = jnp.array(-1e9, dtype=logits.dtype)
            logits = jnp.where(mask, logits, neg_inf)
        
        attention_weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(self.dtype)
        attention_weights = self.dropout(attention_weights, deterministic=deterministic)

        out = jnp.einsum('bhqk, bhkd -> bhqd', attention_weights, v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, L_q, self.d_model)

        return self.out_proj(out)

class PositionWiseFFN(nn.Module): 
    """
    Position Wise feed forward network
    Applies two linear transformations and a relu activation in between them 
    """
    def __init__(self, d_model:int, d_ffn:int, rngs:nn.Rngs, dtype:jnp.dtype = jnp.bfloat16 ) -> None:
        super().__init__()
        self.ll1 = nn.Linear(in_features=d_model, out_features=d_ffn, dtype=dtype, param_dtype=dtype, rngs=rngs) # expansion layer
        self.ll2 = nn.Linear(in_features=d_ffn, out_features=d_model, dtype=dtype,param_dtype=dtype, rngs=rngs ) # contraction layer

    def __call__(self, x:jax.Array) -> jax.Array:
        # original paper utilises ReLU. Modern variants often use GeLU or SwiGLU, but ReLU is highly efficient .
        return self.ll2(jax.nn.relu(self.ll1(x))) # Layer 1 -> ReLU -> Layer 2 -> Output

class EncoderBlock(nn.Module):
    """   
    Single pre layer-norm transformer encoder block, 
    
    Features:
        - Multi Head Self Attention : Queries across all other tokens in a sentence 
        - FFN : Feed forward expands dimension by ~ 3x for more spatial pattern recognition. Then reduce dims to normal
        - Residual Highways : Implemented residual connections across high computation blocks for smooth gradient flow in large networks
    
    """
    def __init__(self, d_model:int, n_heads:int, d_ffn:int, dropout_rate:float, rngs:nn.Rngs, dtype:jnp.dtype=jnp.bfloat16) -> None:
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model=d_model, num_heads=n_heads, rngs=rngs, dropout_rate=dropout_rate, dtype=dtype) 
        self.ffn = PositionWiseFFN(d_model=d_model, d_ffn=d_ffn, rngs=rngs, dtype=dtype)

        # pre Layer-Norm : Norms are applied BEFORE the sublayers
        self.ln1 = nn.LayerNorm(num_features=d_model, dtype=dtype, param_dtype=dtype,rngs=rngs)
        self.ln2 = nn.LayerNorm(num_features=d_model, dtype=dtype, param_dtype=dtype, rngs=rngs) 

        # Stochastic regularisation 
        self.dropout1 = nn.Dropout(rate=dropout_rate, rngs=rngs)
        self.dropout2 = nn.Dropout(rate=dropout_rate, rngs=rngs)

    
    def __call__(self, x:jax.Array, mask:jax.Array|None=None, deterministic:bool=False) -> jax.Array:
        """ Put deterministic=True during eval/val/inference to disable dropout """
        # sublayer 1: Multi Head Self Attention (pre-LN and residual connection) 
        norm_x = self.ln1(x); attn_out = self.mha(norm_x, mask=mask, deterministic=deterministic); attn_out = self.dropout1(attn_out, deterministic=deterministic)
        x = x + attn_out # residual connection

        # sublayer 2: Positional wise feed forward network 
        norm_x = self.ln2(x); ffn_out = self.ffn(norm_x) ; ffn_out = self.dropout2(ffn_out, deterministic=deterministic)
        x = x + ffn_out # residual connection 
        return x 

class TransformerEncoder(nn.Module):
    """
    Complete N layer transformer encoder stack.
    Maps discrete (B,L) token indices to continuous (B, L, d_model) contextual representations
    """
    def __init__(self, vocab_size:int, d_model:int, n_heads:int, d_ff:int, n_layers:int, max_len:int, dropout_rate:float, rngs:nn.Rngs, dtype:jnp.dtype=jnp.bfloat16): 
        super().__init__() 

        # Input Embeddings 
        self.embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_len, dropout_rate=dropout_rate, rngs=rngs, dtype=dtype) 

        # Deep stack of N independent encoder blocks
        # Using a standard list comprehension cleanly instantiates N distinct parameter states 
        # XLA will unroll this seamlessly during compilation
        self.layers = nn.List([EncoderBlock(d_model=d_model, n_heads=n_heads, d_ffn=d_ff, dropout_rate=dropout_rate, rngs=rngs, dtype=dtype)
                       for _ in range(n_layers)])
        # Final Layer Norm Projection 
        self.final_norm = nn.LayerNorm(num_features=d_model, dtype=dtype, param_dtype=dtype, rngs=rngs)

    def __call__(self, x:jax.Array, padding_mask:jax.Array|None=None, deterministic:bool=False) -> jax.Array:
        """
        x: (batch, seq_length) integer tokens 
        padding_mask: (Batch, 1, 1, seq_length) shape boolean mask to ignore [PAD] tokens
        deterministic: bool = True during inference/eval to disable dropouts
        """
        # project tokens to continuous space + positional encodings 
        hidden_states = self.embedding(x, deterministic=deterministic) 
        # iteratively refine representations through the block sequence 
        for block in self.layers: hidden_states = block(hidden_states, mask=padding_mask, deterministic=deterministic) 
        # Final geometry stabilization
        return self.final_norm(hidden_states)

class DecoderBlock(nn.Module): 
    """
    Single Block of transformer decoder. Final block will have 6 of these stacked.

    Features (implemented from attention is all you need paper):
        1. Masked Multi Head Attention: Attend to previous tokens in target sequence (prevents look ahead) 
        2. Cross Attention: Attends to the encoder's output to mix with decoder 
        3. Feed Forward Network: Expands and contracts dimensions for non-;linear feature transformations  
    
    NOTE: we used pre-layer norm in this implementation (as opposed to post-layer norm in the original paper). This is known to be stable for training deep transformers than post-layer norm.
    """
    def __init__(self, d_model:int, n_heads:int, d_ffn:int, rngs:nn.Rngs,dropout:float=0.1, dtype:jnp.dtype=jnp.bfloat16):
        super().__init__() 
        # SUBLAYER 1: Masked Multi Head Self Attention (causal mask)
        self.masked_mha = CausalMultiHeadAttention(d_model=d_model, num_heads=n_heads, rngs=rngs, dropout_rate=dropout, dtype=dtype) 
        self.ln1 = nn.LayerNorm(num_features=d_model, dtype=dtype, param_dtype=dtype, rngs=rngs) 
        self.dropout1 = nn.Dropout(rate=dropout, rngs=rngs) 

        # SUBLAYER 2: Cross Attention (Decoder-Encoder attention) 
        self.cross_mha = CrossAttention(d_model=d_model, num_heads = n_heads, rngs=rngs, dropout_rate=dropout, dtype=dtype) 
        self.ln2 = nn.LayerNorm(num_features=d_model, dtype=dtype, param_dtype=dtype, rngs=rngs)
        self.dropout2 = nn.Dropout(rate=dropout, rngs=rngs) 

        # SUBLAYER 3: Position wise feed forward network 
        self.ffn = PositionWiseFFN(d_model=d_model, d_ffn=d_ffn, rngs=rngs, dtype=dtype)
        self.ln3 = nn.LayerNorm(num_features=d_model, dtype=dtype, param_dtype=dtype, rngs=rngs)
        self.dropout3 = nn.Dropout(rate=dropout, rngs=rngs) 

    def __call__(self, x:jax.Array, encoder_output:jax.Array, target_padding_mask:jax.Array|None=None, source_padding_mask:jax.Array|None=None, deterministic:bool=False) -> jax.Array:
        """
        Args: 
            x: target seq embeddings/prev layer output (batch, target_seq_len, d_model) 
            encoder_output: Encoder stack final context (batch, source_seq_len, d_model) 
            target_padding_mask: Mask for target sequence (batch, 1, 1, target_seq_len) 
            source_padding_mask: Mask for source sequence (used in cross attention ) (batch, 1, 1, source_seq_len) 
            deterministic: bool = True during inference/eval to disable dropouts
        """ 

        # SUBLAYER 1: Masked multi head self attention
        # purpose: allow the decoder to look at previous words in target sequence it has generated 
        norm_x = self.ln1(x) # norm is applied to inputs so this is pre-layer norm
        attn_out = self.masked_mha(x=norm_x, padding_mask=target_padding_mask, deterministic=deterministic) 
        attn_out = self.dropout1(attn_out, deterministic=deterministic) 
        x = x + attn_out # residual connection

        # SUBLAYER 2: Cross Atention Forward pass 
        # purpose: allow the decoder to focus on relevant parts of the encoder output 
        # Q comes from decoder layer or embedding layer 
        # k,v comes from the encoder outpt 
        normx = self.ln2(x) 
        # in cross attn , x is query vector, enc o/p is key and value 
        attn_out = self.cross_mha(x=norm_x, context=encoder_output, mask=source_padding_mask, deterministic=deterministic)
        attn_out = self.dropout2(attn_out, deterministic=deterministic)
        x = x + attn_out # residual connection

        # SUBLAYER 3: position wise feed forward layer 
        # purpose: apply additional non-linear transformations to every position independently, also increase dimensionality and reduce it 
        normx = self.ln3(x) 
        ffn_out = self.ffn(normx) 
        ffn_out = self.dropout3(ffn_out, deterministic=deterministic) 
        x = x + ffn_out # reidual connection of final sublayer 
        return x # shape: (batch, target_seq_len, d_model) 

# here i will implement full decoder (6 stacked blocks)
class TransformerDecoder(nn.Module):
    """
    Complete N Layer transformer decoder stack 
    Maps discrete (B, L_target) target token indices to continuous (B, L_tar, d_model) contextual representations . (translated sentence in simple words ;)
    """
    def __init__(self, vocab_size:int, d_model:int, n_heads:int, d_ffn:int, n_layers:int,max_len:int, dropout:float, rngs:nn.Rngs, dtype:jnp.dtype=jnp.bfloat16):
        super().__init__() 

        # target input embeddings + pos context 
        self.embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_len, dropout_rate=dropout, rngs=rngs, dtype=dtype) 

        # Deep stack of N independent decoder blocks (paper used N=6) 
        # nn.List natively tracks parameter states across the block sequence 
        self.layers = nn.List([ 
            DecoderBlock(d_model=d_model, n_heads=n_heads, d_ffn=d_ffn, rngs=rngs, dropout=dropout, dtype=dtype) 
            for _ in range(n_layers) 
        ])

        # FInal geometry stabilisation layer using Norm 
        self.final_norm = nn.LayerNorm(num_features=d_model, dtype=dtype, param_dtype=dtype, rngs=rngs) 

    def __call__(self, x:jax.Array, encoder_output:jax.Array, target_padding_mask:jax.Array|None=None, source_padding_mask:jax.Array|None=None, deterministic:bool=False) -> jax.Array:
        """
        x: (batch, target_seq_len) integer tokens representing right shifted target 
        encoder_output: (batch, source_seq_len, d_model) output from encoder stack 
        target_padding_mask: (batch, 1, 1, target_seq_len) causal mask for target sequence 
        source_padding_mask: (batch, 1, 1, source_seq_len) mask for source sequence 
        deterministic: bool = True during inference/eval to disable dropouts
        """

        # Embed discrete tokens in continuos space and apply pos embedding 
        hidden_states = self.embedding(x, deterministic=deterministic) 

        # padding through N decoder blocks 
        for block in self.layers:
            hidden_states = block(x=hidden_states, encoder_output=encoder_output, target_padding_mask=target_padding_mask, source_padding_mask=source_padding_mask, deterministic=deterministic)
        
        # final layer norm ouput
        return self.final_norm(hidden_states) # (batch, target_seq_len, d_model) 

class Transformer(nn.Module): 
    """
    Complete Sequence to Sequence Transformer Architecture (Attention is all you need paper) 
    Combined the Encoder and Decoder stack with final vocabulary projection layer 
    """
    def __init__(self, vocab_size:int, d_model:int, n_heads:int, d_ff:int, enc_n_layers:int, dec_n_layers:int, enc_max_len:int,dec_max_len:int, enc_dropout:float, dec_dropout:float, rngs:nn.Rngs, dtype:jnp.dtype=jnp.bfloat16) :
        super().__init__() 

        # ENCODER STACK 
        self.encoder = TransformerEncoder(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, d_ff=d_ff, n_layers=enc_n_layers, max_len=enc_max_len, dropout_rate=enc_dropout, rngs=rngs, dtype=dtype)

        # DECODER STACK 
        self.decoder = TransformerDecoder(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, d_ffn=d_ff, n_layers=dec_n_layers, max_len=dec_max_len, dropout=dec_dropout, rngs=rngs, dtype=dtype)

        # final generator linear layer 
        self.generator = nn.Linear(in_features=d_model, out_features=vocab_size, use_bias=False, dtype=dtype, param_dtype=dtype, rngs=rngs) 

    def __call__(self, source:jax.Array, target:jax.Array, source_padding_mask:jax.Array|None=None, target_padding_mask:jax.Array|None=None, deterministic:bool=False) -> jax.Array: 
        """
        Finally! Forward pass for the full transformer model. 
        
        Args: 
            source: (batch, source_seq_len) integer tokens representing source sequence 
            target: (batch, target_seq_len) integer tokens representing target sequence        
            source_padding_mask: (batch, 1, 1, source_seq_len), boolean mask. True=attend, False=ignore pad 
            target_padding_mask: (batch, 1, 1, target_seq_len), boolean mask . 
            deterministic: bool = True , during inference/eval to skip stochastics (dropouts)    

        Returns: 
            logits: (batch, target_seq_len, vocab_size)
            UnNormalized log probabilities over the vocabulary
        """
        # Encode the source sequence (batch, source_seq_len) -> (batch, source_seq_len, d_model) 
        encoded_output = self.encoder(x=source, padding_mask=source_padding_mask, deterministic=deterministic) 

        # Decode using target and encoder context: (batch. target_seq_len) + (batch, source_seq_len, d_model) -> (batch, target_seq_len, d_model) 
        decoded_output = self.decoder(x=target, encoder_output=encoded_output, target_padding_mask=target_padding_mask, source_padding_mask=source_padding_mask, deterministic=deterministic)

        # Final linear projection to project to vocabulary space: (batch, target_seq_len, d_model) -> (batch, target_seq_len, vocab_size) 
        logits = self.generator(decoded_output) # (batch, target_seq_len, vocab_size)
        return logits


###########################################################################
#                     MODULE TESTING FUNCTIONS
###########################################################################

# testing out Multi head attention 
def test_mha():
    # static architecture parameters 
    B = 32 # batch_size, 
    L = 128 # max_seq_len 
    d_model=512 # paper replica
    n_heads=8  # paper replica 

    # random state maneger 
    rng = nn.Rngs(744)

    # initiate the model 
    mha = MultiHeadSelfAttention(
        d_model=d_model, 
        num_heads=n_heads, 
        rngs=rng,
        dtype=jnp.bfloat16 # Maximize RTX 3060 throughput
    )

    # dummy data simulating embedding layer output 
    key = jax.random.PRNGKey(seed=0) 
    dummy_x = jax.random.normal(key, (B, L, d_model), dtype=jnp.bfloat16) 

    # dummy mask to test decoder attention
    # Brodcast across batch and heads 
    # Lower triangular matrix of boolean, (true allows attention , false masks it) 
    # shape : (1, 1, L, L) 
    causal_mask = jnp.tril(jnp.ones((1, 1, L, L), dtype=jnp.bool_))

    # forward pass 
    print("--- Multi-Head Attention Forward Pass ---")
    print(f"Input shape (B, L, d_model): {dummy_x.shape} | dtype: {dummy_x.dtype}")
    print(f"Mask shape: {causal_mask.shape}")
    output = mha(dummy_x, mask=causal_mask)
    
    print(f"Output shape (B, L, d_model): {output.shape} | dtype: {output.dtype}")
    print(f"Output device: {output.device}")
    
    # Assertions to ensure strict mathematical invariants
    assert output.shape == dummy_x.shape, "Output geometry must match input geometry"
    assert output.dtype == jnp.bfloat16, "Precision leaked during compute"
    print("All architectural invariants passed.")

def test_encoder_block() -> nn.Module:
    B:int = 32 
    L:int = 128 
    d_model:int = 512 
    n_heads:int = 8 
    d_ffn:int = 2048 # Standard expansion factor of 4x from 512
    dropout_rate:float = 0.1 

    # initialize NNX random state manager with specific hardware streams 
    # 'params' governs weight initialisation 
    # 'dropout' governs the stochastic masking in dropout
    rngs = nn.Rngs(params=42, dropout=42) 
    
    # initialise the encoder block 
    encoder_block = EncoderBlock(d_model=d_model, n_heads=n_heads, d_ffn=d_ffn, dropout_rate=dropout_rate, rngs=rngs, dtype=jnp.bfloat16)

    # creating dummy input simulating the output of embedding layer 
    key = jax.random.PRNGKey(0) 
    dummy_x = jax.random.uniform(key, (B, L, d_model), dtype=jnp.bfloat16)

    print("--- Encoder Block Forward Pass ---")
    print(f"Input shape (B, L, d_model): {dummy_x.shape} | dtype: {dummy_x.dtype}")
    
    # 5. Forward pass (Training mode: deterministic=False)
    output_train = encoder_block(dummy_x, deterministic=False)
    
    print(f"Output shape (B, L, d_model): {output_train.shape} | dtype: {output_train.dtype}")
    
    assert output_train.shape == dummy_x.shape, "Output geometry must match input geometry"
    assert output_train.dtype == jnp.bfloat16, "Precision leaked during compute"
    print("Encoder block passed dimensional and precision invariants."); 
    state=nn.state(encoder_block, nn.Param)
    num_params=sum(leaf.size for leaf in jax.tree_util.tree_leaves(state))
    num_bytes=sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(state))
    print(f"Encoder Block:\nTotal trainable parameters: {num_params:,}")
    print(f"Memory size ≈ {num_bytes / 1e6:.2f} MiB (float32)")
    
    return encoder_block

# testing embedding layer 
def test_embedding_layer():
    B:int = 32 
    L:int = 128 
    d_model:int = 512 
    vocab_size:int = 32_000 # Matching the BPE tokeniser vocab we trained 

    # intialise nnx state with params and dropout streams 
    rngs = nn.Rngs(dropout=42, params=42) 

    # instantiate the embedding sub system 
    embed_layer = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model, max_seq_len=1024, dropout_rate=0.1, rngs=rngs, dtype=jnp.bfloat16)
    
    # dummy input of integer token ids simulating jax dataloader output
    key = jax.random.PRNGKey(0) 
    # token ids must strict be integers in range [0, vocab_size-1] 
    dummy_input_ids = jax.random.randint(key, shape=(B, L), minval=0, maxval=vocab_size) 
    print("--- Transformer Embedding Pass ---")
    print(f"Discrete Input shape (B, L): {dummy_input_ids.shape} | dtype: {dummy_input_ids.dtype}")
    
    # Forward pass mapping integers to continuous vectors
    output_embeddings = embed_layer(dummy_input_ids, deterministic=False)
    
    print(f"Continuous Output shape (B, L, d_model): {output_embeddings.shape} | dtype: {output_embeddings.dtype}")
    
    # Assert structural invariants
    assert output_embeddings.shape == (B, L, d_model), "Embedding failed to project to d_model manifold"
    assert output_embeddings.dtype == jnp.bfloat16, "Precision constraint violated"
    print("Embedding projection and symmetry breaking passed.")

def test_transformer_encoder():
    # Architecture definitions matching the original paper's base model 
    B:int = 1024
    L:int = 128
    d_model:int = 512 
    num_heads:int = 8
    d_ffn:int = 2048 
    num_layers:int = 16 # replicating Vaswani et. al 
    vocab_size:int = 32_000

    rngs = nn.Rngs(params=42, dropout=42) 

    # instantiate the full tower 
    encoder = TransformerEncoder(vocab_size=vocab_size, d_model=d_model, n_heads=num_heads, d_ff=d_ffn, n_layers=num_layers, max_len=1024, dropout_rate=0.1, rngs=rngs, type=jnp.bfloat16)

    # Generate dummy input tokens (batch=32, seq_len=128) 
    key = jax.random.PRNGKey(0) 
    input_ids = jax.random.randint(key, shape=(B, L), minval=0, maxval=vocab_size) 
    
    # PADDING MASK 
    # in our tokeniser 1 is pad token ID 
    # we create a boolean mask where True means "attend" and False means "ignore" 
    # shape needs to broadcast over (Batch, NumHeads, SeqLen, SeqLen) in the attention logic
    pad_id = 1 
    # Shape: (B, 1, 1, L) 
    padding_mask = (input_ids != pad_id)[:, jnp.newaxis, jnp.newaxis, :] 

    print("--- Full Transformer Encoder Pass ---")
    print(f"Discrete Input shape (B, L): {input_ids.shape}")
    print(f"Padding Mask shape: {padding_mask.shape}")
    
    # Execute the deep representation learning
    encoder_output = encoder(input_ids, padding_mask=padding_mask, deterministic=False)
    
    print(f"Contextualized Output shape (B, L, d_model): {encoder_output.shape} | dtype: {encoder_output.dtype}")
    state = nn.state(encoder, nn.Param) # returns state dict with only Param leaves
    num_params = sum(leaf.size for leaf in jax.tree_util.tree_leaves(state)) 
    num_bytes = sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(state))
    print(f"\nEncoder :\nTotal trainable parameters: {num_params:,}")
    print(f"Number of Encoder Layers: {num_layers}")
    print(f"Memory size ≈ {num_bytes / 1e6:.2f} MiB (float32)")
    
    assert encoder_output.shape == (B, L, d_model), "Encoder tower failed to maintain spatial/channel geometry."
    assert encoder_output.dtype == jnp.bfloat16, "Precision leaked during deep propagation."
    print("Encoder Tower is structurally flawless.")

# function to test causal multihead attention mechanism 
def test_causal_multihead_attention(): 
    """ 
    Small standalone function which tests causal multi head attention 
    used in decoder block of the transformer model. 
    This testing func, checks dtype consistencies, device placement and basic masking behavior
    """
    # hyperparameters 
    B:int = 16 # batch size for parallel comp
    L:int = 128  # max_seq_len (decoder) in our case same as encoder
    d_model:int = 512 # replicating attention paper 
    num_heads:int = 8 # from paper 
    dropout_rate:float = 0.1 

    rngs=nn.Rngs(params=744, dropout=744) 

    causal_mha = CausalMultiHeadAttention(d_model=d_model, num_heads=num_heads, rngs=rngs, dropout_rate=dropout_rate, dtype=jnp.bfloat16) 

    # creating dummy input to simulate encoder output (input for decoder) 
    key = jax.random.PRNGKey(744) 
    dummy_x = jax.random.normal(key, shape=(B, L, d_model), dtype=jnp.bfloat16) 
    
    # padding mask here 
    pad_tok_fraction:float = 0.15 # 15% tokens 
    padding_mask = jax.random.bernoulli(key, p=1.0-pad_tok_fraction, shape=(B,L))[:, None, None, :] # (B, 1, 1, L) -> True = attend

    print("=== Causal Multi-Head Attention Test ===")
    print(f"Input shape              : {dummy_x.shape} | dtype: {dummy_x.dtype}")
    print(f"Padding mask shape       : {padding_mask.shape}")
    print(f"Model dtype              : {causal_mha.mha.dtype}")
    print(f"Number of heads          : {num_heads}")
    print(f"d_k (head dimension)     : {d_model // num_heads}")

    #forward pass (dropout active ) 
    output_train = causal_mha(dummy_x, padding_mask=padding_mask, deterministic=False) 

    # forward pass - inference mode (dropout off ) 
    output_infer = causal_mha(dummy_x, padding_mask=padding_mask, deterministic=True) 

    # ─── Basic shape and type checks ──────────────────────────────────────────
    print("\nResults:")
    print(f"Output shape (train)     : {output_train.shape} | dtype: {output_train.dtype}")
    print(f"Output shape (infer)     : {output_infer.shape} | dtype: {output_infer.dtype}")

    assert output_train.shape == (B, L, d_model), "Shape mismatch in training mode"
    assert output_infer.shape == (B, L, d_model), "Shape mismatch in inference mode"
    assert output_train.dtype == jnp.bfloat16, "Dtype leak in training mode"
    assert output_infer.dtype == jnp.bfloat16, "Dtype leak in inference mode"

    # ─── Very basic correctness check: outputs should be different due to dropout ─
    diff_norm = jnp.mean(jnp.abs(output_train - output_infer))
    print(f"Mean absolute difference (train vs infer): {float(diff_norm):.6f}")
    print("(should be > 0 because dropout is active in train mode)")

    # ─── Count parameters ----
    state = nn.state(causal_mha, nn.Param)
    num_params = sum(leaf.size for leaf in jax.tree_util.tree_leaves(state))
    num_bytes  = sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(state))

    print(f"\nCausal MHA module parameters : {num_params:,}")
    print(f"Approx memory (bf16)         : {num_bytes / 1e6:.2f} MiB")

    print("\n→ All basic invariants passed.")
    print("   (shape, dtype, dropout behavior, parameter count)")

def test_cross_attention():
    B:int = 16
    L_q:int = 64 # target seq len
    L_kv:int = 128 # source seq len
    d_model:int = 512
    num_heads:int = 8
    dropout_rate:float = 0.1

    rngs = nn.Rngs(params=744, dropout=744)
    cross_attn = CrossAttention(d_model=d_model, num_heads=num_heads, rngs=rngs, dropout_rate=dropout_rate, dtype=jnp.bfloat16)

    key = jax.random.PRNGKey(744)
    dummy_q = jax.random.normal(key, shape=(B, L_q, d_model), dtype=jnp.bfloat16)
    dummy_kv = jax.random.normal(key, shape=(B, L_kv, d_model), dtype=jnp.bfloat16)

    pad_tok_fraction:float = 0.15
    padding_mask = jax.random.bernoulli(key, p=1.0-pad_tok_fraction, shape=(B, L_kv))[:, None, None, :]

    print("=== Cross-Attention Test ===")
    print(f"Decoder Search Shape (Q) : {dummy_q.shape}")
    print(f"Encoder Context Shape (KV) : {dummy_kv.shape}")
    print(f"Padding Mask shape       : {padding_mask.shape}")

    output_train = cross_attn(dummy_q, dummy_kv, mask=padding_mask, deterministic=False)
    output_infer = cross_attn(dummy_q, dummy_kv, mask=padding_mask, deterministic=True)

    print("\nResults:")
    print(f"Output shape (train)     : {output_train.shape} | dtype: {output_train.dtype}")
    print(f"Output shape (infer)     : {output_infer.shape} | dtype: {output_infer.dtype}")

    assert output_train.shape == (B, L_q, d_model), "Shape mismatch in training mode"
    assert output_infer.shape == (B, L_q, d_model), "Shape mismatch in inference mode"

    state = nn.state(cross_attn, nn.Param)
    num_params = sum(leaf.size for leaf in jax.tree_util.tree_leaves(state))
    print(f"\nCross-Attention parameters : {num_params:,}")
    print("→ All basic invariants passed.")

def test_decoder_block(): 
    # Hyperparameters 
    batch:int = 16 
    target_seq_len:int = 128 
    source_seq_len:int = 128
    d_model:int = 512 
    n_heads:int = 8 
    d_ffn:int = 2048 
    dropout:float = 0.1 
    rngs = nn.Rngs(params=744, dropout=744)
    
    decoder_block = DecoderBlock(d_model=d_model, n_heads=n_heads, d_ffn=d_ffn, rngs=rngs, dropout=dropout, dtype=jnp.bfloat16)
    
    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4 = jax.random.split(key, 4)

    # dummy data 
    dummy_target_seq = jax.random.uniform(key1, (batch, target_seq_len, d_model), dtype=jnp.bfloat16) 
    dummy_encoder_out = jax.random.uniform(key2, (batch, target_seq_len, d_model) , dtype=jnp.bfloat16)  

    # simulate padding masks 
    target_padding_frac:float=0.25
    encoder_padding_frac:float=0.25 

    target_padding_mask = jax.random.bernoulli(key3, p = 1.0 - target_padding_frac, shape=(batch, target_seq_len))[:, None, None, :] 
    encoder_padding_mask = jax.random.bernoulli(key4, p = 1.0-encoder_padding_frac, shape=(batch, source_seq_len))[:, None, None, :] 

    print("=== Decoder Block Forward Pass ===")
    print(f"Target Input shape (B, L_tar, d_model): {dummy_target_seq.shape} | dtype : {dummy_target_seq.dtype}") 
    print(f"Encoder Output shape (B, L_src, d_model) : {dummy_encoder_out.shape}") 
    print(f"Target padding mask shape: {target_padding_mask.shape}")
    print(f"Source padding mask shape: {encoder_padding_mask.shape}")

    # execute forward pass 
    output_train = decoder_block(x=dummy_target_seq, encoder_output=dummy_encoder_out, target_padding_mask=target_padding_mask, source_padding_mask=encoder_padding_mask, deterministic=False)
    print(f"\nOutput Shape (B, L_tar, d_model): {output_train.shape} | dtype: {output_train.dtype}")

    # run assertions for testing 
    assert output_train.shape == dummy_target_seq.shape, "Output geometry must match input target geometry" 
    assert output_train.dtype == jnp.bfloat16, "Precision leaked during compute" 

    # Parameter count 
    state = nn.state(decoder_block, nn.Param) 
    num_params = sum(leaf.size for leaf in jax.tree_util.tree_leaves(state)) 
    print(f"Decoder Block | Total trainable parameters: {num_params:,}") 
    print("-> Decoder block passed dimensional and precision invariants")

def test_transformer_decoder():

    # parameters 
    vocab_size:int = 32_000 
    d_model:int = 512 
    n_heads:int = 8 
    d_ffn:int = 2048 
    n_layers:int = 16 
    dropout:float = 0.1 
    batch:int = 1024  
    target_seq_len:int = 128 
    source_seq_len:int = 128 
    
    rngs = nn.Rngs(params=744, dropout=744) 

    # initialize full decoder tower of 6 decoder blocks 
    decoder = TransformerDecoder(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, d_ffn=d_ffn, n_layers=n_layers, max_len=target_seq_len,dropout=dropout, rngs=rngs, dtype=jnp.bfloat16)

    # generate dummy target input tokens (batch=16, seq_len = 128 )
    key = jax.random.PRNGKey(0) 
    key_tgt, key_enc, key_mask = jax.random.split(key, 3)
    
    # dummy target tokens
    target_ids = jax.random.randint(key_tgt, shape=(batch, target_seq_len), minval=0, maxval=vocab_size, dtype=jnp.int32)

    # generate dummy encoder output (batch=16, seq_len=128, d_model=512)
    dummy_encoder_out = jax.random.normal(key_enc, shape=(batch, source_seq_len, d_model), dtype=jnp.bfloat16)

    #  generating padding masks 
    pad_id:int = 1 
    # target padding mask (b, 1, 1, L_target)
    target_padding_mask = (target_ids != pad_id)[:, jnp.newaxis, jnp.newaxis, :]

    # source padding mask simulaiting 12% pads in original incoming source tokens 
    source_padding_mask = jax.random.bernoulli(key_mask, p=1-0.12, shape=(batch, source_seq_len))[:, jnp.newaxis, jnp.newaxis, :] # (batch, 1, 1, target_seq_len)

    print("=== Full Transformer Decodder Pass ===")
    print(f"Discrete Target Input Shape (B, L_tar) : {target_ids.shape}") 
    print(f"Encoder OUtput shape (B, L_src, d_model) : {dummy_encoder_out.shape}") 
    print(f"Target padding mask shape: {target_padding_mask.shape}")
    print(f"Source padding mask shape: {source_padding_mask.shape}") 

    # propogate the computation through the decoder stack 
    decoder_output = decoder(x=target_ids, encoder_output=dummy_encoder_out, target_padding_mask=target_padding_mask, source_padding_mask=source_padding_mask, deterministic=False)
    print(f"Contextualized Decoder Output Shape (B, L_tar, d_model) : {decoder_output.shape} | dtype : {decoder_output.dtype}")

    # getting parameter counts 
    state = nn.state(decoder, nn.Param)
    num_params = sum(leaf.size for leaf in jax.tree_util.tree_leaves(state))
    num_bytes = sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(state))
    num_megabytes = num_bytes / 1024 / 1024

    print(f"\nDecoder :\n Total Trainable Parameters: {num_params:,}") 
    print(f"Number of Decoder Layers: {n_layers}") 
    print(f"Memory Size (MB): {num_megabytes:.2f}")
    
    # assertions and bound checks 
    assert decoder_output.shape == (batch, target_seq_len, d_model), "Decoder tower failed to maintain exact spatial/channel geometry constraints." 
    assert decoder_output.dtype == jnp.bfloat16, "Precision unexpectedly leaked during deep propogation through 6 decoder blocks" 
    print("-> Decoder Tower computations works perfectly finee and flawless.")

def TEST_TRANSFORMER(): 
    """Final transformer block testing"""
    
    # parameters 
    batch:int = 32
    vocab_size:int = 32_000 
    d_model:int = 512 
    n_heads:int = 8 
    d_ff:int = 2048 
    pad_id:int = 1 
    enc_n_layers:int = 6 
    dec_n_layers:int = 6 
    enc_max_len:int = 1024 
    dec_max_len:int = 1024 
    enc_dropout:float = 0.1 
    dec_dropout:float = 0.1 
    rngs = nn.Rngs(params=744, dropout=744) 

    # initialising full model 
    transformer = Transformer(
        vocab_size=vocab_size, 
        d_model=d_model, 
        n_heads=n_heads, 
        d_ff=d_ff, 
        enc_n_layers=enc_n_layers, 
        dec_n_layers=dec_n_layers, 
        enc_max_len=enc_max_len, 
        dec_max_len=dec_max_len, 
        enc_dropout=enc_dropout, 
        dec_dropout=dec_dropout, 
        rngs=rngs, 
        dtype=jnp.bfloat16
    )

    # generate dummy input data 
    key = jax.random.PRNGKey(744) 
    key_src, key_tgt, key_mask = jax.random.split(key, 3)

    # source tokens (english)
    source_ids = jax.random.randint(
        key = key_src, 
        shape = (batch, enc_max_len), 
        minval = 0, 
        maxval = vocab_size, 
        dtype = jnp.int32
    )

    # target tokens (hindi)
    target_ids = jax.random.randint(
        key = key_tgt, 
        shape = (batch, dec_max_len), 
        minval = 0, 
        maxval = vocab_size, 
        dtype = jnp.int32
    )


    # generating padding masks (pad_id = 1)
    source_padding_mask = (source_ids != pad_id)[:, jnp.newaxis, jnp.newaxis, :] # shape: (batch, 1, 1, enc_max_len)
    target_padding_mask = (target_ids != pad_id)[:, jnp.newaxis, jnp.newaxis, :] # shape: (batch, 1, 1, dec_max_len)

    print("=== Complete Transformer Pass ===")
    print(f"Source Input Shape (B, max_seq_len) : {source_ids.shape}")
    print(f"Target Input Shape (B, max_seq_len) : {target_ids.shape}") 
    print(f"Source padding mask shape : {source_padding_mask.shape}")
    print(f"Target padding mask shape : {target_padding_mask.shape}")

    # complete forward pass 
    # this outputs raw non-normalized logits representing likelihood array of next-token dictionary matches 
    logits = transformer(
        source = source_ids, 
        target = target_ids, 
        source_padding_mask = source_padding_mask, 
        target_padding_mask = target_padding_mask, 
        deterministic = False # since this is not an inference run and we need dropout
    )

    print(f"Final Logits Output (batch, max_seq_len, vocab_size) : {logits.shape} | dtype : {logits.dtype}")
    
    # parameter stats of this model 
    state = nn.state(transformer, nn.Param)
    num_params = sum(leaf.size for leaf in jax.tree_util.tree_leaves(state))
    num_bytes = sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(state))
    num_megabytes = num_bytes / 1024 / 1024

    print(f"\nFull Transformer Model:")
    print(f"Total Trainable Parameter             : {num_params:,}")
    print(f"Memory Size (BF16)                    : {num_megabytes:2f} MB")

    # assertions and bound checks 
    assert logits.shape == (batch, dec_max_len, vocab_size), "Final Logits failed to map to vocabulary space"
    assert logits.dtype == jnp.bfloat16, "Precision constraint violated in Generator Projection."
    print("-> Full Transformer Architecture is functionally flawless and dimensionally consistent.")


def test_all_block_sequentially():
    prettify = lambda : print("*"*50)
    # print("Testing Embedding Layer"); test_embedding_layer(); prettify()
    # print("Testing multi head attention"); test_mha(); prettify()
    # print("Testing Causal Multi-Head Attention"); test_causal_multihead_attention(); prettify()
    # print("Testing Cross-Attention"); test_cross_attention(); prettify()
    
    # # Encoder testing code 
    # print("Testing Encoder Block"); test_encoder_block(); prettify()
    # print("Testing Full Encoder"); test_transformer_encoder(); prettify()

    # # Decoder testing code
    # print("Testing Decoder Block"); test_decoder_block(); prettify()
    # print("Testing Full Decoder"); test_transformer_decoder(); prettify()

    # Final transformer testing code
    print("Testing Full Transformer"); TEST_TRANSFORMER(); prettify()

if __name__ =="__main__":
    # automatically tests all blocks
    test_all_block_sequentially()
