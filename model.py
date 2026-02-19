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
#                     MULTI HEAD SELF ATTENTION
###########################################################################
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

    # forward pass for multi-head self attention
    def __call__(self, x:jax.Array, mask:jax.Array|None=None) -> jax.Array:
        """
        x: Input tensor of shape (batch_size, seq_length, d_model)
        mask: Optional boolean mask of shape (batch_size, 1, seq_length, seq_length) 
              for masking out padding tokens or future tokens in decoder
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

        # multiplying attention weights to V 
        # v : (B, num_heads, L, d_k) 
        # attention_weights : (B, num_heads, L, L)
        # attention_weights @ v^T shape: (B, num_heads, L, d_k) 
        context = jnp.einsum('bhqk, bhkd -> bhqd',attention_weights, v) 

        # recombining heads and flattening the last two dimensions 
        context = jnp.transpose(context, (0,2,1,3)).reshape(B, L, self.d_model)

        # final linear projections to get maximum context
        return self.out_proj(context)


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

if __name__ =="__main__":
    prettify = lambda : print("*"*50)
    print("testing multi head attention"); test_mha(); prettify()
