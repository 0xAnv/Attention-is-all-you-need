# generic imports 
import numpy as np
import pandas as pd 
from collections.abc import Iterator
from pathlib import Path 

# dataset handling
from datasets import DatasetDict, Dataset
from datasets import load_dataset, load_from_disk

# tokenisation 
from tokenizers import Tokenizer 
from tokenizers.models import BPE 
from tokenizers.pre_tokenizers import Whitespace 
from tokenizers.trainers import BpeTrainer 
from tokenizers.decoders import BPEDecoder 

# jax 
import jax 
import jax.numpy as jnp


from typing import Any
from tqdm.auto import tqdm 

# variables 
RAW_DATA_PATH:str = "data" 
FILTERED_DATA_PATH:str = "filtered_data"
ENCODED_DATA_PATH:str = "encoded_data"
PROCESSED_DATA_PATH:str = "processed_data"
BATCH_SIZE:int = 64_000
VOCAB_SIZE:int = 32_000
MAX_SEQ_LEN:int = 2**7 # 128 (computed for this dataset)


def load_iitb_data(datapath:str=RAW_DATA_PATH) -> DatasetDict:
    """Function loads iitb data from HF"""
    assert isinstance(datapath, str)
    if Path(datapath).exists(): return load_from_disk(datapath) #type:ignore
    else: 
        Path(datapath).mkdir(); print("Downloading iitb data")
        df = load_dataset("cfilt/iitb-english-hindi"); df.save_to_disk(datapath); print('data saved')
        return df

def filter_empty_translations(
        dataset_dict: DatasetDict, 
        src_lang: str = "en", 
        tgt_lang: str = "hi" ) -> DatasetDict:
    """
    Pure function to filter out anomalous rows where either the source 
    or target translation is missing or purely whitespace.
    """
    def _is_valid_pair(example: dict[str, dict[str, str]]) -> bool:
        translations = example.get("translation", {})
        src_text = translations.get(src_lang, "").strip()
        tgt_text = translations.get(tgt_lang, "").strip()
        
        # Both must have a non-zero length after stripping whitespace
        return bool(src_text) and bool(tgt_text)

    # dataset.filter returns a new DatasetDict, preserving immutability
    df = dataset_dict.filter(_is_valid_pair)
    # df.save_to_disk(FILTERED_DATA_PATH); print(f"Filtered data saved to {FILTERED_DATA_PATH}")
    return df

def generate_joint_corpus(
        dataset: Dataset, 
        batch_size:int=10_000,
        src_lan:str='en', 
        tgt_lan:str='hi'
    ) -> Iterator[list[str]] :

    """
    Pure generator yielding batches of text for memory-efficient BPE Training. 
    Extracts both languages to construct a unified embedding space.
    """
    # iterate over dataset in chunks to maintain memory footprint
    for i in range(0, len(dataset), batch_size):
        # we slice dataset directly, this is deterministic and avoids stateful cursors 
        batch = dataset[i : i+batch_size]['translation']
        # flatten source and target text into single list of string for this batch 
        yield [pair[lang] for pair in batch for lang in (src_lan, tgt_lan)]

# tokeniser function 
def build_tokeniser(
        corpus_iterator:Iterator[list[str]], 
        vocab_size:int=VOCAB_SIZE
    ) -> Tokenizer :

    """
    Constructs and train a BPE using a streaming corpus
    """
    def get_tokeniser_from_file(filename:str) -> Tokenizer: 
        print(f"Already trained tokeniser found at {filename=}")
        print("Skipping training and loading tokeniser.")
        tokeniser:Tokenizer = Tokenizer(BPE(unk_token='[UNK]', end_of_word_suffix="</w>"))
        tokeniser.pre_tokenizer = Whitespace()
        tokeniser.decoder=BPEDecoder(suffix="</w>")
        return tokeniser.from_file(filename)

    if Path(f"bpe_{vocab_size}.json").exists(): 
        return get_tokeniser_from_file(f'bpe_{vocab_size}.json')

    print("Training Tokeniser")
    tokeniser: Tokenizer = Tokenizer(BPE(unk_token='[UNK]', end_of_word_suffix="</w>"))
    tokeniser.pre_tokenizer = Whitespace()
    bpe_trainer: BpeTrainer = BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=['[UNK]', '[PAD]', '[BOS]', '[EOS]'], 
        end_of_word_suffix="</w>"
    )
    tokeniser.decoder = BPEDecoder(suffix="</w>")
    tokeniser.train_from_iterator(corpus_iterator, trainer=bpe_trainer)
    print(f"Tokeniser is trained for {vocab_size=}")
    tokeniser.save(f'bpe_{vocab_size}.json')
    return tokeniser

def encode_parallel_corpus(
            dataset_dict: DatasetDict, 
            tokeniser: Tokenizer, 
            src_lan:str="en", 
            tgt_lan:str="hi", 
            batch_size:int=10_000
    ) -> DatasetDict :

    """
    Pure Function that maps the arrow backed string corpus into discrete integer
    sequences, injecting casual boundary markers for seq2seq training.
    """
    # Fetch structural token IDs to anchor the temporal sequence
    bos_id = tokeniser.token_to_id('[BOS]')
    eos_id = tokeniser.token_to_id('[EOS]')

    if bos_id is None or eos_id is None: 
        raise ValueError('Tokeniser is missing [EOS] or [BOS] token in vocabulary')
    
    def _encode_batch(batch: dict[str, list[dict[str, str]]]) -> dict[str, list[list[int]]]:
        # vectorized extraction of text streams from the Arrow table chunk
        src_texts = [item[src_lan] for item in batch['translation']]
        tgt_texts = [item[tgt_lan] for item in batch['translation']] 

        # rust ffi call: encodes the entire batch simultaneously , bypassing the GIL 
        src_encodings = tokeniser.encode_batch(src_texts)
        tgt_encodings = tokeniser.encode_batch(tgt_texts)

        # Functional transformation: wrap the learned subwords in strict casual boundaries 
        return {
            "en" : [[bos_id] + enc.ids + [eos_id] for enc in src_encodings] , 
            "hi" : [[bos_id] + enc.ids + [eos_id] for enc in tgt_encodings]
        }

    # execute the map over the entire DatasetDict. We remove the string column to enforce strict memory hygiene 
    return dataset_dict.map(
        _encode_batch, 
        batched=True,
        batch_size=batch_size, 
        remove_columns=['translation'], 
        desc="Projecting text to integer sequences"
    )

# function to find what should be the max seq length of our data 
def compute_sequence_len_bounds(
        dataset: DatasetDict, 
        percentile:float=99.0
    ) -> dict[str, int] :
    
    """
    Function to compute optimal seq length boundary 
    by analysing asymptotic tail of length distribution across all splits 
    """
    
    def _extract_lengths(dataset_split: Dataset) -> list[int]:
        # lazily evaluate length of both target and source sequence 
        en_lengths = [len(seq) for seq in dataset_split['en']] 
        hi_lengths = [len(seq) for seq in dataset_split['hi']] 

        return en_lengths+hi_lengths
    
    # aggregate lengths functionally across train, test and validation splits 
    all_lengths : list[int] = sum(
     (_extract_lengths(dataset[split]) for split in dataset.keys()), []
    )

    # cast to contiguous memory block for fast C-level statistical computation
    lengths_array = np.array(all_lengths, dtype=np.int32) 

    # compute bounds 
    p_val = int(np.percentile(lengths_array, percentile)) 
    max_val = int(np.max(lengths_array))

    print(f"Absolute Maximum Sequence Length: {max_val}")
    print(f"Optimal {percentile}th Percentile Bound: {p_val}")
    
    return {"p_val": p_val, "max_val": max_val}

def project_to_static_shape(
        dataset_dict: DatasetDict, 
        tokeniser:Tokenizer, 
        max_seq_len:int=128, 
        save_path:str = "processed_data"
    ) -> DatasetDict:
    """
    Function to enforce static shapes via truncation/padding
    Exporting to contiguios array of (N, 2, L) where N=no of sentences, L=max length of each sentence
    """
    pad_id:int=tokeniser.token_to_id('[PAD]')
    eos_id:int=tokeniser.token_to_id('[EOS]') 

    if pad_id is None or eos_id is None: raise ValueError("Tokeniser must contain [PAD] and [EOS] tokens in vocab") 

    def _pad_and_truncate(batch:dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        """Function to pad and truncate"""

        # linear mapping list to fit seq_len
        def _process_seq(seq:list[int]) -> list[int] :
            seq_len:int = len(seq)
            if seq_len>max_seq_len:
                # truncate and enforce the temporal halt state '[EOS]' 
                truncated = seq[:max_seq_len]
                truncated[-1] = eos_id
                return truncated
            # pad to static dimension boundary 
            return seq + [pad_id] * (max_seq_len-seq_len)

        return {
            "en" : [_process_seq(seq) for seq in batch['en']], 
            "hi" : [_process_seq(seq) for seq in batch['hi']]
        }
        
    # applying padding and truncation functionally across dataset 
    static_dataset = dataset_dict.map(
        _pad_and_truncate, 
        batched=True, 
        desc=f"Enforcing static shape L={max_seq_len}"
    )
    static_dataset.save_to_disk(save_path)
    print(f'Saved dataset to: {save_path=}')
    return static_dataset

def build_jax_dataloader(
        dataset:Dataset, 
        batch_size:int,
        prng_key:jax.Array, 
        drop_last:bool=True
    ) -> Iterator[np.ndarray] :
    """ Generator to yield batches of data for jax """

    dataset_len:int = len(dataset)
    # we create shuffled indices for out dataset using jax backend ("cuda")
    shuffled_indices = jax.random.permutation(prng_key, jnp.arange(dataset_len))
    # cast back to numpy array to avoid host memory transfers in each epoch of python iteration 
    shuffled_indices = np.array(shuffled_indices)

    # boundary for dropping last batch or not 
    num_batches:int = dataset_len // batch_size
    if not drop_last and dataset_len % batch_size != 0: num_batches += 1

    # streaming data for iteration in batches
    for i in range(num_batches):
        start_idx:int = i * batch_size 
        end_idx:int = min((i+1) * batch_size, dataset_len)

        # slicing the index shuffled array 
        batch_indices = shuffled_indices[start_idx:end_idx]
        batch_data = dataset[batch_indices.tolist()] # select the batch data using the shuffled indices

        # project python list to contiguous numpy array 
        en_batch = np.array(batch_data['en'], dtype=np.int32)
        hi_batch = np.array(batch_data['hi'], dtype=np.int32)

        # stacking and returning to shape (BatchSize, 2, max_seq_len)
        yield np.stack([en_batch, hi_batch], axis=0)

def run_data_config_pipeline(data_iterator:bool=False, batch_size:int=64) -> None | Iterator[np.ndarray] :

    prettify = lambda: print("*"*70)
    #loading iitb data 
    df:DatasetDict = load_iitb_data(datapath=RAW_DATA_PATH)
    print("Data loaded successfully. Sample data:")
    print(df["train"][0]); prettify()  # Print first sample from training set

    # cleaning data (removing empty translations)
    df = filter_empty_translations(df)
    print("Data cleaned successfully. Sample data after cleaning:")
    print(df["train"][0]); prettify()  # Print first sample from training set after cleaning

    # training BPE tokenizer on joint corpus
    print("Generating joint corpus for BPE training...")
    train_corpus_iterator:Iterator[list[str]] = generate_joint_corpus(df["train"],batch_size=BATCH_SIZE)
    print("Joint corpus generated successfully. Sample batch:")
    print(next(train_corpus_iterator)[:10]); prettify()  # Print first 10 sentences from the first batch of the joint corpus

    # Training tokeniser and loading from corpus iterator 
    tokeniser:Tokenizer = build_tokeniser(train_corpus_iterator, vocab_size=VOCAB_SIZE)
    print("Tokeniser built successfully. Sample tokenization:")
    sample_text = "Hello नमस्ते, I am an AI Engineer."
    print(f"Original text: {sample_text}")
    encoded = tokeniser.encode(sample_text)
    print(f"Encoded tokens: {encoded.tokens}")
    print(f"Encoded ids: {encoded.ids}")
    print(f"Decoded text: {tokeniser.decode(encoded.ids)}"); prettify()

    # Encoding dataset into token ids 
    if Path(ENCODED_DATA_PATH).exists(): 
        encoded_dataset=load_from_disk(ENCODED_DATA_PATH)
        print(f"Encoded dataset already exists at {ENCODED_DATA_PATH}, loading from disk.")
    
    else: 
        encoded_dataset = encode_parallel_corpus(df, tokeniser, batch_size=BATCH_SIZE)
        encoded_dataset.save_to_disk(ENCODED_DATA_PATH)
        print(f"Encoded dataset saved to {ENCODED_DATA_PATH}")

    print("Data encoding complete. Sample encoded data:")
    print(encoded_dataset["train"][0]); prettify()  # Print first sample from encoded training set

    p = [99.0, 99.9, 99.99]
    get_lengths_of_percentiles = lambda per, data: {p:compute_sequence_len_bounds(data, p)['p_val'] for p in per}
    # per_stats = get_lengths_of_percentiles(p, encoded_dataset) # this will take time to execute 
    # print(per_stats)
    print(
        """
        Absolute Maximum Sequence Length: 3083
        Optimal 99.0th Percentile Bound: 93
        Absolute Maximum Sequence Length: 3083
        Optimal 99.9th Percentile Bound: 175
        Absolute Maximum Sequence Length: 3083
        Optimal 99.99th Percentile Bound: 287
        {99.0: 93, 99.9: 175, 99.99: 287}
        """
    ); prettify()

    # now performing padding and truncation to project to static shape
    print(f"Cache cleared: {encoded_dataset.cleanup_cache_files()}")
    if not Path(PROCESSED_DATA_PATH).exists():
        jax_ready_data = project_to_static_shape(encoded_dataset, tokeniser, max_seq_len=MAX_SEQ_LEN, save_path=PROCESSED_DATA_PATH) #type:ignore
    else: 
        jax_ready_data = load_from_disk(PROCESSED_DATA_PATH)
        print(f"Processed dataset already exists at {PROCESSED_DATA_PATH}, loading from disk.")
    print(f"Data projected to static shape successfully. Sample processed data:")
    dummy_index:int=70404
    print(jax_ready_data["train"][dummy_index])  # Print first sample from processed training set
    print(f"Shape of first training sample (en, hi): ({len(jax_ready_data['train'][dummy_index]['en'])}, {len(jax_ready_data['train'][dummy_index]['hi'])})")
    print(f"Tokeniser decoded sample English text: {tokeniser.decode(jax_ready_data['train'][dummy_index]['en'])}")
    print(f"Tokeniser decoded sample Hindi text: {tokeniser.decode(jax_ready_data['train'][dummy_index]['hi'])}"); prettify()

    # getting jax data loader which loads batch of numpy array
    prng_key = jax.random.PRNGKey(744)
    dataloader = build_jax_dataloader(jax_ready_data['train'], batch_size=batch_size, prng_key=prng_key, drop_last=True) # type:ignore 
    print(f"JAX dataloader built successfully. Sample batch shape:") 
    sample_batch = next(dataloader)
    print(f"Shape of sample batch: {sample_batch.shape}")  # Should be (BatchSize, 2, max_seq_len)
    print(f"Sample batch (first sentence in English and Hindi):")
    print(f"English token IDs: {sample_batch[0,0,:10]}")  # Print first 10 token IDs of the first English sentence in the batch
    print(f"Hindi token IDs: {sample_batch[0,1,:10]}")  # Print first 10 token IDs of the first Hindi sentence in the batch
    print(f"Decoded English text from token IDs: {tokeniser.decode(sample_batch[0,0,:10].tolist())}")
    print(f"Decoded Hindi text from token IDs: {tokeniser.decode(sample_batch[1,0,:10].tolist())}"); prettify()

    if data_iterator: return dataloader
    else: return None


if __name__ == "__main__":
    run_data_config_pipeline()