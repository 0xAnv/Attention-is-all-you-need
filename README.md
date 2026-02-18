
# Transformer Implementation from Scratch

A JAX-based implementation of the "Attention Is All You Need" paper on the IITB Hindi-English parallel corpus.

## Overview

This project implements a complete Transformer architecture in JAX for machine translation between English and Hindi. It includes data preprocessing, tokenization, and model training pipeline.

## Features

- **BPE Tokenization**: Joint vocabulary training on English-Hindi corpus (32K tokens)
- **Data Processing**: Automated cleaning, encoding, and static shape projection
- **Transformer Architecture**: Full attention-based seq2seq model
- **Efficient Pipeline**: Memory-efficient batch processing and caching

## Project Structure

```
.
├── data_config.py          # Data loading, preprocessing, and tokenization
├── model/                  # Transformer model implementation
├── data/                   # Raw IITB dataset
├── filtered_data/          # Cleaned dataset
├── encoded_data/           # Tokenized sequences
└── processed_data/         # Padded/truncated static shapes
```

## Data Pipeline

1. **Load**: IITB English-Hindi dataset from HuggingFace
2. **Filter**: Remove empty or whitespace-only translations
3. **Tokenize**: Train BPE on joint corpus (vocab size: 32K)
4. **Encode**: Convert text to integer sequences with BOS/EOS markers
5. **Normalize**: Pad/truncate to static length (128 tokens)

## Configuration

```python
VOCAB_SIZE = 32_000
MAX_SEQ_LEN = 128
BATCH_SIZE = 64_000
```

## Usage

```python
from data_config import run_data_config_pipeline

run_data_config_pipeline()
```

## Requirements

- JAX
- Hugging Face Datasets
- Tokenizers
- NumPy, Pandas

## References

Vaswani et al., "Attention Is All You Need" (2017)
