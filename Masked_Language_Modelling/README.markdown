# Masked Language Modeling with DistilBERT

This project demonstrates how to fine-tune a DistilBERT model for Masked Language Modeling (MLM) using the IMDB dataset. The model is trained to predict masked tokens in text, enabling it to learn contextual representations of words.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)

## Overview
This project uses the Hugging Face Transformers library to fine-tune a DistilBERT model (`distilbert-base-uncased`) for Masked Language Modeling on the IMDB dataset. The dataset contains movie reviews, and the model is trained to fill in masked tokens in text sequences.

## Dataset
The IMDB dataset is used, which contains:
- **Train**: 25,000 examples
- **Test**: 25,000 examples
- **Unsupervised**: 50,000 examples

Each example includes a `text` field (movie review) and a `label` field (positive or negative sentiment). For MLM, only the text is used, and labels are ignored.

## Installation
To run this project, install the required dependencies:

```bash
pip install transformers datasets torch
```

Optional: For GPU support, ensure PyTorch is installed with CUDA support.

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Log in to Hugging Face** (for pushing the model to the Hub):
   ```bash
   huggingface-cli login
   ```

3. **Run the notebook**:
   Open the Jupyter notebook (`train_mlm.ipynb`) in your environment (e.g., JupyterLab or VSCode) and execute the cells to preprocess data, train the model, and push it to the Hugging Face Hub.

## Training
The model is fine-tuned using the `Trainer` API from Hugging Face with the following setup:
- **Model**: `distilbert-base-uncased`
- **Epochs**: 3
- **Learning Rate**: 2e-5
- **Weight Decay**: 0.01
- **Batch Size**: 64
- **Evaluation Strategy**: Per epoch
- **Metric**: Perplexity (lower is better)
- **MLM Probability**: 0.15 (15% of tokens are masked)
- **Dataset**: Downsampled to 10,000 training and 1,000 test examples
- **Chunk Size**: 128 tokens (to handle long sequences)

The notebook tokenizes the dataset, splits long sequences into chunks, and trains the model. The fine-tuned model is saved to the Hugging Face Hub.

### Training Results
| Epoch | Training Loss | Validation Loss | Perplexity |
|-------|---------------|-----------------|------------|
| 1     | 2.6804        | 2.4932          | 21.94      |
| 2     | 2.5832        | 2.4480          | -          |
| 3     | 2.5255        | 2.4808          | 12.02      |

Perplexity decreased from 21.94 to 12.02 after training, indicating improved model performance.

## Inference
To use the fine-tuned model for masked language modeling, load it from the Hugging Face Hub:

```python
from transformers import pipeline

model_checkpoint = "praful-goel/distilbert-finetuned-mlm-imdb"
mask_filler = pipeline("fill-mask", model=model_checkpoint)

text = "This is a great [MASK]"
preds = mask_filler(text)

for pred in preds:
    print(f">>> {pred['sequence']}")
```

**Example Output**:
```python
>>> this is a great!
>>> this is a great.
>>> this is a great deal
>>> this is a great film
>>> this is a great movie
```