# Named Entity Recognition with BERT

This project demonstrates how to fine-tune a BERT model for Named Entity Recognition (NER) using the CoNLL-2003 dataset. The model is trained to identify entities such as persons, organizations, locations, and miscellaneous entities in text.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)

## Overview
This project uses the Hugging Face Transformers library to fine-tune a BERT model (`bert-base-cased`) for token classification on the CoNLL-2003 dataset. The dataset includes annotations for named entities, and the model is trained to predict these entities in new text inputs.

## Dataset
The CoNLL-2003 dataset is used, which contains:
- **Train**: 14,041 examples
- **Validation**: 3,250 examples
- **Test**: 3,453 examples

Each example includes tokens, POS tags, chunk tags, and NER tags with the following labels:
- `O`: Non-entity
- `B-PER`, `I-PER`: Person
- `B-ORG`, `I-ORG`: Organization
- `B-LOC`, `I-LOC`: Location
- `B-MISC`, `I-MISC`: Miscellaneous

## Installation
To run this project, install the required dependencies:

```bash
pip install transformers datasets evaluate numpy torch
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
   Open the Jupyter notebook (`Token_Classification.ipynb`) in your environment (e.g., JupyterLab or VSCode) and execute the cells to preprocess data, train the model, and push it to the Hugging Face Hub.

## Training
The model is fine-tuned using the `Trainer` API from Hugging Face with the following setup:
- **Model**: `bert-base-cased`
- **Epochs**: 3
- **Learning Rate**: 2e-5
- **Weight Decay**: 0.01
- **Evaluation Strategy**: Per epoch
- **Metrics**: Precision, Recall, F1, Accuracy (using `seqeval`)

The notebook processes the dataset, aligns labels with tokenized inputs, and trains the model. The fine-tuned model is saved to the Hugging Face Hub.

### Training Results
| Epoch | Training Loss | Validation Loss | Precision | Recall | F1     | Accuracy |
|-------|---------------|-----------------|-----------|--------|--------|----------|
| 1     | 0.0748        | 0.0606          | 0.9095    | 0.9372 | 0.9232 | 0.9828   |
| 2     | 0.0334        | 0.0652          | 0.9332    | 0.9482 | 0.9406 | 0.9860   |
| 3     | 0.0192        | 0.0614          | 0.9382    | 0.9507 | 0.9444 | 0.9870   |

## Inference
To use the fine-tuned model for inference, you can load it from the Hugging Face Hub:

```python
from transformers import pipeline

model_checkpoint = "praful-goel/bert-finetuned-ner"
finetuned_pipeline = pipeline("token-classification", model=model_checkpoint, aggregation_strategy="simple")

# Example
result = finetuned_pipeline("My name is Praful Prakash Goel and I study in IIIT Guwahati")
print(result)
```

**Example Output**:
```python
[
    {'entity_group': 'PER', 'score': 0.9981, 'word': 'Praful Prakash Goel', 'start': 11, 'end': 30},
    {'entity_group': 'ORG', 'score': 0.6514, 'word': 'IIIT', 'start': 46, 'end': 50},
    {'entity_group': 'LOC', 'score': 0.9334, 'word': 'Guwahati', 'start': 51, 'end': 59}
]
```
