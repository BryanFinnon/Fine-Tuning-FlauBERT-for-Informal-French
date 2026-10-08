# FlauBERT for Informal French Similarity

An NLP research project evaluating whether task-specific fine-tuning improves FlauBERT's understanding of informal French, including slang, abbreviations and conversational phrasing.

## Result

| Model | Spearman correlation |
|---|---:|
| Baseline | 0.32 |
| Fine-tuned model | 0.81 |

These figures come from the evaluation setup contained in the project notebooks. They measure correlation, not classification accuracy, and should be interpreted within the supplied dataset and split.

## What the repository contains

- Data preparation and informal French corpora
- Fine-tuning and evaluation notebooks
- Spearman-based semantic similarity analysis
- Visualisation of experimental results
- Streamlit interface for a compatible fine-tuned checkpoint
- Full academic project report

## Technology

Python · PyTorch · Hugging Face Transformers · FlauBERT · scikit-learn · Streamlit · Jupyter

## Run the interface

Install the application dependencies, then provide a compatible local or Hugging Face checkpoint:

```bash
export MODEL_NAME=/path/to/fine-tuned-checkpoint
streamlit run app.py
```

The application deliberately has no base-model fallback: loading `flaubert/flaubert_base_cased` with a new regression head would produce untrained similarity scores.

## Repository structure

```text
app.py          Streamlit interface
notebooks/      preprocessing, training, evaluation and visualisation
data/           research datasets
assets/         project figures
Report/         final academic report
```

## Limitations

- The fine-tuned checkpoint is not published in this repository.
- Results depend on the included data and experimental split.
- Reproduction may vary with dependency versions and random seeds.
