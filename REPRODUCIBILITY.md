# Reproducing the reported evaluation

The saved output in `notebooks/main_work.ipynb` records the following evaluation on the 503 pairs in `data/STS_dataset.csv`:

| Configuration | Spearman | Pearson |
|---|---:|---:|
| `flaubert/flaubert_base_cased`, CLS cosine similarity | 0.3169 | 0.2611 |
| Fine-tuned regression checkpoint, clipped to 0–5 | 0.8420 | 0.7676 |

These are correlation metrics, not classification accuracy. The fine-tuned checkpoint is not stored in this repository, so the second result cannot be independently reproduced from the repository alone.

## Evaluation commands

Install the declared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Re-run the public baseline:

```bash
python scripts/evaluate_checkpoint.py \
  --mode baseline \
  --model flaubert/flaubert_base_cased
```

Evaluate a compatible fine-tuned checkpoint:

```bash
python scripts/evaluate_checkpoint.py \
  --mode regression \
  --model /path/to/flaubert_informel
```

Both commands use the same dataset, maximum sequence length of 128 and metric implementation. Full reproduction of training additionally requires publishing the original checkpoint and a clean training script with pinned dependency versions.
