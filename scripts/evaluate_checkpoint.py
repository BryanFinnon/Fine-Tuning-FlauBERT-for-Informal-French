"""Evaluate a FlauBERT checkpoint on the repository's STS test set."""

import argparse
import csv

import torch
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model ID or checkpoint path")
    parser.add_argument("--dataset", default="data/STS_dataset.csv")
    parser.add_argument(
        "--mode",
        choices=("baseline", "regression"),
        required=True,
        help="Use CLS cosine similarity or a fine-tuned regression head",
    )
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def load_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))

    required = {"sentence1", "sentence2", "similarity_score"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Dataset must contain columns: {sorted(required)}")
    return rows


def evaluate_baseline(model_name, rows, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    predictions = []

    with torch.no_grad():
        for row in rows:
            embeddings = []
            for sentence in (row["sentence1"], row["sentence2"]):
                inputs = tokenizer(
                    sentence,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                embeddings.append(model(**inputs).last_hidden_state[:, 0, :])
            predictions.append(
                torch.nn.functional.cosine_similarity(
                    embeddings[0], embeddings[1]
                ).item()
            )
    return predictions


def evaluate_regression(model_name, rows, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    predictions = []

    with torch.no_grad():
        for row in rows:
            inputs = tokenizer(
                row["sentence1"],
                row["sentence2"],
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            score = model(**inputs).logits.squeeze().item()
            predictions.append(max(0.0, min(5.0, score)))
    return predictions


def main():
    args = parse_args()
    rows = load_rows(args.dataset)
    labels = [float(row["similarity_score"]) for row in rows]

    if args.mode == "baseline":
        predictions = evaluate_baseline(args.model, rows, args.max_length)
    else:
        predictions = evaluate_regression(args.model, rows, args.max_length)

    spearman = spearmanr(labels, predictions).statistic
    pearson = pearsonr(labels, predictions).statistic
    print(f"Pairs: {len(rows)}")
    print(f"Spearman: {spearman:.4f}")
    print(f"Pearson: {pearson:.4f}")


if __name__ == "__main__":
    main()
