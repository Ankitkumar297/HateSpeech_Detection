"""CPU-friendly BERT-tiny experiment using the official HateXplain splits.

Run with: py -3.12 -m ml_pipeline.train_transformer
The published test split is evaluated only after the best validation epoch is chosen.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml_pipeline.train import ROOT, load_hatexplain


MODEL_ID = "prajjwal1/bert-tiny"
OUTPUT = ROOT / "models" / "bert_tiny"
MAX_LENGTH = 96
EPOCHS = 4
BATCH_SIZE = 32


def make_loader(tokenizer, texts, labels, shuffle=False):
    encoded = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LENGTH,
                        return_tensors="pt")
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"],
                            torch.tensor(labels, dtype=torch.long))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    truth, predictions = [], []
    for ids, mask, labels in loader:
        logits = model(input_ids=ids, attention_mask=mask).logits
        truth.extend(labels.tolist())
        predictions.extend(logits.argmax(dim=1).tolist())
    return {"accuracy": round(100 * accuracy_score(truth, predictions), 2),
            "macro_f1": round(f1_score(truth, predictions, average="macro"), 4)}


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(6)
    splits = load_hatexplain()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=3)
    train = make_loader(tokenizer, *splits["train"], shuffle=True)
    val = make_loader(tokenizer, *splits["val"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    best = -1
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for step, (ids, mask, labels) in enumerate(train, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % 100 == 0:
                print(f"epoch={epoch} step={step}/{len(train)} loss={running_loss/step:.4f}", flush=True)
        score = evaluate(model, val)
        history.append({"epoch": epoch, "train_loss": round(running_loss/len(train), 4), **score})
        print("validation", history[-1], flush=True)
        if score["macro_f1"] > best:
            best = score["macro_f1"]
            OUTPUT.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUTPUT)
            tokenizer.save_pretrained(OUTPUT)
    best_model = AutoModelForSequenceClassification.from_pretrained(OUTPUT)
    test = make_loader(tokenizer, *splits["test"])
    test_score = evaluate(best_model, test)
    results = {"base_model": MODEL_ID, "selection_metric": "validation macro_f1",
               "validation_history": history, "test": test_score}
    (OUTPUT / "experiment_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("held_out_test", test_score, flush=True)


if __name__ == "__main__":
    main()
