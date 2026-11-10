"""Load the locally fine-tuned transformer and return three-class scores."""

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TransformerClassifier:
    def __init__(self, model_dir):
        path = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        self.model.eval()
        torch.set_num_threads(min(torch.get_num_threads(), 6))

    @torch.no_grad()
    def predict_proba(self, texts, batch_size=64):
        results = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[start:start + batch_size], truncation=True,
                                   padding=True, max_length=96, return_tensors="pt")
            results.append(self.model(**batch).logits.softmax(dim=1).cpu().numpy())
        return np.vstack(results)
