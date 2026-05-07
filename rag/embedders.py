import os
import sys

import numpy as np
import torch

# Allow `python rag/embedders.py` style runs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embeddings.model import EmbeddingModel


class TrainedEmbedder:
    """Inference wrapper around the SNLI-fine-tuned EmbeddingModel.

    - Loads weights from `saved/model/trained_embedder.pt` if available;
      otherwise falls back to the base pretrained checkpoint.
    - L2-normalizes outputs so cosine similarity == dot product
      (lets us use FAISS IndexFlatIP).
    """

    def __init__(self, model_name="distilbert-base-uncased",
                 weights_path=None, device=None):
        self.model = EmbeddingModel(model_name, device=device)
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.model.device)
            self.model.model.load_state_dict(state)
            print(f"Loaded fine-tuned weights from {weights_path}")
        else:
            print(f"No fine-tuned weights at {weights_path} — using base {model_name}")
        self.model.model.eval()

    @torch.no_grad()
    def encode(self, texts, batch_size=32, normalize=True):
        if isinstance(texts, str):
            texts = [texts]
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = self.model.get_embedding(batch)
            if normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            all_embeddings.append(emb.cpu().numpy())
        return np.vstack(all_embeddings).astype("float32")
