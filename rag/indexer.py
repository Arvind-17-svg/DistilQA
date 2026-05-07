import json
import os
import sys

import faiss
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.chunker import chunk_documents, load_squad_contexts
from rag.embedders import TrainedEmbedder


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "saved", "model", "trained_embedder.pt")
DEFAULT_INDEX_DIR = os.path.join(PROJECT_ROOT, "saved", "index")
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.json"


def build_index(chunks, embedder):
    """Encode chunks and return a FAISS IndexFlatIP.
    Embeddings are L2-normalized so inner product == cosine similarity."""
    embeddings = embedder.encode(chunks, normalize=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index, chunks, out_dir=DEFAULT_INDEX_DIR):
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, INDEX_FILE))
    with open(os.path.join(out_dir, CHUNKS_FILE), "w") as f:
        json.dump(chunks, f)
    print(f"Saved {index.ntotal} vectors to {out_dir}")


def load_index(out_dir=DEFAULT_INDEX_DIR):
    index = faiss.read_index(os.path.join(out_dir, INDEX_FILE))
    with open(os.path.join(out_dir, CHUNKS_FILE)) as f:
        chunks = json.load(f)
    return index, chunks


def build_squad_index(num_contexts=None, weights_path=DEFAULT_WEIGHTS,
                      out_dir=DEFAULT_INDEX_DIR):
    """End-to-end: SQuAD contexts -> chunks -> embeddings -> FAISS -> disk."""
    contexts = load_squad_contexts("train")
    if num_contexts is not None:
        contexts = contexts[:num_contexts]
    print(f"Loaded {len(contexts)} contexts")

    chunks = chunk_documents(contexts)
    print(f"Produced {len(chunks)} chunks")

    embedder = TrainedEmbedder(weights_path=weights_path)
    index = build_index(chunks, embedder)
    save_index(index, chunks, out_dir)
    return index, chunks


if __name__ == "__main__":
    # Default: index a 500-context subset for a quick first build.
    # Pass an integer arg to override (e.g. `python rag/indexer.py 2000`),
    # or `all` to use every SQuAD context.
    n = 500
    if len(sys.argv) > 1:
        n = None if sys.argv[1] == "all" else int(sys.argv[1])
    build_squad_index(num_contexts=n)
