import os
import sys
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.embedders import TrainedEmbedder
from rag.indexer import DEFAULT_INDEX_DIR, DEFAULT_WEIGHTS, load_index


@dataclass
class RetrievedChunk:
    text: str
    score: float
    rank: int


class Retriever:
    """Query a FAISS index of chunked passages.

    Loads the FAISS index + chunk texts from disk and the TrainedEmbedder
    once at construction; subsequent .retrieve() calls are cheap.
    """

    def __init__(
        self,
        index_dir: str = DEFAULT_INDEX_DIR,
        weights_path: str = DEFAULT_WEIGHTS,
        device: Optional[str] = None,
    ):
        self.index, self.chunks = load_index(index_dir)
        self.embedder = TrainedEmbedder(weights_path=weights_path, device=device)
        print(f"Retriever ready: {self.index.ntotal} vectors, "
              f"dim={self.index.d}")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """Return top-k chunks by cosine similarity to `query`.

        score_threshold: optional minimum cosine similarity in [-1, 1].
        Useful for filtering out junk hits when nothing relevant exists.
        """
        query_emb = self.embedder.encode([query], normalize=True)
        scores, indices = self.index.search(query_emb, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS sentinel for "fewer than k results"
                continue
            if score_threshold is not None and score < score_threshold:
                continue
            results.append(
                RetrievedChunk(text=self.chunks[idx], score=float(score), rank=rank)
            )
        return results

    def retrieve_batch(
        self,
        queries: List[str],
        k: int = 5,
    ) -> List[List[RetrievedChunk]]:
        """Batched retrieval — one FAISS call for all queries."""
        query_embs = self.embedder.encode(queries, normalize=True)
        scores, indices = self.index.search(query_embs, k)

        all_results = []
        for q_scores, q_indices in zip(scores, indices):
            results = []
            for rank, (score, idx) in enumerate(zip(q_scores, q_indices)):
                if idx == -1:
                    continue
                results.append(
                    RetrievedChunk(text=self.chunks[idx], score=float(score), rank=rank)
                )
            all_results.append(results)
        return all_results


def _format_hit(hit: RetrievedChunk, max_chars: int = 200) -> str:
    snippet = hit.text[:max_chars] + ("..." if len(hit.text) > max_chars else "")
    return f"[{hit.rank}] score={hit.score:.3f}\n  {snippet}"


if __name__ == "__main__":
    retriever = Retriever()
    queries = sys.argv[1:] or [
        "Who wrote the Declaration of Independence?",
        "What is the speed of light?",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        for hit in retriever.retrieve(q, k=3):
            print(_format_hit(hit))
