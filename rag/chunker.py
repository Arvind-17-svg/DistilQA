from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_squad_contexts(split="train"):
    """Load SQuAD v2 and return a list of unique context passages.
    Multiple QA pairs share the same context, so we dedupe."""
    ds = load_dataset("rajpurkar/squad_v2", split=split)
    seen = set()
    contexts = []
    for ex in ds:
        c = ex["context"]
        if c not in seen:
            seen.add(c)
            contexts.append(c)
    return contexts


def make_splitter(chunk_size=500, chunk_overlap=50):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split a list of documents into a flat list of chunks."""
    splitter = make_splitter(chunk_size, chunk_overlap)
    chunks = []
    for doc in documents:
        chunks.extend(splitter.split_text(doc))
    return chunks


if __name__ == "__main__":
    contexts = load_squad_contexts("train")
    print(f"Loaded {len(contexts)} unique SQuAD contexts")
    chunks = chunk_documents(contexts)
    avg = sum(len(c) for c in chunks) / len(chunks)
    print(f"Produced {len(chunks)} chunks (avg {avg:.0f} chars)")
