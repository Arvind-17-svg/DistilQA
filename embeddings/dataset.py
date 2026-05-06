from datasets import load_dataset


LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}


def load_snli_clean(split="train"):
    """Load SNLI dataset and filter out invalid labels (label=-1)."""
    ds = load_dataset("stanfordnlp/snli", split=split)
    return ds.filter(lambda x: x["label"] != -1)


def label_to_text(label):
    """Convert label ID to text."""
    return LABEL_NAMES.get(label, "unknown")
