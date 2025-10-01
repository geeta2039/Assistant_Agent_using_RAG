EMBEDDING_MODELS = {
    "mini": "sentence-transformers/all-MiniLM-L6-v2",
    "qa_mini": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "mpnet": "sentence-transformers/all-mpnet-base-v2"
}

GENERATION_MODELS = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "distilgpt2": "distilgpt2",
    "opt": "facebook/opt-125m",
    "flan-small": "google/flan-t5-small",
    "flan-t5-small": "google/flan-t5-small",   # 👈 add full name
    "flan-base": "google/flan-t5-base",
    "flan-t5-base": "google/flan-t5-base",     # 👈 add full name
    "flan-large": "google/flan-t5-large",
    "flan-t5-large": "google/flan-t5-large",   # 👈 add full name
}

DEFAULT_EMBEDDING = EMBEDDING_MODELS["mini"]
DEFAULT_GENERATION = GENERATION_MODELS["flan-base"]
