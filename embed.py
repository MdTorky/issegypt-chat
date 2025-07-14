import torch
import hashlib

# Simple hash-based embedding to avoid sentence-transformers
def embed(text: str, dim: int = 384):
    # Deterministic seed based on text hash
    hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    torch.manual_seed(hash_val % (2**32))
    return torch.rand(dim).tolist()
