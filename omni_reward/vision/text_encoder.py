from sentence_transformers import SentenceTransformer
import numpy as np

class TextEncoder:
    def __init__(self, model_name="all-mpnet-base-v2", device="cuda"): # example
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def encode_one(self, text: str):
        emb = self.model.encode([text], normalize_embeddings=True)
        return emb[0]

    def encode_many(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)
