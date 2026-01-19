from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

def load_model():
    return SentenceTransformer(MODEL_NAME)

def generate_embeddings(texts, model):
    return model.encode(texts, show_progress_bar=True)
