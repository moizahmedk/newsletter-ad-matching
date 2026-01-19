from fastapi import FastAPI
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
import faiss

app = FastAPI(title="Ad Recommendation API")

model = SentenceTransformer("all-MiniLM-L6-v2")
ctr_model = joblib.load("models/ctr_model.pkl")
ads = pd.read_csv("data/ads.csv")
ad_embeddings = np.load("models/ad_embeddings.npy")

index = faiss.IndexFlatIP(ad_embeddings.shape[1])
faiss.normalize_L2(ad_embeddings)
index.add(ad_embeddings)

@app.post("/recommend")
def recommend(newsletter_text: str, top_k: int = 3):
    emb = model.encode([newsletter_text])
    faiss.normalize_L2(emb)
    _, idx = index.search(emb, top_k)

    results = []
    for i in idx[0]:
        results.append({
            "ad_text": ads.iloc[i]["ad_text"],
            "historical_ctr": ads.iloc[i]["historical_ctr"]
        })
    return results
