import numpy as np

def rank_ads(article_embedding, ads, ad_embeddings, index, ctr_model):
    faiss.normalize_L2(article_embedding.reshape(1, -1))
    scores, idx = index.search(article_embedding.reshape(1, -1), k=len(ads))

    results = []
    for i in idx[0]:
        sim = np.dot(article_embedding, ad_embeddings[i])
        ctr = ads.iloc[i]["historical_ctr"]
        ctr_pred = ctr_model.predict_proba([[sim, ctr]])[0, 1]

        final_score = 0.7 * ctr_pred + 0.3 * sim
        results.append((ads.iloc[i]["ad_text"], final_score))

    return sorted(results, key=lambda x: x[1], reverse=True)
