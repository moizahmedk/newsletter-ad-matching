Newsletter Ad Matching System

Production-Style ML System for Semantic Ad Ranking & CTR Optimization

End-to-end machine learning system that matches newsletter content to relevant advertisements using transformer embeddings, vector similarity search, CTR prediction, and multi-objective ranking.

This project is inspired by real-world newsletter ad networks (e.g. beehiiv) and demonstrates how production ML systems are designed, evaluated, and deployed.

🔍 Problem Statement

Newsletter platforms must select ads that are:

Semantically relevant to content

Likely to be clicked (high CTR)

Fairly distributed across advertisers

Fast enough for real-time serving

This project solves the problem by combining NLP embeddings, CTR prediction, and ranking logic into a production-ready ML pipeline.

🧠 System Overview

Input:
Newsletter article text

Output:
Top-K ranked advertisements optimized for relevance and CTR

High-Level Architecture
Newsletter Text
      ↓
Transformer Embedding (SentenceTransformer)
      ↓
Vector Similarity Search (FAISS)
      ↓
CTR Prediction (Supervised Model)
      ↓
Multi-Objective Ranking
      ↓
FastAPI Inference Endpoint

🧩 Core Components
1. NLP Embeddings

Model: all-MiniLM-L6-v2

Embeds:

Newsletter content

Ad creatives

Enables semantic similarity beyond keyword matching

2. Vector Search

FAISS (Inner Product / Cosine similarity)

Retrieves candidate ads efficiently

Scales to large embedding collections

3. CTR Prediction Model

Model: Logistic Regression (baseline, interpretable)

Features:

Content–ad semantic similarity

Historical ad CTR

Metrics:

ROC-AUC

Log Loss

Precision–Recall AUC

4. Multi-Objective Ranking

Final score combines:

Final Score =
  0.7 × Predicted CTR
+ 0.3 × Semantic Similarity


This mirrors real ad systems where business objectives and ML signals are blended.

5. Production Inference API

FastAPI service

Stateless inference

Loads:

Trained CTR model

Precomputed embeddings

FAISS index

Example request:

POST /recommend
{
  "newsletter_text": "AI tools for startup productivity",
  "top_k": 3
}

🧪 Evaluation Strategy

Evaluation is separated from training and inference.

Offline Metrics

CTR Model:

ROC-AUC

Log Loss

Precision–Recall Curve

Ranking:

Mean Reciprocal Rank (MRR)

Qualitative Checks

Semantic retrieval sanity tests

End-to-end ranking inspection

All evaluation code lives in:

notebooks/02_evaluation.ipynb

📁 Repository Structure
newsletter-ad-matching-ml/
│
├── data/
│   └── ads.csv
│
├── notebooks/
│   ├── exploration.ipynb
│   └── 02_evaluation.ipynb
│
├── src/
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── features.py
│   ├── ctr_model.py
│   └── ranking.py
│
├── api/
│   └── main.py
│
├── models/
│   ├── ctr_model.pkl
│   ├── ad_embeddings.npy
│   └── faiss.index
│
├── requirements.txt
├── .gitignore
└── README.md

🚀 How to Run
1. Install Dependencies
pip install -r requirements.txt

2. Start API
uvicorn api.main:app --reload

3. Test Endpoint
curl -X POST "http://127.0.0.1:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{"newsletter_text": "AI startups using cloud tools", "top_k": 3}'

🛠 Tech Stack

Python

SentenceTransformers

FAISS

scikit-learn

FastAPI

MLflow (optional)

NumPy / Pandas

📈 Scalability & Production Notes

This system is intentionally designed to scale:

Embeddings can be precomputed offline

FAISS supports millions of vectors

CTR model can be replaced with:

XGBoost

Deep learning models

Ranking layer supports:

Budget constraints

Frequency capping

Multi-objective optimization

Easily deployable via Docker + ECS/EKS

🔮 Future Improvements

Online learning / feedback loops

Cold-start handling using content priors

Subscriber-level personalization

Vector databases (Qdrant / Pinecone)

Model monitoring & drift detection

Multi-armed bandits for exploration

🎯 Why This Project

This project demonstrates:

End-to-end ML ownership

Production-oriented thinking

Ranking & optimization experience

NLP + vector search expertise

Real-world ad system design


📌 Author

Moiz Ahmed Khan
Machine Learning Engineer | Computer Vision | Production ML Systems