import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

def load_documents():
    with open("data/medical_data.json", "r") as f:
        documents = json.load(f)
    return documents

def build_retriever(documents):
    print("Building embeddings...")
    
    texts = [doc["question"] + " " + doc["answer"] for doc in documents]
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    
    tokenized = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    
    print("Retriever Ready!")
    return model, index, bm25, texts

if __name__ == "__main__":
    docs = load_documents()
    build_retriever(docs)