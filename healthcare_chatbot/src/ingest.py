from datasets import load_dataset
import json
import os

def load_medquad():
    print("Loading dataset...")
    dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train[:500]")
    
    documents = []
    for item in dataset:
        doc = {
            "question": item["instruction"],
            "answer": item["output"]
        }
        documents.append(doc)
    
    os.makedirs("data", exist_ok=True)
    with open("data/medical_data.json", "w") as f:
        json.dump(documents, f, indent=2)
    
    print(f"Saved {len(documents)} documents!")
    return documents

if __name__ == "__main__":
    load_medquad()