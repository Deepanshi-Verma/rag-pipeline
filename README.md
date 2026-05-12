# Healthcare RAG Chatbot

A medical question-answering chatbot built using Retrieval-Augmented Generation (RAG) with hybrid search and Groq LLM.

---

## Tech Stack

- LLM: Groq (LLaMA 3.3 70B)
- Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
- Vector Store: FAISS
- Keyword Search: BM25
- Framework: LangChain
- UI: Streamlit
- Evaluation: RAGAS
- Dataset: ChatDoctor-HealthCareMagic-100k (500 records)

---

## How It Works

The chatbot uses hybrid search to retrieve relevant medical documents for any user query. FAISS handles semantic search while BM25 handles keyword matching. The top results are passed as context to the Groq LLM which generates the final answer.

---

## Evaluation

Evaluated using RAGAS framework:

- Faithfulness: 0.79 (answers are grounded in retrieved context, low hallucination)

---

## Setup

1. Clone the repository

git clone https://github.com/Deepanshi-Verma/rag-pipeline.git
cd rag-pipeline/healthcare_chatbot

2. Install dependencies

pip install -r requirements.txt

3. Add your API key — create a .env file

GROQ_API_KEY=your_groq_api_key_here

4. Load the dataset

python src/ingest.py

5. Run the chatbot

streamlit run app.py

---

## Sample Questions

- What are the symptoms of diabetes?
- How is high blood pressure treated?
- What causes anemia?
- What are common side effects of ibuprofen?

---

## Author

Deepanshi Verma
Program Analyst | GenAI Developer
GitHub: https://github.com/Deepanshi-Verma