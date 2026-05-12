import os
import time
import numpy as np
from dotenv import load_dotenv
from datasets import Dataset
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from src.retriever import load_documents, build_retriever
from src.chatbot import create_chatbot

load_dotenv()

# Configure RAGAS to use Groq
groq_llm = LangchainLLMWrapper(
    ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )
)

# Instantiate metrics with ()
faithfulness = Faithfulness(llm=groq_llm)
answer_relevancy = AnswerRelevancy(llm=groq_llm)

test_questions = [
    "What are the symptoms of diabetes?",
    "How is high blood pressure treated?",
    "What causes anemia?",
    "What are common side effects of ibuprofen?",
    "How can I manage asthma at home?"
]

def run_evaluation():
    print("Loading knowledge base...")
    docs = load_documents()
    model, index, bm25, texts = build_retriever(docs)
    answer_fn = create_chatbot(model, index, bm25, texts)

    questions = []
    answers = []
    contexts = []

    print("Generating answers...")
    for q in test_questions:
        ans = answer_fn(q)

        query_embedding = model.encode([q])
        _, indices = index.search(np.array(query_embedding), k=3)
        bm25_scores = bm25.get_scores(q.split())
        top_bm25 = sorted(range(len(bm25_scores)),
                          key=lambda i: bm25_scores[i], reverse=True)[:3]
        combined = list(set(indices[0].tolist() + top_bm25))
        ctx = [texts[i] for i in combined[:4]]

        questions.append(q)
        answers.append(ans)
        contexts.append(ctx)
        print(f"Done: {q[:40]}...")
        time.sleep(2)

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    dataset = Dataset.from_dict(data)

    print("\nRunning RAGAS evaluation...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy]
    )

    print("\n===== RAGAS Scores =====")
    print(results)
    return results

if __name__ == "__main__":
    run_evaluation()