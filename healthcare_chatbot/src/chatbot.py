from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

def create_chatbot(model, index, bm25, texts):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )
    
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful medical assistant.
    Answer the question based on the context below.
    If you don't know, say "I don't have enough information."
    Always recommend consulting a doctor for medical advice.
    
    Context: {context}
    Question: {question}
    
    Answer:
    """)
    
    def retrieve(query):
        query_embedding = model.encode([query])
        _, indices = index.search(np.array(query_embedding), k=3)
        
        bm25_scores = bm25.get_scores(query.split())
        top_bm25 = sorted(range(len(bm25_scores)),
                         key=lambda i: bm25_scores[i], reverse=True)[:3]
        
        combined = list(set(indices[0].tolist() + top_bm25))
        context = "\n".join([texts[i] for i in combined[:4]])
        return context
    
    def answer(question):
        context = retrieve(question)
        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "question": question
        })
        return response.content
    
    return answer