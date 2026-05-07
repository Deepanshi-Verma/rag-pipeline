import streamlit as st
from src.retriever import load_documents, build_retriever
from src.chatbot import create_chatbot

st.title("🏥 Healthcare Q&A Chatbot")
st.caption("Powered by RAG + Groq LLM")

@st.cache_resource
def initialize():
    docs = load_documents()
    model, index, bm25, texts = build_retriever(docs)
    chatbot = create_chatbot(model, index, bm25, texts)
    return chatbot

with st.spinner("Loading medical knowledge base..."):
    answer = initialize()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = answer(question)
            st.write(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})