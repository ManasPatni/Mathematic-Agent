import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb
import streamlit as st
import os
import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util
from sympy import symbols, Eq, solve, simplify, latex

# Initialize models and database
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="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")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ChromaDB initialization
chroma_client = chromadb.PersistentClient(path="./math_chroma_db")
collection = chroma_client.get_or_create_collection(name="math_knowledge_base")

# Retrieve Context
def retrieve_context(query, top_k=2):
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results.get("documents", [[]])

# Solve Mathematical Problems
def solve_math_problem(problem):
    try:
        x = symbols('x')
        equation = Eq(eval(problem.replace('^2', '*2').replace('^3', '*3')), 0)
        solutions = solve(equation, x)
        formatted_solutions = [latex(simplify(sol)) for sol in solutions]
        return formatted_solutions
    except Exception as e:
        return f"Error: {e}"

# Chat Handling
def query_math_assistant(user_query):
    system_prompt = """
    You are an advanced mathematics assistant, designed to solve problems of any complexity across all mathematical domains.
    """
    retrieved_context = retrieve_context(user_query)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"📖 Context: {retrieved_context}\n\n📝 Problem: {user_query}")
    ]
    try:
        response = chat.invoke(messages)
        memory.save_context({"input": user_query}, {"output": response.content})
        return response.content if response else "⚠ No response received."
    except Exception as e:
        return f"⚠ Error: {str(e)}"

# User Interface
st.set_page_config(layout="wide")
st.title("Advanced Mathematics Chatbot 🤖")

st.markdown("### 💬 Chat with the Math Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("📝 Type your question here:")
if st.button("Ask"):  
    if user_query:
        if "=" in user_query:
            try:
                equation = user_query.replace("=", "-").replace("^2", "*2").replace("^3", "*3")
                solutions = solve_math_problem(user_query.replace("=", "==").replace("^2", "*2").replace("^3", "*3"))
                response = f"*Solutions:* {', '.join(solutions)}"
            except Exception as e:
                response = f"Error solving the equation: {e}"
        else:
            response = query_math_assistant(user_query)
        
        st.session_state.chat_history.append((user_query, response))

st.markdown("---")
st.markdown("### 📝 Chat History")
for user_q, bot_response in st.session_state.chat_history[::-1]:
    with st.chat_message("user"):
        st.write(f"**User:** {user_q}")
    with st.chat_message("assistant"):
        st.write(f"**Bot:** {bot_response}")
