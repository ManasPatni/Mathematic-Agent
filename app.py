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

# Step 1: Retrieve Context

def retrieve_context(query, top_k=2):
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results.get("documents", [[]])

# Step 2: Solve Mathematical Problems

def solve_math_problem(problem):
    try:
        x = symbols('x')
        equation = Eq(eval(problem.replace('^2', '*2').replace('^3', '*3')), 0)
        solutions = solve(equation, x)
        formatted_solutions = [latex(simplify(sol)) for sol in solutions]
        return formatted_solutions
    except Exception as e:
        return f"Error: {e}"

# Step 3: Chat Handling

def query_math_assistant(user_query):
    system_prompt = """
    You are an advanced mathematics assistant, designed to solve problems of any complexity across all mathematical domains, including:

    1. Algebra: Solve equations, inequalities, and systems of equations.
    2. Calculus: Perform differentiation, integration, limits, and analyze functions.
    3. Linear Algebra: Handle matrices, vector spaces, eigenvalues, and eigenvectors.
    4. Geometry: Analyze shapes, compute areas, volumes, and handle coordinate geometry.
    5. Probability and Statistics: Solve problems involving distributions, probability theory, and statistical analysis.
    6. Discrete Mathematics: Tackle combinatorics, graph theory, and logic.
    7. Advanced Topics: Work on differential equations, complex numbers, Fourier transforms, and more.

    Responsibilities:
    - Provide accurate step-by-step solutions to problems of any difficulty level.
    - Explain mathematical concepts clearly, concisely, and with precision.
    - Ensure results are accurate and formatted cleanly using LaTeX when applicable.

    Guidelines:
    - Always verify the correctness of solutions before presenting them.
    - Offer alternative approaches or methods when applicable.
    - Respond politely, professionally, and empathetically to all user queries.
    - Avoid unnecessary details and focus on addressing the query directly.
    """

    # Retrieve context
    retrieved_context = retrieve_context(user_query)

    # Combine prompt
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"📖 Context: {retrieved_context}\n\n📝 Problem: {user_query}")
    ]

    try:
        # Generate response
        response = chat.invoke(messages)
        memory.save_context({"input": user_query}, {"output": response.content})
        return response.content if response else "⚠ No response received."
    except Exception as e:
        return f"⚠ Error: {str(e)}"

# User Interface
st.title("Advanced Mathematics Assistant")
st.header("Mathematical Problem Solver")

user_query = st.text_input("📝 Enter a mathematical problem or equation:")

if user_query:
    if "=" in user_query:
        try:
            equation = user_query.replace("=", "-").replace("^2", "*2").replace("^3", "*3")
            solutions = solve_math_problem(user_query.replace("=", "==").replace("^2", "*2").replace("^3", "*3"))
            st.markdown(f"*Solutions:* {', '.join(solutions)}")
        except Exception as e:
            st.error(f"Error solving the equation: {e}")
    else:
        response = query_math_assistant(user_query)
        st.markdown(f"*Response:* {response}")
