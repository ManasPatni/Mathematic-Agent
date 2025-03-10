import sys
from sympy import symbols, Eq, solve, simplify, latex
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
import streamlit as st
import chromadb

# Initialize models and database
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="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")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ✅ Updated ChromaDB initialization
chroma_client = chromadb.PersistentClient(path="./math_chroma_db")  # For ChromaDB < 0.4.24
# chroma_client = chromadb.Client()  # If using latest ChromaDB (>=0.4.24)

collection = chroma_client.get_or_create_collection(name="math_knowledge_base")

# Function to retrieve context
def retrieve_context(query, top_k=2):
    try:
        query_embedding = embedding_model.embed_query(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results.get("documents", [[]])
    except Exception as e:
        return f"⚠ Error retrieving context: {e}"

# Function to solve mathematical problems
def solve_math_problem(problem):
    try:
        x = symbols('x')
        equation = Eq(eval(problem.replace('^2', '**2').replace('^3', '**3')), 0)
        solutions = solve(equation, x)
        formatted_solutions = [latex(simplify(sol)) for sol in solutions]
        return formatted_solutions
    except Exception as e:
        return [f"⚠ Error: {e}"]

# Function to handle queries to the math assistant
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

    # Combine prompt and user query
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"📖 Context: {retrieved_context}\n\n📝 Problem: {user_query}")
    ]

    try:
        # Generate response from the chat assistant
        response = chat.invoke(messages)
        memory.save_context({"input": user_query}, {"output": response.content})
        return response.content if response else "⚠ No response received."
    except Exception as e:
        return f"⚠ Error: {str(e)}"

# Streamlit user interface
st.title("Advanced Mathematics Assistant")
st.header("Mathematical Problem Solver")

user_query = st.text_input("📝 Enter a mathematical problem or equation:")

if user_query:
    if "=" in user_query:
        try:
            solutions = solve_math_problem(user_query)
            if solutions:
                st.markdown(f"*Solutions:* {', '.join(solutions)}")
            else:
                st.warning("No solutions found.")
        except Exception as e:
            st.error(f"⚠ Error solving the equation: {e}")
    else:
        response = query_math_assistant(user_query)
        st.markdown(f"*Response:* {response}")
