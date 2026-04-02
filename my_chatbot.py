import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph  
import math
import re

mistral_api_key = "diekRCMZhHW7GP14HpuVnL6OWV0Wiqu0"

st.header("My Chatbot")
#Sidebar
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Dynamically calculate chunk size and overlap based on text length, model context window, and overlap ratio
def get_dynamic_chunk_size(text, max_context_tokens=4000, overlap_ratio=0.15):
    
    # Roughly assume 1 word ≈ 1 token
    total_words = len(text.split())
    
    # Aiming for ~10 chunks max (tuneable)
    target_chunks = min(10, math.ceil(total_words / 1000))
    
    chunk_size = max_context_tokens // target_chunks
    chunk_overlap = int(chunk_size * overlap_ratio)
    
    return chunk_size, chunk_overlap

#1) Upload PDF and extract text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    chunk_size, chunk_overlap = get_dynamic_chunk_size(text)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
#2) Spliting text into chunks 
    chunks = text_splitter.split_text(text)

#3)Creating embeddings 
    embeddings = HuggingFaceEmbeddings()

#4)store in vector database
    vector_store = FAISS.from_texts(chunks, embeddings)

#5)Ask questions
    user_question = st.text_input("Type your question here")

    # Step 1: classify response type i.e. brief or detailed
    def classify_response_type(user_question, llm):
        prompt = f"""The user asked: "{user_question}" Decide the maximum number of tokens needed to answer this question fully.Return ONLY a number, nothing else."""
        raw = llm.invoke(prompt).content.strip()
        match = re.search(r'\d+', raw)
        return max(500, int(match.group())) if match else 500

    if user_question:
        llm_for_classification = ChatMistralAI(
            mistral_api_key=mistral_api_key,
            temperature=0.7,  # deterministic output
            max_tokens=10,    # just need a short response
            model="mistral-small")

#6) Similarity search in vector database
        match = vector_store.similarity_search(user_question,k=6)
        max_tokens = int(classify_response_type(user_question, llm_for_classification))
        llm = ChatMistralAI(
            mistral_api_key=mistral_api_key,
            temperature=0.7,
            max_tokens=max_tokens,
            model="mistral-small"
        )

#7) Create a graph to retrieve relevant chunks and generate an answer
        graph = StateGraph(dict)

        def retrieve(state):
            return {"docs": match}

        def answer(state):
            docs = state["docs"]

            response1 = llm.invoke( 
                f"Answer the question based on these docs:\n{docs}\n\nQuestion: {user_question}"
            )
            response = llm.invoke( #convert the answer to a concise and clear response with a maximum of {max_tokens} tokens.
                f"convert this answer into a concise and clear response with a maximum of {max_tokens} tokens:\n{response1}"
            )
            return {"answer": response}

        graph.add_node("retrieve", retrieve)
        graph.add_node("answer", answer)
        graph.add_edge("retrieve", "answer")
        graph.set_entry_point("retrieve")
        graph.set_finish_point("answer")

        compiled = graph.compile()
        result = compiled.invoke({})
        st.write(result["answer"].content)












    
