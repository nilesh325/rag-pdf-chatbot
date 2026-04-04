from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
import math

app = Flask(__name__)

mistral_api_key = "my-api-key"  # Replace with your Mistral API key

# ✅ Load embeddings ONCE (fix performance)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# ✅ Cache vector store
vector_store = None


def get_dynamic_chunk_size(text, max_context_tokens=4000, overlap_ratio=0.15):
    total_words = len(text.split())
    target_chunks = min(10, math.ceil(total_words / 1000))
    chunk_size = max_context_tokens // target_chunks
    chunk_overlap = int(chunk_size * overlap_ratio)
    return chunk_size, chunk_overlap


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    global vector_store

    if "pdf" not in request.files:
        return jsonify({"answer": "❌ No PDF uploaded!"})

    file = request.files["pdf"]
    question = request.form.get("question", "")

    if file.filename == "":
        return jsonify({"answer": "❌ Please select a PDF file."})

    if question.strip() == "":
        return jsonify({"answer": "❌ Please enter a question."})

    try:
        # ✅ Process PDF ONLY ONCE
        if vector_store is None:
            pdf_reader = PdfReader(file)
            text = ""

            for page in pdf_reader.pages:
                text += page.extract_text() or ""

            if text.strip() == "":
                return jsonify({"answer": "❌ Could not extract text from PDF."})

            chunk_size, chunk_overlap = get_dynamic_chunk_size(text)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            chunks = splitter.split_text(text)

            vector_store = FAISS.from_texts(chunks, embeddings)

        # 🔍 Retrieve relevant chunks
        docs = vector_store.similarity_search(question)

        # ✅ Fixed token limit (no truncation)
        llm = ChatMistralAI(
            mistral_api_key=mistral_api_key,
            temperature=0.7,
            max_tokens=800,
            model="mistral-small"
        )

        # ✅ SINGLE LLM CALL (important fix)
        response = llm.invoke(
            f"""
            Answer the question in a detailed and complete way.
            Do NOT shorten the answer.

            Context:
            {docs}

            Question:
            {question}
            """
        )

        return jsonify({"answer": response.content})

    except Exception as e:
        return jsonify({"answer": f"❌ Error: {str(e)}"})


if __name__ == "__main__":
    app.run(debug=True)