import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import pipeline

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Resume RAG Chatbot", layout="wide")

# ----------------------------
# FUNCTIONS
# ----------------------------

def load_documents(uploaded_files):
    docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())

    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vectorstore(chunks):
    embeddings = load_embeddings()
    return FAISS.from_documents(chunks, embeddings)


# 🔥 FIXED MODEL PIPELINE
@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",   # ✅ FIXED (no more error)
        model="google/flan-t5-base",
        max_length=200,
        do_sample=False
    )


def generate_answer(vectorstore, query):
    if vectorstore is None:
        return "⚠️ Please upload and process resumes first."

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # ✅ Latest LangChain method
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found."

    context = "\n\n".join([doc.page_content for doc in docs])

    model = load_model()

    prompt = f"""
You are an AI assistant helping HR analyze resumes.

Answer ONLY from the context below.
If the answer is not present, say "Not mentioned in resume".

Context:
{context}

Question: {query}
"""

    result = model(prompt)

    return result[0]["generated_text"]


# ----------------------------
# UI
# ----------------------------

st.title("📄 Resume RAG Chatbot (FREE - No API)")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success("✅ Files uploaded successfully!")

    if st.button("Process Resumes"):
        with st.spinner("Processing resumes..."):
            docs = load_documents(uploaded_files)

            if not docs:
                st.error("❌ Failed to read PDFs")
            else:
                chunks = split_documents(docs)
                vectorstore = create_vectorstore(chunks)

                st.session_state["vectorstore"] = vectorstore

        st.success("✅ Resumes processed successfully!")

# ----------------------------
# CHAT SECTION
# ----------------------------

st.divider()
st.subheader("💬 Ask Questions")

query = st.text_input("Ask about candidates")

if query:
    if "vectorstore" not in st.session_state:
        st.warning("⚠️ Please upload and process resumes first.")
    else:
        with st.spinner("Thinking..."):
            answer = generate_answer(
                st.session_state["vectorstore"],
                query
            )

        st.markdown("### 🧠 Answer:")
        st.write(answer)
