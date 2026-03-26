import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# ----------------------------
# FUNCTIONS
# ----------------------------

def load_documents(uploaded_files):
    docs = []
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(file.name)
        docs.extend(loader.load())
    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(chunks, embeddings)


# 🔥 LOCAL MODEL (NO API)
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-small")


def generate_answer(vectorstore, query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    model = load_model()

    prompt = f"""
Answer based on context:

{context}

Question: {query}
"""

    result = model(prompt, max_length=200)
    return result[0]["generated_text"]


# ----------------------------
# UI
# ----------------------------

st.title("📄 Resume RAG Chatbot (100% FREE - No API)")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success("Files uploaded successfully!")

    if st.button("Process Resumes"):
        with st.spinner("Processing..."):
            docs = load_documents(uploaded_files)
            chunks = split_documents(docs)
            st.session_state["vectorstore"] = create_vectorstore(chunks)
        st.success("Resumes processed!")

query = st.text_input("Ask about candidates")

if query and "vectorstore" in st.session_state:
    with st.spinner("Thinking..."):
        answer = generate_answer(st.session_state["vectorstore"], query)
        st.write("### Answer:")
        st.write(answer)
