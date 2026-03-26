import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------
# CONFIG
# ----------------------------
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

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
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        huggingfacehub_api_token=HUGGINGFACE_API_KEY
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the provided context.

Context:
{context}

Question:
{input}
"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    return qa_chain


# ----------------------------
# STREAMLIT UI
# ----------------------------

st.title("📄 Resume RAG Chatbot (FREE VERSION)")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success("Files uploaded successfully!")

    if st.button("Process Resumes"):
        with st.spinner("Processing..."):
            documents = load_documents(uploaded_files)
            chunks = split_documents(documents)
            vectorstore = create_vectorstore(chunks)
            st.session_state["vectorstore"] = vectorstore
        st.success("Resumes processed!")

query = st.text_input("Ask about candidates")

if query and "vectorstore" in st.session_state:
    qa_chain = create_qa_chain(st.session_state["vectorstore"])
    response = qa_chain.invoke({"input": query})

    st.write("### Answer:")
    st.write(response["answer"])
