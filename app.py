import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ----------------------------
# CONFIG
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

# ----------------------------
# STREAMLIT UI
# ----------------------------

st.title("📄 Resume RAG Chatbot")

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

query = st.text_input(
    "Ask about candidates (e.g., 'Find Python developers with 5 years experience')"
)

if query and "vectorstore" in st.session_state:
    qa_chain = create_qa_chain(st.session_state["vectorstore"])
    response = qa_chain.run(query)
    st.write("### Answer:")
    st.write(response)
