import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

@st.cache_data
def load_vector_db(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    return db

st.title("LLM Chatbot with RAG")
pdf_file = st.file_uploader("Upload a PDF for QA", type="pdf")

if pdf_file:
    db = load_vector_db(pdf_file)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    query = st.text_input("Ask a question based on the document:")
    if query:
        result = qa_chain.run(query)
        st.markdown(f"**Answer:** {result}")
