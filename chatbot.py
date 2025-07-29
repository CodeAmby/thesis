# chatbot.py
import os
import streamlit as st
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. Configure your OpenAI key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]  # or set via env var

# 2. Prepare the QA chain (runs once on startup)
@st.cache_resource
def init_qa_chain(pdf_folder: str):
    # — Load all PDFs in folder
    from langchain.document_loaders import DirectoryLoader
    loader = DirectoryLoader(pdf_folder, glob="**/*.pdf", loader_cls=PDFPlumberLoader)
    docs = loader.load()

    # — Chunk
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n")
    chunks = splitter.split_documents(docs)

    # — Embed + index
    embeddings = OpenAIEmbeddings()
    vector_index = FAISS.from_documents(chunks, embeddings)

    # — Build chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_index.as_retriever(search_kwargs={"k": 4})
    )

qa_chain = init_qa_chain(r"C:\Users\ambyb\Desktop\Budgeting\Performance_Reports\PDF")

def ask_pdf(question: str) -> str:
    return qa_chain.run(question)

# 3. Streamlit UI
def main():
    st.title("📄 PDF-Powered Chatbot")
    st.write("Ask me anything about the documents you’ve loaded.")

    question = st.text_input("Your question:")
    if question:
        with st.spinner("Thinking…"):
            answer = ask_pdf(question)
        st.markdown("**Answer:**")
        st.write(answer)

if __name__ == "__main__":
    main()

