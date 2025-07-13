from langchain_community.document_loaders import PyPDFLoader # type: ignore
from langchain.text_splitter import CharacterTextSplitter # type: ignore
from langchain.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain.vectorstores import FAISS # type: ignore
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline # type: ignore
from langchain.chains import RetrievalQA # type: ignore
from transformers import pipeline # type: ignore
import streamlit as st # type: ignore

st.set_page_config(page_title="RAG PDF Chatbot", layout="centered")
st.title("PDF Question Answering Chatbot")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing your PDF..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        st.success(f"PDF loaded with {len(docs)} chunks.")
        st.subheader("Preview of first chunk:")
        st.code(docs[0].page_content[:500] + "..." if docs else "No content")

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embedding_model)

        hf_pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
        llm = HuggingFacePipeline(pipeline=hf_pipe)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff"
        )

        st.subheader("Ask a question from the PDF")
        query = st.text_input("Your question:")

        if query:
            with st.spinner("Generating response..."):
                try:
                    response = qa_chain.run(query)
                    st.success("Answer:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Please enter a question to get a response.")
else:
    st.info("Please upload a PDF to begin.")