import streamlit as st 
import os
import pickle
import time
import langchain
from langchain.chat_models import init_chat_model
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss  import FAISS
from dotenv import load_dotenv

load_dotenv() # Load necessary keys from .env

st.title("New Research Tool 📈")
st.sidebar.title("News Article URLs")
file_path = "faiss_store_huggingface.pkl"

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

main_placeholder = st.empty()
process_url_clicked = st.sidebar.button("Process URLs")
if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    
    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    
    # Create embeddings
    embeedings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore_huggingface = FAISS.from_documents(docs, embedding=embeedings)    
    main_placeholder.text("Embedding Vector Starte Building...✅✅✅")
    # Save the FAISS index to pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_huggingface, f)
        
query = main_placeholder.text_input("Question: ")
if query:
    model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
    if os.path.exists(file_path):
        with open(file_path , "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=model, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            
            # {"answer" : "", "sources" : []}
            st.header("Answer")
            st.subheader(result["answer"])
            
            # Display sources, if available
            
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources: ")
                sources_list = sources.split("\n")
                
                for source in sources_list:
                    st.write(source)