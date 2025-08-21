from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores.faiss  import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
import dotenv
import os 

# Init a model
dotenv.load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0.7)

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings( model="models/gemini-embedding-001")
vectordb_file_path = "3_Codebasics_Q&A/faiss_index"

def create_vector_db():
    # Load a data
    loader = CSVLoader(file_path="3_Codebasics_Q&A/codebasics_faqs.csv", source_column="prompt")
    data = loader.load()
    
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    # Save to local memory
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database from the local folder
    vector_db = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True  # faqat ishonchli fayl uchun True qiling
)
    
    # Add a prompt template
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    retriever = vector_db.as_retriever(score_threshold=0.7)
    
    chain = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
        
    )
    return chain
    
if __name__ == "__main__":
    chain = get_qa_chain()
    print(chain("Do you have javascript cource"))