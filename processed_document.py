# processed_document.py

import os
from dotenv import load_dotenv # type: ignore
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI

# Initialize global variables
vector_store = None
llm = None

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

def load_processed_documents(folder_path="data\docs"):
    global vector_store, llm

    # Load PDF documents from the folder
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the specified folder.")
        return

    # Load and process documents
    docs = []
    for pdf in pdf_files:
        loader = PyPDFLoader(os.path.join(folder_path, pdf))
        docs.extend(loader.load())

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Initialize the language model
    llm = ChatOpenAI(temperature=0)

def get_relevant_chunks(query: str, k: int = 3) -> List[str]:
    global vector_store

    if vector_store is None:
        return []

    # Perform similarity search
    relevant_docs = vector_store.similarity_search(query, k=k)

    # Extract the content from the documents
    relevant_chunks = [doc.page_content for doc in relevant_docs]

    return relevant_chunks

def chat_with_processed_documents(user_input: str) -> str:
    global llm

    if vector_store is None or llm is None:
        return "No processed documents available. Please check the document folder."

    # Get relevant chunks
    relevant_chunks = get_relevant_chunks(user_input)

    if not relevant_chunks:
        return "I couldn't find any relevant information to answer your question."

    # Prepare the prompt
    prompt = f"""Based on the following information, please answer the user's question. 
If the information provided is not sufficient to answer the question, please say so.

User question: {user_input}

Relevant information:
{' '.join(relevant_chunks)}

Answer:"""

    # Generate the answer
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    response = llm.generate([messages])
    answer = response.generations[0][0].text

    return answer.strip()