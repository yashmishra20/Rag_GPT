from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Initialize global variables
vector_store = None
qa_chain = None

def process_uploaded_documents(uploaded_files):
    global vector_store, qa_chain

    # Load uploaded documents
    loaders = [PyPDFLoader(file) for file in uploaded_files if file.name.endswith('.pdf')]

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    docs = []
    for loader in loaders:
        docs.extend(loader.load_and_split(text_splitter))

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)

    # Create a conversational chain
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

def chat_with_uploaded_documents(user_input: str) -> str:
    global qa_chain

    if qa_chain is None:
        return "Please upload and process documents first."

    result = qa_chain({"question": user_input})
    return result['answer']