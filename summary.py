from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

def generate_summary(uploaded_file):
    # Load the document
    if uploaded_file.name.endswith('.pdf'):
        loader = PyPDFLoader(uploaded_file)
        pages = loader.load_and_split()
    else:
        return "Unsupported file format. Please upload a PDF file."

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    docs = text_splitter.split_documents(pages)

    # Initialize the language model
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    # Load the summarization chain
    chain = load_summarize_chain(llm, chain_type="map_reduce")

    # Generate the summary
    summary = chain.run(docs)

    return summary