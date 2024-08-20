import streamlit as st
from dotenv import load_dotenv
import os

# Import functions from other files
from preprocessed2 import main as preprocessed_main
from upload2 import main as upload_main
from summary_doc import main as summary_main
from sql_preprocessed import main as sql_main
from csv_xlsx_preprocessed import main as csv_xlsx_main
from upload_csv_xlsx import main as upload_csv_xlsx_main

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

def main():
    st.set_page_config(page_title="Multi-PDF Chat and Summary", page_icon=":books:")
    
    # Apply custom CSS
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("style.css")
    
    st.header("Multi-PDF Chat and Summary :books:")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Home", "PreProcessed", "Upload", "Summary", "Sql Query","csv or xlsx Query", "Upload csv or xlsx File for Query"])

    if app_mode == "Home":
        st.write("""
        # Welcome to Multi-PDF Chat and Summary!
        
        This application offers three main functionalities:
        
        1. **PreProcessed Chat**: Chat with pre-processed PDF files using RAG (Retrieval-Augmented Generation).
        2. **Upload and Chat**: Upload your own PDF files, process them, and then chat with the content.
        3. **PDF Summary**: Upload a PDF file and get a comprehensive summary.
        
        To get started, select a mode from the sidebar on the left.
        
        ## How to use:
        
        1. Choose your desired functionality from the sidebar.
        2. Follow the instructions on the screen for each mode.
        3. Enjoy interacting with your PDFs!
        
        This tool is powered by advanced language models and document processing techniques to provide you with accurate and context-aware responses.
        
        ## Pre-processed PDFs
        
        In the PreProcessed Chat mode, you can interact with information from the following pre-processed PDFs:
        
        1. **Research Paper**: "Learning Transferable Visual Models From Natural Language Supervision"
        2. **Business Guide**: "How To Start The Startup"
        3. **Story Book**: Containing three stories:
           - Amarok the lone wolf
           - Fred the red fish
           - Lily the bee
        4. **Research Paper**: "AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE"
        
        Feel free to ask questions about these documents in the PreProcessed Chat mode!
        """)
        
        st.info("Select an option from the sidebar to begin!")

    elif app_mode == "PreProcessed":
        preprocessed_main()
    elif app_mode == "Upload":
        upload_main()
    elif app_mode == "Summary":
        summary_main()
    elif app_mode == "Sql Query":
        sql_main()
    elif app_mode == "csv or xlsx Query":
        csv_xlsx_main()
    elif app_mode == "Upload csv or xlsx File for Query":
        upload_csv_xlsx_main()

if __name__ == "__main__":
    main()