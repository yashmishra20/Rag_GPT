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
    st.set_page_config(page_title="Multi-Format Data Interaction", page_icon=":mag:")
    
    # Apply custom CSS
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    local_css("style.css")
    
    st.header("Multi-Format Data Interaction :mag:")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Home", "PreProcessed", "Upload", "Summary", "SQL Query", "CSV/XLSX Query", "Upload CSV/XLSX File for Query"])
    
    if app_mode == "Home":
        st.write("""
        # Welcome to Multi-Format Data Interaction!
        
        This application offers multiple functionalities for interacting with different types of data:
        
        1. **PreProcessed Chat**: Chat with pre-processed PDF files using RAG (Retrieval-Augmented Generation).
        2. **Upload and Chat**: Upload your own PDF files, process them, and then chat with the content.
        3. **PDF Summary**: Upload a PDF file and get a comprehensive summary.
        4. **SQL Query**: Interact with the Chinook database using natural language queries.
        5. **CSV or XLSX Query**: Analyze cancer and diabetes datasets using natural language queries.
        6. **Upload CSV or XLSX File for Query**: Upload your own CSV or XLSX file for querying.
        
        To get started, select a mode from the sidebar on the left.
        
        ## How to use:
        
        1. Choose your desired functionality from the sidebar.
        2. Follow the instructions on the screen for each mode.
        3. Enjoy interacting with your data!
        
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

        ## SQL Query
        
        In the SQL Query mode, you can interact with the Chinook database, which represents a digital media store. You can ask questions about artists, albums, tracks, invoices, and customers.

        ## CSV or XLSX Query
        
        In the CSV or XLSX Query mode, you can analyze cancer and diabetes datasets. Ask questions about patient data, medical features, and correlations within these datasets.
        """)
        
        st.info("Select an option from the sidebar to begin!")

    elif app_mode == "PreProcessed":
        preprocessed_main()
    elif app_mode == "Upload":
        upload_main()
    elif app_mode == "Summary":
        summary_main()
    elif app_mode == "SQL Query":
        sql_main()
    elif app_mode == "CSV/XLSX Query":
        csv_xlsx_main()
    elif app_mode == "Upload CSV/XLSX File for Query":
        upload_csv_xlsx_main()

if __name__ == "__main__":
    main()