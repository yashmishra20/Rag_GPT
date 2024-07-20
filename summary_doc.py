import streamlit as st
from langchain.document_loaders import PyPDFLoader
import os
import tempfile
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv 

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

max_final_token = 3000
character_overlap = 100
token_threshold = 0

def get_pdf_text(pdf_docs):
    docs = []
    docs.extend(PyPDFLoader(pdf_docs).load())
    max_summarizer_output_token = int(max_final_token/len(docs)) - token_threshold
    full_summary = ""
    counter = 1
    if len(docs) > 1:
        for i in range(len(docs)):
            # NOTE: This part can be optimized by considering a better technique for creating the prompt. (e.g: lanchain "chunksize" and "chunkoverlap" arguments.)
            if i == 0:  # For the first page
                prompt = docs[i].page_content + \
                    docs[i+1].page_content[:character_overlap]
            elif i < len(docs)-1:  # For pages except the first and the last one.
                prompt = docs[i-1].page_content[-character_overlap:] + \
                    docs[i].page_content + \
                    docs[i+1].page_content[:character_overlap]
            else:  # For the last page
                prompt = docs[i-1].page_content[-character_overlap:] + \
                    docs[i].page_content
            full_summary += page_response(max_summarizer_output_token,prompt)

    else:  # if the document has only one page
        full_summary = docs[0].page_content
        counter += 1
    final_summary = complete_summary(full_summary)
    return final_summary

def page_response(max_summarizer_output_token,prompt1):
    prompt_template="""You are an expert text summarizer. You will receive a text and your task is to summarize and keep all the key information. \
    Keep the maximum length of summary within {token} number of tokens.
    text:\n{prompt}\n
    """
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["token","prompt"])
    chain=LLMChain(llm=model,prompt=prompt,output_key="output_key")
    response = chain(
        {"token": max_summarizer_output_token, "prompt": prompt1}
    )
    text=response["output_key"]
    return text

def complete_summary(prompt1):
    prompt_template="""You are an expert text summarizer. \
    You will receive a text and your task is to give a comprehensive summary and keep all the key information.
    text:\n{prompt}\n
    """
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["prompt"])
    chain=LLMChain(llm=model,prompt=prompt,output_key="output_key")
    response = chain(
        {"prompt": prompt1}
    )
    text=response["output_key"]
    return text

def main():
    # st.set_page_config(page_title="Summarize your PDF",
    #                    page_icon=":books:")
    # st.header("Summarize your PDF :books:")

    # Main chat area
    chat_area = st.empty()

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF File and Click on the Submit & Process Button", accept_multiple_files=False)
        if st.button("Submit & Process"):
            if pdf_docs is not None:
                with st.spinner("Processing..."):
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(pdf_docs.getvalue())
                        tmp_file_path = tmp_file.name

                    # Get the summary
                    summary = get_pdf_text(tmp_file_path)
                    
                    # Remove the temporary file
                    os.remove(tmp_file_path)
                    
                    # Display the summary in the main chat area
                    chat_area.markdown("## PDF Summary")
                    chat_area.write(summary)
                    
                    st.success("Done")
            else:
                st.error("Please upload a PDF file.")

# if __name__ == "__main__":
#     main()