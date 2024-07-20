import streamlit as st
import os
from PyPDF2 import PdfReader # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv 

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Contextualize question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    retriever = vector_store.as_retriever()
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Answer question
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def main():
    # st.set_page_config(page_title="Chat with multiple PDFs",
    #                    page_icon=":books:")
    
    # #--Heade Section --
    # # Custom CSS to style the text
    # def local_css(file_name):
    #     with open(file_name) as f:
    #         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # local_css("C:/Users/Yash mishra/Desktop/PROJECTS/Rag GPT/Rag GPT/style.css")
    
    # st.header("Chat with multiple PDFs :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(text)
                vector_store = get_vector_store(chunks)
                st.session_state.conversation = get_conversation_chain(vector_store)
                st.success("Processing complete! You can now ask questions.")

    # Display chat history
    st.markdown('<div class="message-container">', unsafe_allow_html=True)
    for message in st.session_state.chat_history.messages:
        if message.type == "human":
            st.markdown(f'''
            <div class="message-bubble" style="justify-content: flex-end;">
                <div class="user-message">{message.content}</div>
                <img src="https://img.icons8.com/fluency-systems-filled/48/228BE6/user.png" class="message-icon user-icon" alt="User">
            </div>
            ''', unsafe_allow_html=True)
        elif message.type == "ai":
            st.markdown(f'''
            <div class="message-bubble">
                <img src="https://img.icons8.com/ios-filled/50/228BE6/chatgpt.png" class="message-icon" alt="AI">
                <div class="ai-message">{message.content}</div>
            </div>
            ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.conversation:
        user_question = st.chat_input("Ask a question about your documents:")
        if user_question:
            # Add the user's question to the chat history
            st.session_state.chat_history.add_user_message(user_question)

            # Display user message
            st.markdown('<div class="message-container">', unsafe_allow_html=True)
            st.markdown(f'''
            <div class="message-bubble" style="justify-content: flex-end;">
                <div class="user-message">{user_question}</div>
                <img src="https://img.icons8.com/fluency-systems-filled/48/228BE6/user.png" class="message-icon user-icon" alt="User">
            </div>
            ''', unsafe_allow_html=True)
            
            # Create a placeholder for the AI response
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            for chunk in st.session_state.conversation.stream({
                "input": user_question,
                "chat_history": st.session_state.chat_history.messages
            }):
                if chunk.get('answer'):
                    full_response += chunk['answer']
                    response_placeholder.markdown(f'''
                    <div class="message-bubble">
                        <img src="https://img.icons8.com/ios-filled/50/228BE6/chatgpt.png" class="message-icon" alt="AI">
                        <div class="ai-message">{full_response}▌</div>
                    </div>
                    ''', unsafe_allow_html=True)
            
            # Display the final response
            response_placeholder.markdown(f'''
            <div class="message-bubble">
                <img src="https://img.icons8.com/ios-filled/50/228BE6/chatgpt.png" class="message-icon" alt="AI">
                <div class="ai-message">{full_response}</div>
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add the AI's response to the chat history
            st.session_state.chat_history.add_ai_message(full_response)

    else:
        st.write("Please upload and process your PDFs to start chatting.")

        
# if __name__ == "__main__":
#     main()