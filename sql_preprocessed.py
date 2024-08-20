import streamlit as st
import os
from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from dotenv import load_dotenv 
import time

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

def respond(question: str):
    sqldb_directory="C:/Users/Yash mishra/Desktop/PROJECTS/Rag GPT/Rag GPT/data/sqldb.db"
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    agent_llm_system_role= """Given the following user question, corresponding SQL query, and SQL result, answer the user question in one line.\n
    Question: {question}\n
    SQL Query: {query}\n
    SQL Result: {result}\n
    Answer: """

    if os.path.exists(sqldb_directory):
        db = SQLDatabase.from_uri(f"sqlite:///{sqldb_directory}")
        execute_query = QuerySQLDataBaseTool(db=db)
        write_query = create_sql_query_chain(llm, db)
        answer_prompt = PromptTemplate.from_template(agent_llm_system_role)
        answer = answer_prompt | llm | StrOutputParser()
        chain = ( RunnablePassthrough.assign(query=write_query).assign(
                    result=itemgetter("query") | execute_query)| 
                    answer
                )
        response = chain.invoke({"question": question})
        
    return response

def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()
    
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

    user_question = st.chat_input("Ask a question about your sql database:")
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
        
        # Getting response from the llm
        response = respond(user_question)
        
        # Simulate streaming for demonstration purposes
        for char in response:
            full_response += char
            response_placeholder.markdown(f'''
            <div class="message-bubble">
                <img src="https://img.icons8.com/ios-filled/50/228BE6/chatgpt.png" class="message-icon" alt="AI">
                <div class="ai-message">{full_response}▌</div>
            </div>
            ''', unsafe_allow_html=True)
            time.sleep(0.01)  # Adjust this value to control the speed of the "streaming"
        
        # Display the final response without the cursor
        response_placeholder.markdown(f'''
        <div class="message-bubble">
            <img src="https://img.icons8.com/ios-filled/50/228BE6/chatgpt.png" class="message-icon" alt="AI">
            <div class="ai-message">{full_response}</div>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add the AI's response to the chat history
        st.session_state.chat_history.add_ai_message(full_response)

# if __name__ == "__main__":
#     main()