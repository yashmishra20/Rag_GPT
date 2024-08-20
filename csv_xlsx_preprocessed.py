import streamlit as st
import os
from PyPDF2 import PdfReader # type: ignore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv 
from langchain_community.utilities import SQLDatabase
import time
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.prompts import (ChatPromptTemplate,
                                    FewShotPromptTemplate,
                                    MessagesPlaceholder,
                                    PromptTemplate,
                                    SystemMessagePromptTemplate)

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

examples = [
{
"input": "Show all columns from the cancer dataset.",
"query": "SELECT * FROM cancer LIMIT 5;"
},
{
"input": "Count the total number of records in the cancer dataset.",
"query": "SELECT COUNT() FROM cancer;"
},
{
"input": "Find the average clump thickness in the cancer dataset.",
"query": "SELECT AVG(clump_thickness) FROM cancer;"
},
{
"input": "List all records where uniformity of cell size is greater than 5 in the cancer dataset.",
"query": "SELECT * FROM cancer WHERE uniformity_of_cell_size > 5;"
},
{
"input": "Count how many records have a mitosis value of 1 in the cancer dataset.",
"query": "SELECT COUNT() FROM cancer WHERE mitosis = 1;"
},
{
"input": "Show all columns from the diabetes dataset.",
"query": "SELECT * FROM diabetes LIMIT 5;"
},
{
"input": "Count the total number of records in the diabetes dataset.",
"query": "SELECT COUNT() FROM diabetes;"
},
{
"input": "Calculate the average BMI for patients in the diabetes dataset.",
"query": "SELECT AVG(BMI) FROM diabetes;"
},
{
"input": "Find all patients with glucose levels higher than 150 in the diabetes dataset.",
"query": "SELECT * FROM diabetes WHERE Glucose > 150;"
},
{
"input": "Count how many patients have diabetes (Outcome = 1) in the diabetes dataset.",
"query": "SELECT COUNT() FROM diabetes WHERE Outcome = 1;"
},
{
"input": "List patients older than 50 years in the diabetes dataset.",
"query": "SELECT * FROM diabetes WHERE Age > 50;"
},
{
"input": "Find the maximum blood pressure in the diabetes dataset.",
"query": "SELECT MAX(BloodPressure) FROM diabetes;"
}
]

def _prepare_db(file_directory):
    file_dir_list = os.listdir(file_directory)
    db_path = "data/csv_xlsx_sqldb.db"
    db_path = f"sqlite:///{db_path}"
    engine = create_engine(db_path)
    for file in file_dir_list:
        full_file_path = os.path.join(file_directory, file)
        file_name, file_extension = os.path.splitext(file)
        if file_extension == ".csv":
            df = pd.read_csv(full_file_path)
        elif file_extension == ".xlsx":
            df = pd.read_excel(full_file_path)
        else:
            raise ValueError("The selected file type is not supported")
        df.to_sql(file_name, engine, index=False, if_exists='replace')  # Added if_exists='replace'

def get_response(question):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    system_prefix = """You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the given tools. Only use the information returned by the tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    If the question does not seem related to the database, just return "I don't know" as the answer.

    Here are some examples of user inputs and their corresponding SQL queries:"""

    stored_csv_xlsx_sqldb_directory="data/csv_xlsx_sqldb.db"
    if os.path.exists(stored_csv_xlsx_sqldb_directory):
        engine = create_engine(f"sqlite:///{stored_csv_xlsx_sqldb_directory}")
        db = SQLDatabase(engine=engine)
        example_selector = SemanticSimilarityExampleSelector.from_examples(examples,
                                                                           OpenAIEmbeddings(),
                                                                           FAISS,
                                                                           k=5,
                                                                           input_keys=["input"])
        few_shot_prompt = FewShotPromptTemplate( example_selector=example_selector,
                                                example_prompt=PromptTemplate.from_template(
                                                    "User input: {input}\nSQL query: {query}"),
                                                    input_variables=["input", "dialect", "top_k"],
                                                    prefix=system_prefix,
                                                    suffix="")
        full_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate(prompt=few_shot_prompt),
                                                        ("human", "{input}"),
                                                        MessagesPlaceholder("agent_scratchpad")])
        agent = create_sql_agent(llm=llm,
                                 db=db,
                                 prompt=full_prompt,
                                 verbose=True,
                                 agent_type="openai-tools")
        response = agent.invoke({"input": question})
        return response["output"]


def main():
    file_dir_list="C:/Users/Yash mishra/Desktop/PROJECTS/Rag GPT/Rag GPT/data/csv_xlsx"
    with st.spinner("Loading..."):
        _prepare_db(file_dir_list)
    
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
        response = get_response(user_question)
        
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




