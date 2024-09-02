# Multi-Format Data Interaction with RAG

Multi-Format Data Interaction is an advanced document and data analysis tool that leverages Retrieval-Augmented Generation (RAG) to allow users to chat with pre-processed PDFs, upload and interact with their own PDFs, generate summaries of PDF documents, query SQL databases, and analyze CSV/XLSX files. This application combines the power of natural language processing, document analysis, and RAG to provide an intuitive and informative user experience across various data formats.

---

## Features

- Chat with pre-processed PDF documents using RAG
- Upload and interact with custom PDF files through RAG-powered conversations
- Generate comprehensive summaries of PDF documents
- Query SQL databases using natural language (Chinook database)
- Analyze CSV/XLSX files with natural language queries (Cancer and Diabetes datasets)
- Upload and query custom CSV/XLSX files
- User-friendly interface with easy navigation

---

## What is RAG?

Retrieval-Augmented Generation (RAG) is a hybrid AI model that combines information retrieval with text generation. It enhances the capabilities of large language models by allowing them to access and utilize external knowledge bases. In our application, RAG enables more accurate and context-aware responses when interacting with various data formats.

---

## Technologies Used

- Python
- Streamlit
- LangChain
- LangChain Agents
- LangSmith
- Few-shot Prompting
- OpenAI GPT
- PyPDF2
- FAISS
- Retrieval-Augmented Generation (RAG)
- SQLite
- Pandas
---

## Results
- Designed a Streamlit app for interactive AI-driven chats and document processing (PDFs, CSVs, XLSXs).
- Integrated RAG for PDF Q&A with OpenAI embedding, FAISS, and LangChain for chat history, using Langsmith.
- Applied SQL agents and few-shot prompting for natural language queries and analysis of uploaded CSV and XLSX files.
- Deployed a Dockerized app via CI/CD pipeline using GitHub Actions on AWS (EC2, ECR, IAM).
