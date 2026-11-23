
# DSA AI Assistant 🤖

An AI-powered Data Structures and Algorithms
Question-Answering Assistant built using
Retrieval-Augmented Generation (RAG).

## 🚀 Project Overview

This project allows users to ask questions about
Data Structures and Algorithms and receive
context-based answers using an AI model.

The system retrieves relevant information from
a 100-page DSA practice book and generates
answers using the Groq API.

## 🛠️ Technologies Used

- Python
- LangChain
- Groq API
- FastEmbed
- Qdrant Vector Database
- PyPDF
- RAG Architecture

## 📚 Knowledge Base

DSA PYQ-Style Practice Book:
- 660 practice questions
- 6 major topic sections
- 100 pages
- Answer key
- Detailed solutions

Topics include:
- Arrays and Linked Lists
- Stacks and Queues
- Trees and BST
- Heaps and Hashing
- Sorting and Searching
- Graphs
- Dynamic Programming
- Algorithms and Complexity

## ⚙️ How It Works

1. Load the DSA PDF.
2. Extract text using PyPDF.
3. Split documents into smaller chunks.
4. Generate embeddings using FastEmbed.
5. Store embeddings in Qdrant.
6. Retrieve relevant document chunks.
7. Generate answers using Groq LLM.

## 💻 Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/DSA-AI-Assistant.git

Move into the project:

cd DSA-AI-Assistant

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Create a .env file and add your Groq API key.

## ▶️ Run the Project

python main.py

Ask a custom question:

python main.py "Explain AVL trees"

## 👨‍💻 Author

Sashakt Agarwal

B.Tech CSE (CC + AI)
