# 📄 RAG PDF Chatbot  

*Transform your PDFs into interactive conversations powered by AI.*  

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)  
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)  
![LangChain](https://img.shields.io/badge/AI-LangChain-green?logo=chainlink)  
![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-yellow?logo=google)  
![License](https://img.shields.io/badge/License-MIT-purple)  

---

## 📑 Table of Contents  
- [Overview](#overview)  
- [Features](#features)  
- [Tech Stack](#tech-stack)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
- [Usage](#usage)  
- [Screenshots](#screenshots)  
- [Future Improvements](#future-improvements)  
- [Contributing](#contributing)  
- [License](#license)  

---

## 🔎 Overview  

**RAG PDF Chatbot** is an **AI-powered assistant** that lets you upload PDF documents and chat with them.  
It combines **retrieval-augmented generation (RAG)** with **Google Gemini** to answer context-aware questions from your documents, all inside a modern **Streamlit interface**.  

This project simplifies the way you extract insights from PDFs: upload → process → chat.  

---

## ✨ Features  

- 🔍 **Semantic Search** – Retrieves the most relevant chunks using HuggingFace embeddings  
- 📂 **Document Processing** – Extracts and splits PDF content into knowledge chunks  
- 🤖 **Conversational AI** – Google Gemini for human-like contextual responses  
- 💬 **Chat Memory** – Maintains conversation history for natural multi-turn dialogue  
- 🎨 **Dark Mode UI** – Sleek chat interface with custom CSS, AI 🤖 and User 👤 avatars  
- ⚡ **Efficient Retrieval** – Vector store powered by FAISS  

---

## 🛠 Tech Stack  

- [Python 3.10](https://www.python.org/)  
- [Streamlit](https://streamlit.io/) – frontend interface  
- [LangChain](https://www.langchain.com/) – chaining & RAG pipeline  
- [FAISS](https://github.com/facebookresearch/faiss) – vector similarity search  
- [HuggingFace](https://huggingface.co/) – embeddings (`all-mpnet-base-v2`)  
- [Google Gemini API](https://ai.google.dev/) – LLM for responses  

---

## 🚀 Getting Started  

### ✅ Prerequisites  
- Programming Language: **Python 3.10+**  
- Package Manager: **pip**  
- Google Gemini API key  

### 📥 Installation  

1. Clone the repository:  
```bash
git clone https://github.com/your-username/rag-pdf-chatbot.git
cd rag-pdf-chatbot

### Install dependencies:
'''bash
pip install -r requirements.txt

