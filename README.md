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
- [Configuration](#configuration)  
- [Project Structure](#project-structure)  
- [Troubleshooting](#troubleshooting)  
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
- Google Gemini API key ([Get one here](https://ai.google.dev/))  

### 📥 Installation  

1. **Clone the repository:**  
```bash
git clone https://github.com/Rajsharma27/DocBot.git
cd DocBot
```

2. **Create a virtual environment:**  
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies:**  
```bash
pip install -r requirements.txt
```

4. **Set up your API key:**  
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

---

## 💬 Usage  

1. **Run the Streamlit app:**  
```bash
streamlit run main.py
```

2. **Open your browser:**  
Navigate to `http://localhost:8501`

3. **Upload a PDF:**  
Click the file uploader and select your PDF document

4. **Ask questions:**  
Type your questions in the chat input and get instant answers based on your document

**Example Questions:**
- "What is the main topic of this document?"
- "Summarize the key points"
- "Find information about [specific topic]"

---

## ⚙️ Configuration  

Edit `main.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Size of text chunks for embedding |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `TOP_K` | 3 | Number of relevant chunks to retrieve |
| `MODEL_NAME` | `gemini-pro` | Google Gemini model version |

---

## 📁 Project Structure  

```
DocBot/
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
├── .gitignore             # Git ignore rules
├── README.md              # This file
└── LICENSE                # MIT License
```

---

## 🐛 Troubleshooting  

**Issue:** `ModuleNotFoundError: No module named 'streamlit'`  
**Solution:** Run `pip install -r requirements.txt`

**Issue:** `Google API key error`  
**Solution:** Verify your `.env` file has the correct `GOOGLE_API_KEY`

**Issue:** `PDF upload fails`  
**Solution:** Ensure the PDF is not corrupted and under 50MB

**Issue:** `Slow response time`  
**Solution:** Reduce `CHUNK_SIZE` or use a smaller PDF for faster processing

---

## 🎯 Future Improvements  

- [ ] Support for multiple file formats (DOCX, TXT, PPT)
- [ ] PDF highlighting for cited chunks
- [ ] Export chat history as PDF
- [ ] Multi-document RAG support
- [ ] User authentication & session management
- [ ] Faster embeddings with GPU acceleration
- [ ] Custom prompt templates
- [ ] Chat history persistence to database

---

## 🤝 Contributing  

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License  

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author  

**Raj Sharma**  
GitHub: [@Rajsharma27](https://github.com/Rajsharma27)  

---

## 📧 Support  

Have questions or issues? Open an issue on the [GitHub repository](https://github.com/Rajsharma27/DocBot/issues).

---

**⭐ If you find this helpful, please star the repository!**
