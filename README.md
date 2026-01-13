# 📄 RAG PDF Chatbot 🤖

A powerful interactive chatbot that leverages **Retrieval Augmented Generation (RAG)** to answer questions about your PDF documents. Built with LangChain, Google Generative AI, and Streamlit for a seamless user experience.

## Features

- 📤 **Multi-PDF Upload**: Upload multiple PDF documents simultaneously
- 🤖 **AI-Powered Responses**: Uses Google's Gemini 2.5 Pro model for intelligent question answering
- 🔍 **Semantic Search**: Employs advanced embedding models to find relevant document sections
- 💬 **Conversational Memory**: Maintains chat history for context-aware responses
- 🎨 **Modern UI**: Custom-styled Streamlit interface with dark theme and gold accents
- ⚡ **Real-time Processing**: Instant processing and response generation

## How It Works

The chatbot uses a Retrieval Augmented Generation (RAG) pipeline:

1. **PDF Extraction**: Extracts text from uploaded PDF files
2. **Text Chunking**: Splits documents into manageable chunks (300 characters)
3. **Embedding Generation**: Converts text chunks into vector embeddings using HuggingFace's `sentence-transformers/all-mpnet-base-v2`
4. **Vector Storage**: Stores embeddings in a FAISS vector database for fast retrieval
5. **Intelligent Retrieval**: Retrieves the 5 most relevant chunks based on user queries
6. **Response Generation**: Uses Google Generative AI to synthesize answers from retrieved documents

## Prerequisites

- Python 3.8+
- Google API Key for Gemini access
- HuggingFace account (for embeddings)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/docbot.git
cd docbot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root directory:

```env
GOOGLE-API-KEY=your_google_api_key_here
```

Get your Google API Key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Usage

### Running the Application

```bash
streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Chatbot

1. **Upload PDFs**: Click on the file uploader and select one or multiple PDF files
2. **Process Documents**: Click the "Process PDF" button to extract and index the documents
3. **Ask Questions**: Once processing is complete, type your questions in the chat input
4. **Get Answers**: The chatbot will retrieve relevant content and provide AI-generated answers

## Project Structure

```
docbot/
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not committed)
└── README.md              # This file
```

## Dependencies

- **streamlit** - Web application framework
- **python-dotenv** - Environment variable management
- **pypdf** - PDF text extraction
- **langchain** - LLM orchestration and chains
- **langchain-google-genai** - Google Generative AI integration
- **langchain-community** - Community integrations
- **langchain-huggingface** - HuggingFace embeddings
- **sentence-transformers** - Advanced embedding models
- **faiss-cpu** - Vector similarity search (installed via dependencies)

## Configuration

### Text Splitting Parameters

Edit the `text_split_into_chunks()` function to customize:
- `chunk_size`: Size of text chunks (default: 300)
- `chunk_overlap`: Overlap between chunks (default: 0)
- `separator`: Text separator for splitting (default: '\n')

### Vector Retrieval

Edit the `create_conversational_chain()` function to customize:
- `search_kwargs={"k": 5}`: Number of relevant chunks to retrieve (default: 5)
- `model`: Change from `gemini-2.5-pro` to another available model

## API Models

- **LLM**: Google Generative AI - Gemini 2.5 Pro
- **Embeddings**: HuggingFace - sentence-transformers/all-mpnet-base-v2

## Limitations

- Maximum context length depends on your selected LLM model
- Large PDFs may take longer to process
- Requires internet connection for API calls
- Memory usage scales with document size

## Future Enhancements

- [ ] Support for additional document formats (DOCX, TXT, etc.)
- [ ] Custom embedding model selection
- [ ] Persistent conversation history/database
- [ ] Cost tracking for API usage
- [ ] Response source citations
- [ ] Batch processing mode
- [ ] Custom system prompts

## Troubleshooting

### "GOOGLE-API-KEY not found"
- Ensure your `.env` file is in the project root directory
- Verify the key name matches exactly: `GOOGLE-API-KEY`
- Restart the Streamlit app after updating `.env`

### PDF Processing Takes Too Long
- Large PDFs will naturally take longer to process
- Consider splitting very large PDFs into smaller files
- Increase chunk size to process documents faster (may reduce accuracy)

### Low Quality Responses
- Try adjusting the number of retrieved chunks (increase `k` in search_kwargs)
- Ensure PDF text extraction is working properly
- Verify document quality and readability

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [Google Generative AI](https://ai.google.dev/)
- UI built with [Streamlit](https://streamlit.io/)
- Embeddings from [HuggingFace](https://huggingface.co/)

## Contact

For questions or support, please open an issue on GitHub.

---

**Happy questioning! 🚀**
