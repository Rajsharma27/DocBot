import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE-API-KEY")

st.set_page_config(page_title="RAG PDF Chatbot ", layout="wide")


# Uploading multiple pdfs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        loader = PdfReader(pdf)
        for page in loader.pages:
            text += page.extract_text()
    return text

# Splitting the text 
def text_split_into_chunks(text):
    splitter = CharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 0,
        separator = '\n',
    )
    chunks = splitter.split_text(text)
    return chunks


# Storing the embeddings
def store_vector_embeds(embeddings, chunks):
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# Creating the converstaion chain
def create_conversational_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-pro')

    memory = ConversationBufferMemory(
        memory_key = 'chat_history',
        return_messages = True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(search_kwargs={"k": 5}),
        memory = memory
    )
    return conversation_chain

# Handling user query
def handle_user_query(conversation_chain, user_query):
    response = conversation_chain({"question": user_query})
    return response['answer']


def main():
    # Custom CSS
    st.markdown("""
    <style>
        /* General App Background */
        .stApp {
            background-color: #1E1E2F;
            color: #EAEAEA;
        }

        /* Title Styling */
        .stMarkdown h1 {
            color: #FFD700;
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }

        /* File Uploader Styling */
        .stFileUploader {
            background-color: #2E2E3F;
            border: 1px solid #FFD700;
            border-radius: 10px;
            padding: 10px;
        }

        /* Buttons Styling */
        .stButton button {
            background-color: #FFD700;
            color: #1E1E2F;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: bold;
            padding: 10px 20px;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #FFC107;
        }

        /* Chat Input Styling */
        .stTextInput > div > div > input {
            background-color: #2E2E3F;
            color: #EAEAEA;
            border: 1px solid #FFD700;
            border-radius: 10px;
            padding: 10px;
        }

        /* Chat Messages */
        .chat-message {
            display: flex;
            align-items: flex-start;
            margin: 10px 0;
            padding: 10px;
            border-radius: 12px;
            max-width: 80%;
        }

        .chat-message.user {
            background-color: #2E3B55;
            color: #FFD700;
            margin-left: auto;
            border: 1px solid #FFD700;
        }

        .chat-message.ai {
            background-color: #2E2E3F;
            color: #EAEAEA;
            margin-right: auto;
            border: 1px solid #444;
        }

        /* Icons inside chat */
        .chat-icon {
            font-size: 1.5rem;
            margin-right: 10px;
        }

        .chat-content {
            flex: 1;
        }

        /* Spinner Styling */
        .stSpinner {
            color: #FFD700;
        }
    </style>
""", unsafe_allow_html=True)


    # Setting the title
    st.title("📄 RAG PDF Chatbot 🤖")
    st.write("Upload your PDFs and ask questions about them!")

    #Handling the file uploading

    pdf_docs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if pdf_docs and st.button("Process PDF"):
        with st.spinner("Processing...⌚"):
            raw_text = get_pdf_text(pdf_docs)
            chunks = text_split_into_chunks(raw_text)
            embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
            vector_store = store_vector_embeds(embeddings, chunks)
            st.session_state.conversation = create_conversational_chain(vector_store)
        st.success("PDFs processed successfully")



    if "conversation" in st.session_state:
        user_query = st.chat_input("Ask any question from your PDFs...")
        if user_query:
            answer = handle_user_query(st.session_state.conversation, user_query)
            st.chat_message("User").write(user_query)
            st.chat_message("AI assistant").write(answer)



if __name__ == "__main__":
    main()
