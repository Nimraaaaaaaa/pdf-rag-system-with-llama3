import streamlit as st
import os
import requests
from PyPDF2 import PdfReader
import docx
import torch
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from pydantic import BaseModel, Field  # ✅ BaseModel import already hai
from dotenv import load_dotenv  # ✅ import dotenv

# ✅ Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")  # 🔐 fetch Groq API key

# ✅ Modified ChatGroq class
class ChatGroq(LLM, BaseModel):
    model_name: str = Field(default="llama3-8b-8192")
    temperature: float = Field(default=0.7)
    callbacks: None = None  # ✅ Field define krdiya

    def _call(self, prompt, stop=None):
        import requests

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ Groq API key not found. Please check your .env file.")

        api_url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": 512
        }

        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"Groq API Error: {response.text}")

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name, "temperature": self.temperature}

    @property
    def _llm_type(self):
        return "groq"

# ❗❗❗ Baaki sab tumhara code same ka same chalega — main(), get_files_text(), get_pdf_text(), sab kuch same. ❗❗❗




def main():
    if groq_api_key:
        pass  # ✅ Token loaded successfully
    else:
        st.error("❌ Groq API key not found. Please check your .env file.")
        return

    st.set_page_config(page_title="Chat with your file")
    st.header("DocumentGPT with Llama 3 🤖")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
        process = st.button("Process")

    if process:
        files_text = get_files_text(uploaded_files)
        st.write("✅ File loaded...")

        text_chunks = get_text_chunks(files_text)
        st.write("✅ File chunks created...")

        vectorstore = get_vectorstore(text_chunks)
        st.write("✅ Vector Store Created...")

        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.session_state.processComplete = True

    if st.session_state.processComplete:
        user_question = st.chat_input("Ask a question about your files.")
        if user_question:
            handle_userinput(user_question)

def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        else:
            text += get_csv_text(uploaded_file)
    return text

def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(file):
    doc = docx.Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    text = ' '.join(allText)
    return text

def get_csv_text(file):
    return "CSV reading is not implemented yet."

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=100,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base

def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0.7
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    response_container = st.container()
    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))

if __name__ == '__main__':
    main()
