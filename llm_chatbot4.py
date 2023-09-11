import streamlit as st
import requests
import json
import PyPDF2
import pyperclip
from dotenv import load_dotenv
import clipboard

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import bot_template, user_template, html_code, css

# Read PDF and return text
def read_pdf(file):
    pdf_file = PyPDF2.PdfReader(file)
    text = ""
    st.session_state.conversation = ""
    st.session_state.chat_history = ""
    for i in range(len(pdf_file.pages)):
        text += pdf_file.pages[i].extract_text()
    return text

#Splitting the text
def get_chunk_text(text):   #NEW
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# For OpenAI Embeddings
def get_vector_store(text_chunks): #NEW
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore
              
def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory,
    )
    return conversation_chain             

def on_copy_click(text):
    # st.session_state.copied.append(text)
    clipboard.copy(text)

def handle_user_input(question):
    response = st.session_state.conversation({'question':question})
    st.session_state.chat_history = response['chat_history']
    chat_length = len(st.session_state.chat_history)
    for i in range(int(chat_length/2)):
       index = chat_length-(i*2+2)
       message = st.session_state.chat_history[index]
       st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True) 
       message = st.session_state.chat_history[index+1]        
       st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
       #st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True) 
       #st.button("📋", on_click=on_copy_click, args=(message.content,), key=i)
       #copy_button_key = st.checkbox('Copy', key=f"copy_button_{i}", value=False)
       #if copy_button_key:
           
       #     response_to_copy = message.content
       #     pyperclip.copy(response_to_copy)
       #     st.write("Response copied to clipboard!")
           
           
# Streamlit Frontend
def main():
    #load_dotenv()
    st.title("LLM Chatbot - Explore your PDF document data")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ""

    unique_input_key = "question input"
    question = st.text_input("Ask anything to your PDF: ", key="unique_input_key", accept_multiple_files=True)
    #question = st.text_input("Ask anything", html_code)
    if question:
        handle_user_input(question)
        unique_input_key = ""
        
    # Upload PDF and read it
    pdf_file = st.sidebar.file_uploader("Upload your PDF file.", type=['pdf'])

    st.sidebar.write("PDF Pre-Process")
    if st.sidebar.button("OK"):
        #st.session_state.conversation = None
        #st.session_state.chat_history = None
        with st.spinner("Processing the PDFs..."):

    #if pdf_file:
            context = read_pdf(pdf_file)
            st.sidebar.write("PDF successfully uploaded and read.")
            text_chunks = get_chunk_text(context)
            vector_store = get_vector_store(text_chunks)
            st.session_state.conversation =  get_conversation_chain(vector_store)
        
if __name__ == "__main__":
    main()
