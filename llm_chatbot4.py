import streamlit as st
import requests
import json
import PyPDF2
import pyperclip
from dotenv import load_dotenv

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

def handle_user_input(question):
    response = st.session_state.conversation({'question':question})
    st.session_state.chat_history = response['chat_history']
    #if st.session_state['generated']: ##TRY THIS
    # Reverse iteration through the list 
    #for i in range(len(st.session_state['generated'])-1, -1, -1):
        # message from streamlit_chat
    #    message(st.session_state['past'][::-1][i], is_user=True, key=str(i) + '_user')
    #    message(st.session_state['generated'][::-1][i], key=str(i))
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            #st.write(message.content, unsafe_allow_html=True)
        else:
            #st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            #st.write(message.content, unsafe_allow_html=True)
            st.markdown(
                f'<div style="display: flex; align-items: center;">'
                f'<img src="https://github.com/marpulli246/llm_pdf_chatbot/blob/main/chatbot.png?raw=true" alt="chatbot.png" class="inline" style="width: 50px; height: 50px; margin-right: 10px;">'
                f'<div>{bot_template.replace("{{MSG}}", message.content)}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            # Copy text feature for each response
            copy_button_key = st.checkbox('Copy', key=f"copy_button_{i}", value=False)
            if copy_button_key:
                response_to_copy = message.content
                pyperclip.copy(response_to_copy)
                st.write("Response copied to clipboard!")
            
# Streamlit Frontend
def main():
    #load_dotenv()
    st.title("LLM Chatbot - Explore your PDF document data")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None    

    #question = st.text_input("Ask anything to your PDF: ")
    question = st.text_input("Ask anything", html_code)
    if question:
        handle_user_input(question)
        
    # Upload PDF and read it
    pdf_file = st.sidebar.file_uploader("Upload your PDF file.", type=['pdf'])

    if st.sidebar.button("OK"):
        with st.spinner("Processing your PDFs..."):

    #if pdf_file:
            context = read_pdf(pdf_file)
            st.sidebar.write("PDF successfully uploaded and read.")
            text_chunks = get_chunk_text(context)
            vector_store = get_vector_store(text_chunks)
            st.session_state.conversation =  get_conversation_chain(vector_store)
        
if __name__ == "__main__":
    main()
