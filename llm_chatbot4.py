import streamlit as st
import requests
import json
import PyPDF2
import pyperclip
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
#from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# Your OpenAI API key
api_key = OPENAI_API_KEY

# Initialize OpenAI API
#def initialize_openai():
#    requests.post("https://api.openai.com/v1/engines/davinci-codex/completions")

# Read PDF and return text
def read_pdf(file):
    pdf_file = PyPDF2.PdfFileReader(file)
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
    # For Huggingface Embeddings
    # embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore


# Call OpenAI API and get the model's response
def get_response(messages):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': messages
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
    if response.status_code == 200:
        return json.loads(response.text)['choices'][0]['message']['content']
    else:
        return f"Error: {response.text}"

# Display conversation history
def display_conversation(messages):
    for index, message in enumerate(messages):
        if message["role"] == "user":
            st.markdown(f'<div style="background-color: #555555; color: white; padding: 10px; border-radius: 10px; \
                        margin-bottom: 5px; font-family: "Roboto, sans-serif";">{message["content"]}</div>', unsafe_allow_html=True)
        elif message["role"] == "assistant":
            st.markdown(f'<div style="background-color: #333333; color: white; padding: 10px; border-radius: 10px; \
                        margin-bottom: 9px; font-family: "Roboto, sans-serif";">{message["content"]}</div>', unsafe_allow_html=True)
            # Copy text feature for each response
            #copy_button_key = f"copy_{index}"  # Generate a unique key for each copy button
            #copy_button_key = st.toggle('Copy',label_visibility="hidden")
            #if st.button(copy_button_key):
            copy_button_key = st.checkbox('Copy', key=f"copy_button_{index}", value=False, label_visibility = "hidden")
            if copy_button_key:
                response_to_copy = message["content"]
                pyperclip.copy(response_to_copy)
                st.write("Copied to clipboard!") 
                
def get_conversation_chain(vector_store):
    
    # OpenAI Model
    llm = ChatOpenAI()
    # HuggingFace Model
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )
    return conversation_chain                
                
            
# Streamlit Frontend
def main():
    st.title("LLM Chatbot - Explore your document data")
  
    st.write(unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    question = st.text_input("Ask anything to your PDF: ")
    if question:
        handle_user_input(question)
        
    # Upload PDF
    #pdf_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    pdf_file = st.sidebar.file_uploader("Upload a PDF file", type=['pdf'])
    if pdf_file:
        context = read_pdf(pdf_file)
        st.write("PDF successfully uploaded and read.")
        text_chunks = get_chunk_text(context)
        vector_store = get_vector_store(text_chunks)
        st.session_state.conversation =  get_conversation_chain(vector_store)
     

if __name__ == "__main__":
    main()
