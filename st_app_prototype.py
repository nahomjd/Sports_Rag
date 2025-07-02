#Beta App
import streamlit as st
import json
import requests

#langchain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import BSHTMLLoader # BeautifulSoup HTML Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import BSHTMLLoader # <--- HTML LOADER
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from summary_AI import html_extraction, html_file_extraction, ask_intelligent_assistant, summarize_html_with_langchain_map_reduce
from engineer_AI import run_visualization


raw_html_content, extracted_text_content = html_extraction()
uploaded_file = None
st.title('Your sports data assistant')
#st.write(summarize_html_with_langchain_map_reduce())

prompt = st.chat_input('How can I help you today')    

#Chat History
if 'messages' not in st.session_state:
    st.session_state.messages= []
    
#Display chat messages from history on app run
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
            

uploaded_files = st.sidebar.file_uploader(
    "Choose a CSV file", accept_multiple_files=True
)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    file_type = uploaded_file.name.split('.')[-1]
    
if prompt:
    if uploaded_file is None:
        st.write(ask_intelligent_assistant(prompt, extracted_text_content, raw_html_content))
        fig, error = run_visualization(prompt,raw_html_content)
        if error:
            st.write(error)
        print(fig)
        st.pyplot(fig[0])
    else:
        if file_type == 'html':
            extracted_text_content = html_file_extraction(bytes_data)
            
            st.write(ask_intelligent_assistant(prompt, extracted_text_content, bytes_data))
            fig, error = run_visualization(prompt,bytes_data)
        if error:
            st.write(error)
        print(fig)
        st.pyplot(fig[0])
        
