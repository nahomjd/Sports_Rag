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

from summary_AI import html_extraction, ask_intelligent_assistant, summarize_html_with_langchain_map_reduce
from engineer_AI import run_visualization


raw_html_content, extracted_text_content = html_extraction()

st.title('Your sports data assistant')
#st.write(summarize_html_with_langchain_map_reduce())

prompt = st.chat_input('How can I help you today')
if prompt:
    st.write(ask_intelligent_assistant(prompt, extracted_text_content, raw_html_content))
    fig, error = run_visualization(prompt,raw_html_content)
    if error:
        st.write(error)
    print(fig)
    st.pyplot(fig[0])
    

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
    st.write("filename:", uploaded_file.name)
    st.write(bytes_data)
    
