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
import re
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st

import streamlit as st

#API/LLM/environment definitions
LANGCHAIN_TRACING_V2 = 'true'
LANGCHAIN_ENDPOINT = 'https://api.smith.langchain.com'
#LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]

GOOGLE_API_KEY = st.secrets["GOOGLE_GEMNI_API_KEY"]

#Testing Source file
HTML_FILE_PATH = 'Example_game.html'

#Testing on LLM
GENERATION_MODEL_NAME = "gemini-2.5-flash-preview-05-20"

#Define LLM
LLM = ChatGoogleGenerativeAI(
    model=GENERATION_MODEL_NAME,
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key = GOOGLE_API_KEY,
    # other params...
)

code_LLM = ChatGoogleGenerativeAI(
    model=GENERATION_MODEL_NAME,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key = GOOGLE_API_KEY,
    # other params...
)

def BA_prompt_template(LLM, raw_html_content):
    interpreter_template = """You are a business analyst, an expert at interpreting stakeholders and translating them to technical requirements. You will do three things before responding
    1. Take in the question and determine the best visualization to go along with the question.
    2. Come up with the best way to extract the data necessary from the given file, for example, writing Python code to parse an HTML file.
    3. Ensure that the data needed for the visualization is in the given file.

    Use these libraries in your instructions:
    - Pandas
    - Matplotlib
    - Buetifulsoup

    After evaluating the question on the question. Given specific details on the approach the engineer should take in order to write the Python code necessary for the visualization.

    Provided a description of the purpose and pseudo code for the engineer.

    Make sure to set the visual to a variable "fig" not just plotting the visual.

    **DO NOT WRITE ANY CODE**

    Given File:
    ---
    {raw_html_source}
    ---

    User Question: {question}

    **DO NOT WRITE ANY CODE**
    Content Answer:"""
    interpreter_prompt = PromptTemplate.from_template(interpreter_template)
    interpreter_chain = (
        {
            "raw_html_source": RunnableLambda(lambda x: raw_html_content), # Always provide full extracted text
            "question": RunnablePassthrough() # Pass the original question
        }
        | interpreter_prompt
        | LLM
        | StrOutputParser()
    )
    return interpreter_chain
    
def engineer_prompt_template(LLM, raw_html_content):
    engineer_template = """You are an expert python software engineer, you specialize in extracting data and visualizations. The only output given should be a python code block. 
    You will get technical requirements from a business analyst.

    **IMPORTANT INSTRUCTIONS:**
    -   **DO NOT include the provided 'Raw HTML Source Code' as a string literal within the generated Python script.**
    -   Assume the HTML content will be loaded into a variable named `html_content` within the Python script. Your script should operate on this `html_content` variable.
    -   Provide only the Python code block enclosed in ```python ... ```.
    -   The script should be runnable if the `html_content` variable is pre-populated with the HTML.
    -   The script should run and create the visualization without any additions such as calling a function.
    -   Use HTML_FILE_PATH as an existing variable for the HTML file, assume this variable is already set and does not need to be defined.
    -   Open the HTML file with open(HTML_FILE_PATH, encoding="utf-8") as f: soup = BeautifulSoup(f, "lxml") method to open HTML files **DO NOT COMMENT IT OUT**.
    -   When getting values from pandas columns always use .values, for example, df['column_name'].values.

    **IMPORTANT INFORMATION:**
    -   Pandas version being used is 2.1.4.
    -   matplotlib version being used is 3.5.1.
    -   BeautifulSoup version being used is 4.11.1.

    Technical Requirements:
    ---
    {technical_requirements}
    ---

    Given File:
    ---
    {raw_html_source}
    ---

    Code Block:"""
    engineer_template_prompt = PromptTemplate.from_template(engineer_template)
    engineer_prompt_chain = (
        {
            "raw_html_source": RunnableLambda(lambda x: raw_html_content), # Always provide full extracted text
            "technical_requirements": RunnablePassthrough() # Pass the original question
        }
        | engineer_template_prompt
        | code_LLM
        | StrOutputParser()
    )
    return engineer_prompt_chain

def extract_code_blocks(markdown_text):
    code_blocks = []
    pattern = r"```(?P<language>\w*)\n(?P<code>[\s\S]*?)```"
    matches = re.finditer(pattern, markdown_text)
    for match in matches:
        language = match.group("language")
        code = match.group("code").strip()
        code_blocks.append({"language": language, "code": code})
    return code_blocks
    
def run_code_and_collect_figures(code: str, path: str):
    """
    Execute arbitrary Matplotlib code and return the list of Figure objects
    that are still open afterwards.
    """
    # (Optional) wipe previous figures each Streamlit rerun
    plt.close('all')

    safe_globals = {
        "__builtins__": __builtins__,          # replace with whitelist for prod
        "plt": plt,
        "HTML_FILE_PATH": path                 #  ←  pass the value in!
    }
    safe_locals = {}

    # Actually run the user's code
    try:
        exec(code, safe_globals, safe_locals)
    except Exception as e:
        return [], e
    # Strategy A ─ any variable literally called 'fig'
    figs = [safe_locals[k] for k in safe_locals
            if isinstance(safe_locals[k], matplotlib.figure.Figure)]

    # Strategy B ─ still-open figures that user created implicitly
    for num in plt.get_fignums():
        fig = plt.figure(num)
        if fig not in figs:
            figs.append(fig)

    return figs, None
        
def html_extraction(HTML_FILE_PATH=HTML_FILE_PATH):
    print(f"1. Processing HTML from '{HTML_FILE_PATH}'...")
    raw_html_content = ""
    extracted_text_content = ""
    try:
        with open(HTML_FILE_PATH, 'r', encoding='utf-8') as f:
            raw_html_content = f.read()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw_html_content, 'html.parser')
        for unwanted_tag in soup(["script", "style", "nav", "footer", "aside"]): # Customize as needed
            unwanted_tag.decompose()
        extracted_text_content = soup.get_text(separator='\n', strip=True)
    except Exception as e:
        print(f"Error processing HTML: {e}")
    return raw_html_content, extracted_text_content
    
    
def ask_intelligent_engineer(user_question, raw_html_content, LLM=LLM, code_LLM=code_LLM, code=False):
    if code:
        response = engineer_prompt_template(code_LLM, raw_html_content).invoke(user_question)
    else:
        response = BA_prompt_template(LLM, raw_html_content).invoke(user_question)
    
    return response
    
def run_visualization(user_question, raw_html_content):
    response = ask_intelligent_engineer(user_question, raw_html_content, code=False)
    output = ask_intelligent_engineer(response, raw_html_content, code=True)

    match = re.search(r"```python(.*?)```", output, re.DOTALL)
    code_to_run = match.group(1).strip() if match else output.strip()
    print(code_to_run)
    fig, error=run_code_and_collect_figures(code_to_run, HTML_FILE_PATH)
    return fig, error
