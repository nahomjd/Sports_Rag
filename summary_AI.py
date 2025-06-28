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
    temperature=0.8,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key = GOOGLE_API_KEY,
    # other params...
)

def summarize_html_with_langchain_stuff(file_path=HTML_FILE_PATH, llm_model=LLM):
    print(f"\n--- Summarizing with 'stuff' method: {file_path} ---")
    loader = BSHTMLLoader(file_path) # Use BSHTMLLoader
    try:
        documents = loader.load() # BSHTMLLoader extracts text content
    except Exception as e:
        return f"Error loading HTML document: {e}"

    if not documents or not documents[0].page_content.strip():
        return "Error: No content loaded from the HTML document."

    print(f"   Loaded content length (chars): {len(documents[0].page_content)}")
    # Add token check if needed here

    chain = load_summarize_chain(llm_model, chain_type="stuff", verbose=False)
    try:
        summary_output = chain.invoke(documents)
        return summary_output['output_text']
    except Exception as e:
        return f"Error during LangChain 'stuff' summarization: {e}"
        
 
def summarize_html_with_langchain_map_reduce(file_path=HTML_FILE_PATH, llm_model=LLM):
    print(f"\n--- Summarizing with 'map_reduce' method: {file_path} ---")
    loader = BSHTMLLoader(file_path) # Use BSHTMLLoader
    try:
        documents = loader.load()
    except Exception as e:
        return f"Error loading HTML document: {e}"

    if not documents or not documents[0].page_content.strip():
        return "Error: No content loaded from the HTML document."

    print(f"   Loaded content length (chars): {len(documents[0].page_content)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, # Adjust based on LLM and content. Gemini 1.5 can handle larger.
        chunk_overlap=500,
        separators=["\n\n", "\n", " ", ""] # Good for extracted text
    )
    # documents from BSHTMLLoader is a list (usually with one Document for the whole HTML's text)
    # We split this Document object.
    split_docs = text_splitter.split_documents(documents)
    print(f"   Document split into {len(split_docs)} chunks for map_reduce.")

    if not split_docs:
        return "Error: Failed to split document into manageable chunks."

    chain = load_summarize_chain(llm_model, chain_type="map_reduce", verbose=False)
    try:
        summary_output = chain.invoke(split_docs)
        return summary_output['output_text']
    except Exception as e:
        return f"Error during LangChain 'map_reduce' summarization: {e}"
        
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
    
class QueryType(JsonOutputParser):
    def get_format_instructions(self) -> str:
        return 'Return a JSON object with a single key "query_type" and one of the following values: "content_summary", "content_qa", "structure_code", "other".'
        
def router(LLM, Querytype = QueryType()):
    router_prompt_template = """Classify the user's question into one of the following categories:
    - "content_summary": If the question asks for a summary, overview, or general understanding of the document's textual content.
    - "content_qa": If the question asks a specific factual question that can likely be answered from the document's textual content.
    - "file_structure": If the question is about a file structure, such as CSV or HTML structure.
    - "code_generation: If the question specifically asks to write code to use on the file.
    - "other": If the question does not fit well into the above categories or is ambiguous.

    User Question: {question}

    Classification (JSON with "query_type" key):
    """
    router_prompt = PromptTemplate.from_template(router_prompt_template)
    router_chain = router_prompt | LLM | QueryType() # QueryType is a JsonOutputParser
    return router_chain

def content_qa_prompt_template(LLM, extracted_text_content, raw_html_content):
    content_qa_prompt_template = """You are a business analyst turn into a sports analyst, an expert at gaining insight from sports information. You will do three things before responding
    1. Take in the question and determine what type of insights would be benificial ontop of answering the question.
    2. Come up with the best format using markdown or not to give the response.
    3. Ensure that the data needed for answering the question is in the file.
    
    Utilize the information in the document to answer questions. Make inferences based on the available data. When possible, use stats/data backed by the document.
    Your job has one part:
    1) Answer the user's question based *only* on the provided Document Text

    Document Text:
    ---
    {document_text}
    ---

    User Question: {question}

    Content Answer:"""
    content_qa_prompt = PromptTemplate.from_template(content_qa_prompt_template)
    content_qa_chain = (
        {
            "document_text": RunnableLambda(lambda x: extracted_text_content), # Always provide full extracted text
            "question": RunnablePassthrough() # Pass the original question
        }
        | content_qa_prompt
        | LLM
        | StrOutputParser()
    )
    return content_qa_chain
 
def content_qa_prompt_html_template(LLM, extracted_text_content, raw_html_content):
    content_qa_prompt_html_template = """You are a business analyst turn into a sports analyst, an expert at gaining insight from sports information. You will do three things before responding
    1. Take in the question and determine what type of insights would be benificial ontop of answering the question.
    2. Come up with the best format using markdown or not to give the response.
    3. Ensure that the data needed for answering the question is in the file.
    
    Your job has one part:
    1) Answer the user's question based *only* on the provided Document Text

    Raw HTML Source Code (FOR YOUR REFERENCE ONLY, DO NOT INCLUDE IN ANSWER):
    ---
    {raw_html_source}
    ---

    User Question: {question}

    Content Answer:"""
    content_qa_html_prompt = PromptTemplate.from_template(content_qa_prompt_html_template)
    content_qa_html_chain = (
        {
            "raw_html_source": RunnableLambda(lambda x: raw_html_content),
            "question": RunnablePassthrough() # Pass the original question
        }
        | content_qa_html_prompt
        | LLM
        | StrOutputParser()
    )
    return content_qa_html_chain
    
    
def file_structure_template(LLM, extracted_text_content, raw_html_content):
    file_structure_template = """You are a helpful AI Sports Data assistant. Use the context of the question and, if possible, the provided file to determine the sport the question is asking.
    Answer to the best of your ability about the structure of a file and what specific terms might mean. When data is mentioned or format it is reference to the content of the file not the file itself unless otherwise noted.
    If you are unsure about a specific term or statistic name, ask the user if there is any more available context they can provide or the term's definition.

    Raw HTML Source Code (FOR YOUR REFERENCE ONLY, DO NOT INCLUDE IN ANSWER):
    ---
    {raw_html_source}
    ---

    Document Text:
    ---
    {document_text}
    ---

    User Question/Task: {question}

    Python Script (assuming `HTML_FILE_PATH` variable exists):"""
    file_structure_prompt = PromptTemplate.from_template(file_structure_template)
    file_structure_chain = (
        {
            "document_text": RunnableLambda(lambda x: extracted_text_content), # Always provide full extracted text
            "raw_html_source": RunnableLambda(lambda x: raw_html_content), # Always provide full raw HTML
            "question": RunnablePassthrough() # Pass the original question
        }
        | file_structure_prompt
        | LLM
        | StrOutputParser()
    )
    return file_structure_chain
    
def code_generation_template(LLM, extracted_text_content, raw_html_content):
    code_generation_template = """Although the question asks to write code, answer the underlying question to the best of your ability without writing code.
    
    You are a business analyst turn into a sports analyst, an expert at gaining insight from sports information. You will do three things before responding
    1. Take in the question and determine what type of insights would be benificial ontop of answering the question.
    2. Come up with the best format using markdown or not to give the response.
    3. Ensure that the data needed for answering the question is in the file.

    Use the context of the question. Answer the user's question based *only* on the provided Document Text.
    
    Utilize the information in the document to answer questions. Make inferences based on the available data. When possible, use stats/data backed by the document.
    Your job has one part:
    1) Answer the user's question based *only* on the provided Document Text

    Document Text:
    ---
    {document_text}
    ---

    User Question/Task: {question}

    Python Script (assuming `HTML_FILE_PATH` variable exists):"""
    code_generation_prompt = PromptTemplate.from_template(code_generation_template)
    code_generation_chain = (
        {
            "document_text": RunnableLambda(lambda x: extracted_text_content), # Always provide full extracted text
            "question": RunnablePassthrough() # Pass the original question
        }
        | code_generation_prompt
        | LLM
        | StrOutputParser()
    )
    return code_generation_chain
    
def ask_intelligent_assistant(user_question, extracted_text_content, raw_html_content, LLM=LLM):
    print(f"\n❓ User Question: '{user_question}'")

    # Step 1: Route the question
    print("   Routing question...")
    try:
        route_result = router(LLM).invoke({"question": user_question})
        query_type = route_result.get("query_type")
        print(f"   Determined query type: {query_type}")
    except Exception as e:
        print(f"   🔴 Error in routing: {e}. Defaulting to general approach.")
        query_type = "other" # Fallback

    # Step 2: Execute specialized chain
    response = ""
    if query_type in ["content_summary", "content_qa"]:
        print("   Processing as content question...")
        # Check if extracted_text_content + question + prompt is too long for context
        # (Add context window check here if necessary for this specific chain's input)
        response = content_qa_prompt_template(LLM, extracted_text_content, raw_html_content).invoke(user_question)
    elif query_type == "file_structure":
        print("   Processing as structure/code question...")
        # Check if raw_html_content + question + prompt is too long for context
        # (Add context window check here; raw HTML can be very large)
        response = file_structure_template(LLM, extracted_text_content, raw_html_content).invoke(user_question)
    elif query_type == "code_generation":
        print("   Processing as structure/code question...")
        # Check if raw_html_content + question + prompt is too long for context
        # (Add context window check here; raw HTML can be very large)
        response = code_generation_template(LLM, extracted_text_content, raw_html_content).invoke(user_question)
    else: # "other" or fallback
        print("   Processing as 'other' or fallback...")
        # For "other", you might have a more general prompt or ask for clarification
        # For simplicity, let's try a general prompt using extracted text
        general_prompt_text = f"Please try to answer the following question based on the document's content: {user_question}\n\nDocument Text:\n{extracted_text_content}"
        try:
            response = LLM.invoke(general_prompt_text).content
        except Exception as e:
            response = f"Sorry, I encountered an error trying to process that: {e}"

    #print("\n💡 Assistant's Response:")
    #print(response)
    return response
