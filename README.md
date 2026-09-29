# Sports Data Assistant

An LLM-powered Streamlit app that answers natural-language questions about a sports game file (HTML box score / play-by-play) and generates an accompanying data visualization on request.

**Status:** Not actively maintained. Built mid-2025 against an earlier LangChain API (`langchain`, `langchain_community`, `langchain_core` with LCEL pipe syntax). LangChain has since restructured several of these modules, so dependencies will need updating before this runs as-is. Kept here as a portfolio reference — see "Known limitations" below.

## What it does

Given an HTML game file, the app:
1. Extracts and cleans the text content (BeautifulSoup)
2. Routes the user's question to one of several specialized chains (content Q&A, summarization, file-structure/code questions, or visualization requests)
3. Returns a natural-language answer, and/or generates and executes Python/matplotlib code to produce a chart
4. Traces every chain call with Langfuse for observability

## Architecture

**1. Ingestion** (`html_extraction`, `html_file_extraction`)
HTML is parsed with BeautifulSoup; script/style/nav tags are stripped and the remaining text is passed downstream as both raw HTML and cleaned plain text.

**2. Query routing** (`router`, in `summary_AI.py`)
A classification chain (`JsonOutputParser`) tags each incoming question as `content_summary`, `content_qa`, `file_structure`, `code_generation`, or `other`, then dispatches to the matching chain.

**3. Content Q&A** — the question is answered directly from the extracted document text via a single LLM call (Gemini 2.5 Flash), grounded strictly in that text.

**4. Summarization** — two interchangeable LangChain summarization strategies are implemented: `stuff` (whole document in one call) and `map_reduce` (the document is chunked with `RecursiveCharacterTextSplitter` and summarized in parts, then combined) — useful for game files too large for a single context window.

**5. Visualization pipeline** (`engineer_AI.py`) — a two-agent handoff:
   - A "business analyst" agent reads the question and the game file, and decides *what* to visualize and *how* to extract the needed data — but is explicitly prompted not to write code.
   - An "engineer" agent takes those requirements and generates a runnable Python/matplotlib script (assumes a pre-loaded `html_content` variable, targets pinned library versions for consistency).
   - The generated code is executed in a controlled namespace (`exec`), and any resulting `matplotlib.figure.Figure` objects are captured and returned to the Streamlit UI.

**6. Frontend** (`st_app_prototype.py`) — a Streamlit chat interface: users upload a game file, ask a question in chat, and see the text answer plus generated chart rendered inline.

**7. Observability** — every chain invocation is wrapped with a Langfuse `CallbackHandler` for tracing and debugging.

## Tech stack

- **LangChain** (LCEL / Runnable pipe syntax) + **LangChain Community** loaders
- **Gemini 2.5 Flash** (`langchain_google_genai`) for all generation and routing
- **BeautifulSoup4 / lxml** for HTML parsing
- **Matplotlib** for generated visualizations
- **Streamlit** for the UI
- **Langfuse** for LLM call tracing

## A note on "RAG"

This project is commonly described as retrieval-augmented generation, but as implemented, retrieval is **not** vector-based: `Chroma` and `GoogleGenerativeAIEmbeddings` are imported but not actually wired into an embedding/similarity-search step anywhere in the pipeline. Instead, the full extracted document text is passed directly into each prompt. For the size of a single game file, this works fine in practice, but it's worth being precise about this distinction rather than describing it as classic vector-store RAG.

## Known limitations / what I'd improve

- Update to current LangChain package structure and pin exact versions in `requirements.txt`
- Wire up the imported `Chroma` vector store for genuine similarity-search retrieval, or remove the unused imports
- Sandbox the `exec()` call in `run_code_and_collect_figures` more strictly (currently runs LLM-generated code with `__builtins__` available)
- Add automated tests / a small evaluation set (e.g., comparing generated summaries or chart data against known box-score values)
- Consolidate `summary_AI.py` and `engineer_AI.py`, which currently duplicate several imports, prompt-chain patterns, and the `html_extraction` function
