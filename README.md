# AI Assistant

This project implements an AI-driven assistant using LLMs (Large Language Models) and a web search integration to answer user queries.

## Features
- **LLM**: Answer questions using a large language model.
- **RAG (Retrieval Augmented Generation)**: Retrieve relevant information from a vector store and generate answers.
- **Web Search**: Perform DuckDuckGo searches to provide up-to-date information.

## Technologies Used
- Streamlit
- LangChain
- Chroma Database
- Redis
- Ollama LLM

## Installation

1. Clone this repository:
```
git clone https://github.com/anaslimem/AI-Assistant.git

```

2. Install the necessary dependencies:

```
pip install -r requirements.txt

```

3. Run the Streamlit app:

```
streamlit run app.py

```
4. Download ollama from google and install the model:

```
ollama run llama3.1

```
