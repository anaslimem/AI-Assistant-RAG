import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain.tools import tools
import os,ast,redis,json,hashlib
from hashlib import md5 as hashlibmd5


# Redis connection setup with error handling
REDIS = redis.Redis(host='localhost', port=6379, db=0)
REDIS.ping()  # Test connection

# Initialize Ollama model
llm = Ollama(model="llama3.1")

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(model="llama3.1")

persist_directory = "./vectorstore"

# Check if the directory exists; if not, create it
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# Check if the directory is empty; if so, add example documents
if not os.listdir(persist_directory):
    # Example documents
    docs = [
        {"text": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn. AI can be used in a variety of applications, including natural language processing, robotics, and decision-making systems."},
        {"text": "Machine learning (ML) is a subset of artificial intelligence (AI) that allows computers to learn from data without being explicitly programmed. The goal of machine learning is to enable computers to improve their performance on a specific task by learning from examples."},
        {"text": "Natural language processing (NLP) is a subfield of AI that deals with the interactions between computers and human languages, in particular how to interpret, generate, and process human language data."},
        {"text": "Deep learning is a subset of machine learning that uses artificial neural networks to model and solve complex problems. Deep learning has been used to achieve state-of-the-art results in a variety of domains, including computer vision, natural language processing, and speech recognition."},
        {"text": "Computer vision is the field of study that deals with extracting and understanding images in digital form. Computer vision can be used in a variety of applications, such as autonomous vehicles, surveillance systems, and medical imaging."},
        {"text": "Speech recognition is the process of converting spoken language into written text. Speech recognition technology has been used for various applications, such as telecommunication, personal assistants, and healthcare."},
        {"text": "Robotics is the field of engineering and technology that designs, develops, and manufactures robots that can perform tasks autonomously or collaboratively with humans."},
        {"text": "Big data refers to vast amounts of structured and unstructured data that can be analyzed to uncover patterns, trends, and associations. In AI, big data is crucial for training more accurate models and making data-driven decisions."}
    ]
    
    # Extract texts from docs
    texts = [doc["text"] for doc in docs]
    
    # Initialize vectorstore with documents
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    # Add texts to the vectorstore
    vectorstore.add_texts(texts)
    
    # Persist the vectorstore for future use
    vectorstore.persist()
else:
    # Load existing vectorstore
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

def cache_query(query: str, response: str):
    """Store the result of the query in Redis."""
    query_hash = hashlibmd5(query.encode('utf-8')).hexdigest() # Unique hash for the query
    REDIS.setex(query_hash, 3600, json.dumps(response)) # Cache with 1 hour expiration

def get_cached_result(query):
    """Retrieve the cached result of the query from Redis."""
    query_hash = hashlib.md5(query.encode('utf-8')).hexdigest() 
    cached_result = REDIS.get(query_hash)
    if cached_result:
        return json.loads(cached_result)
    else:
        return None

def retrieve_context(query: str) -> str:
    """Retrieve the most similar context from the vectorstore."""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])

def ask_rag(query: str) -> str:
    """Ask the Retrieval Augmented Generation model a question."""
    # Check if we already have a cached result for the query
    cached_result = get_cached_result(query)
    if cached_result:
        return cached_result

    # No cache hit, retrieve context and ask the model
    docs = retrieve_context(query)
    prompt = f"""
    Using only the information below, provide a direct answer to the question.
    Information: {docs}
    Question: {query}
    """
    result = llm(prompt)
    # Store the results in the cache
    cache_query(query, result)
    # Return the result
    return result

def make_readable_paragraph(paragraph: str) -> str:
    """Rewrite the paragraph to improve readability."""
    prompt = f"""
    Rewrite the following text to be clear and concise while keeping its meaning intact,provide a direct answer to the question.
    text: {paragraph}
    """
    return llm(prompt)

def browser(q: str) -> str:
    """Search DuckDuckGo and return a concise answer."""
    search_tool = DuckDuckGoSearchRun()
    try:
        results_str = search_tool.run(q, max_results=3)
        # Check if results_str is a string and attempt to parse it
        if isinstance(results_str, str) and results_str.startswith('['):
            results = ast.literal_eval(results_str)
            # Ensure results is a list of dicts with 'body' key
            if isinstance(results, list) and all(isinstance(r, dict) and 'body' in r for r in results):
                formatted_results = "\n\n".join([res['body'] for res in results])
            else:
                formatted_results = results_str  # Fallback to raw string if parsing fails
        else:
            formatted_results = results_str  # Use raw output if not a list
        return make_readable_paragraph(formatted_results).strip()
    except (SyntaxError, ValueError, AttributeError) as e:
        return f"Search failed due to an issue with the results: {str(e)}"

browser_tool = Tool(
    name="DuckDuckGo Search",
    func=browser,
    description="Searches the web using DuckDuckGo"
)

# Streamlit UI
st.set_page_config(page_title="AI Assistant", layout="wide")

st.title("AI Chat")

mode = st.sidebar.radio("Select AI Model", ["LLM", "RAG", "Web Search"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.answers = {}

query = st.text_input("Ask something:", "")

if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.answers = {}
    st.experimental_rerun()

if st.button("Ask AI"):
    if query.strip():
        with st.spinner("Processing..."):
            try:
                if mode == "Web Search":
                    response = browser(query)
                elif mode == "RAG":
                    response = ask_rag(query)
                else:
                    response = llm(query)
                if not response.strip():
                    response = "No valid answer found."
                st.markdown(f"### Answer:")
                st.write(response)
                st.session_state.chat_history.append(query)
                st.session_state.answers[query] = response
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a query.")

st.sidebar.title("Chat History")
for prompt in st.session_state.chat_history:
    if st.sidebar.button(prompt):
        st.markdown(f"### Answer:")
        st.write(st.session_state.answers[prompt])
