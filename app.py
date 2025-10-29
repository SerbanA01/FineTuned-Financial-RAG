import os
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the main pipeline function from our refactored src directory
from src.rag.pipeline import get_rag_response

# --- 1. Configuration ---
# All configuration variables are centralized here for easy management.
# In a production environment, these should be loaded from environment variables
# or a dedicated configuration management system.

# Qdrant vector database configuration
QDRANT_HOST = "your_qdrant_host"  # The hostname or IP address of your Qdrant instance.
QDRANT_PORT = 6333 # The port Qdrant is running on (usually 6333 for HTTP or 443 for HTTPS).
COLLECTION_NAME = "financial_sp500_local_final_v2" # The name of the collection to search.
# The initial number of documents to retrieve from the vector database. This number
# should be relatively high to ensure the reranker has a rich set of candidates to choose from.
TOP_K = 70

# Local paths to the various models used in the RAG pipeline.
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANKER_PATH = "BAAI/bge-reranker-large"
INTENT_MODEL_PATH = "models/model_intent" # Path to the fine-tuned intent classification model.
METADATA_MODEL_PATH = "models/model_metadata" # Path to the fine-tuned query parsing model.
FINAL_ANSWER_MODEL_PATH = "models/model_final_answer" # Path to the fine-tuned answer generation model.

# Automatically select the best available device (GPU if available, otherwise CPU).
# This ensures the application leverages hardware acceleration without manual configuration.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Using device: {DEVICE} ---")

# --- 2. Load Models and Clients (Global Objects) ---
# These objects are initialized once when the application starts up. This is a crucial
# optimization, as loading these large models is time-consuming. By keeping them in memory,
# subsequent API requests can be processed much faster.
print("--- Initializing Models and Clients ---")

# Embedding Model: Converts text queries into vector embeddings for similarity search.
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

# Reranker Model: A more powerful model that re-scores the initial search results for better relevance.
print(f"Loading reranker model: {RERANKER_PATH}")
reranker_model = CrossEncoder(RERANKER_PATH, device=DEVICE)

# Fine-tuned Intent Model: Classifies the user's query into predefined categories.
print(f"Loading intent model from: {INTENT_MODEL_PATH}")
intent_tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_PATH)
# Using bfloat16 for reduced memory usage and potentially faster inference on supported hardware.
# `device_map="auto"` lets the Transformers library intelligently distribute the model across available devices (e.g., multiple GPUs).
intent_model = AutoModelForCausalLM.from_pretrained(
    INTENT_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16
)

# Fine-tuned Metadata Model: Parses the query to extract structured information like tickers and dates.
print(f"Loading metadata model from: {METADATA_MODEL_PATH}")
metadata_tokenizer = AutoTokenizer.from_pretrained(METADATA_MODEL_PATH)
metadata_model = AutoModelForCausalLM.from_pretrained(
    METADATA_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16
)
# Define special tokens that signal the end of a generated sequence for this model.
metadata_terminators = [
    metadata_tokenizer.eos_token_id,
    metadata_tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Fine-tuned Final Answer Generation Model: Synthesizes the final answer from the retrieved context.
print(f"Loading final answer model from: {FINAL_ANSWER_MODEL_PATH}")
final_answer_tokenizer = AutoTokenizer.from_pretrained(FINAL_ANSWER_MODEL_PATH)
final_answer_model = AutoModelForCausalLM.from_pretrained(
    FINAL_ANSWER_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16
)

# Qdrant Client: The connection to the vector database.
print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
# Perform a startup check to ensure the connection to Qdrant is valid.
# If this fails, the API will still run but will return an error for any query.
try:
    qdrant_client.get_collections()
    print("✅ Successfully connected to Qdrant!")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not connect to Qdrant: {e}")
    qdrant_client = None

print("✅ All models and clients initialized and ready.")

# --- 3. FastAPI Application ---
# Initialize the FastAPI application instance.
app = FastAPI()

class QueryRequest(BaseModel):
    """
    Defines the expected structure and data type for the incoming request body.
    FastAPI uses this model for automatic validation and documentation.
    """
    query: str

@app.get("/")
def read_root():
    """
    A simple health check endpoint to confirm that the API server is running.

    @return: A JSON object with the server's status.
    """
    return {"status": "RAG API is running"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    """
    The main endpoint for submitting a query to the RAG pipeline.

    This endpoint receives a user's question, passes it to the full RAG pipeline,
    and returns the synthesized answer.

    @param request: The incoming request body, validated against the QueryRequest model.
    @return: A JSON object containing the final answer, or an error message if the
             database connection is unavailable.
    """
    if not qdrant_client:
        return {"error": "Database connection is not available."}

    # This single function call encapsulates the entire RAG logic, passing in the
    # pre-loaded models and clients for efficient processing.
    response_text = get_rag_response(
        query=request.query,
        qdrant_client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        intent_model=intent_model,
        intent_tokenizer=intent_tokenizer,
        metadata_model=metadata_model,
        metadata_tokenizer=metadata_tokenizer,
        metadata_terminators=metadata_terminators,
        generation_model=final_answer_model,
        generation_tokenizer=final_answer_tokenizer,
        top_k=TOP_K
    )
    return {"answer": response_text}

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    # This block allows the script to be run directly with `python app.py`.
    # Uvicorn is a high-performance ASGI server used to run the FastAPI application.
    # "0.0.0.0" makes the server accessible on the network, not just localhost.
    uvicorn.run(app, host="0.0.0.0", port=8000)