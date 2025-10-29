import yfinance as yf
from datetime import datetime, timedelta
from collections import OrderedDict
from typing import List

# Type hinting for complex objects
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

# Internal RAG modules
from .intent import classify_query_intent
from .query_parser import process_query_with_llm_refinement
from .market_data import fetch_price_chunk, make_price_search_result, needs_price_retrieval
from .utils import extract_date_from_query, remove_chunk_references
from .process import process_and_retrieve_with_augmentation
from .reranker import finalize_and_rerank_results
from .generation import generate_final_answer


def extract_ticker_fast(
    query: str,
    metadata_model: AutoModelForCausalLM,
    metadata_tokenizer: AutoTokenizer,
    metadata_terminators: list
) -> List[str]:
    """
    Quickly extracts only the ticker symbols from a user query.

    This function leverages the full query processing pipeline but discards all
    extracted information except for the tickers. It's a specialized utility for
    when only the company identifiers are needed, such as in the market-data-only flow.

    @param query: The user's raw query string.
    @param metadata_model: The LLM used for entity extraction.
    @param metadata_tokenizer: The tokenizer for the metadata model.
    @param metadata_terminators: A list of terminator token IDs for the model.
    @return: A list of ticker strings found in the query.
    """
    focuses, _ = process_query_with_llm_refinement(
        query,
        metadata_model=metadata_model,
        metadata_tokenizer=metadata_tokenizer,
        metadata_terminators=metadata_terminators
        )
    return [focus.ticker for focus in focuses]


def process_market_data_only(
    query: str,
    metadata_model: AutoModelForCausalLM,
    metadata_tokenizer: AutoTokenizer,
    metadata_terminators: list
) -> str:
    """
    Handles queries classified as needing only market data, bypassing the RAG pipeline.

    This provides a fast path for simple queries like "what is the price of AAPL?".
    It extracts tickers, fetches data directly from Yahoo Finance, and formats a
    concise, direct answer without performing any document retrieval or complex
    LLM-based answer generation.

    @param query: The user's raw query string.
    @param metadata_model: The LLM used for entity extraction.
    @param metadata_tokenizer: The tokenizer for the metadata model.
    @param metadata_terminators: A list of terminator token IDs for the model.
    @return: A formatted string containing the requested market data.
    """
    tickers = extract_ticker_fast(query, metadata_model, metadata_tokenizer, metadata_terminators)
    date_str = extract_date_from_query(query)

    if not tickers:
        return "I couldn't identify any stock tickers in your query. Please specify a company or ticker symbol."

    target_date = datetime.now()
    if date_str:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            # If date parsing fails, silently fall back to the current date.
            pass

    results = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            query_lower = query.lower()
            # Simple keyword matching to decide what data to fetch.
            if any(keyword in query_lower for keyword in ["price", "stock price", "closing", "current"]):
                hist = stock.history(start=target_date.date(), end=(target_date + timedelta(days=1)).date())
                if not hist.empty:
                    close_price = hist['Close'].iloc[-1]
                    results.append(f"**{ticker}** on {target_date.strftime('%Y-%m-%d')}: ${close_price:.2f}")
                else:
                    results.append(f"**{ticker}**: No price data for {target_date.strftime('%Y-%m-%d')}")
            else: # A simple fallback for other general market data queries.
                info = stock.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice', 'N/A')
                results.append(f"**{ticker}**: Current Price: ${current_price:.2f}" if isinstance(current_price, float) else f"**{ticker}**: Price data not available.")
        except Exception as e:
            results.append(f"**{ticker}**: Error fetching data - {str(e)}")

    return "\n".join(results) if results else "Unable to fetch requested market data."


def process_document_search_only(
    query: str, qdrant_client: QdrantClient, collection_name: str, embedding_model: SentenceTransformer,
    reranker_model: CrossEncoder, metadata_model: AutoModelForCausalLM, metadata_tokenizer: AutoTokenizer,
    metadata_terminators: list, generation_model: AutoModelForCausalLM, generation_tokenizer: AutoTokenizer,
    top_k: int
) -> str:
    """
    Handles queries that require only document retrieval and synthesis (standard RAG).

    This function executes the core RAG pipeline:
    1. Parse the query to understand the user's need.
    2. Retrieve relevant document chunks from the vector database.
    3. Rerank the results for relevance.
    4. Generate a final answer based on the retrieved context.

    @param query: The user's query.
    @param qdrant_client: The client for the vector database.
    @param collection_name: The name of the Qdrant collection to search.
    @param embedding_model: The sentence-transformer model for embedding the query.
    @param reranker_model: The cross-encoder model for reranking results.
    @param metadata_model: The LLM for query parsing.
    @param metadata_tokenizer: The tokenizer for the metadata model.
    @param metadata_terminators: Terminator tokens for the metadata model.
    @param generation_model: The LLM for final answer synthesis.
    @param generation_tokenizer: The tokenizer for the generation model.
    @param top_k: The initial number of results to retrieve from the vector DB.
    @return: A synthesized, citable answer as a string.
    """
    results, _ = process_and_retrieve_with_augmentation(
        query=query, client=qdrant_client, collection_name=collection_name, query_model=embedding_model,
        metadata_model=metadata_model, metadata_tokenizer=metadata_tokenizer,
        metadata_terminators=metadata_terminators, min_results_k=top_k
    )

    final_results_for_generation = OrderedDict()
    for focus, candidates in results.items():
        top_k_reranked = finalize_and_rerank_results(
            query=query, candidate_results=candidates, reranker=reranker_model, final_k=7
        )
        final_results_for_generation[focus] = top_k_reranked

    if final_results_for_generation:
        response = generate_final_answer(
            original_query=query, retrieved_results=final_results_for_generation,
            llm_model=generation_model, llm_tokenizer=generation_tokenizer
        )
        return remove_chunk_references(response)
    else:
        return "❌ No relevant documents found."


def process_hybrid_query(
    query: str, qdrant_client: QdrantClient, collection_name: str, embedding_model: SentenceTransformer,
    reranker_model: CrossEncoder, metadata_model: AutoModelForCausalLM, metadata_tokenizer: AutoTokenizer,
    metadata_terminators: list, generation_model: AutoModelForCausalLM, generation_tokenizer: AutoTokenizer,
    top_k: int
) -> str:
    """
    Handles queries that require both document search and live market data.

    This function follows the standard RAG pipeline but adds a crucial step: after
    retrieving and reranking document chunks, it fetches relevant market data
    (e.g., stock price) and injects it into the context before sending it to the
    final generation LLM. This allows the model to synthesize answers from both
    textual and numerical sources.

    @param query: The user's query.
    ... (params are identical to process_document_search_only) ...
    @return: A synthesized answer combining info from documents and market data.
    """
    results, _ = process_and_retrieve_with_augmentation(
        query=query, client=qdrant_client, collection_name=collection_name, query_model=embedding_model,
        metadata_model=metadata_model, metadata_tokenizer=metadata_tokenizer,
        metadata_terminators=metadata_terminators, min_results_k=top_k
    )

    final_results_for_generation = OrderedDict()
    for focus, candidates in results.items():
        top_k_reranked = finalize_and_rerank_results(
            query=query, candidate_results=candidates, reranker=reranker_model, final_k=7
        )
        final_results_for_generation[focus] = top_k_reranked

    # Augment the retrieved results with fresh market data if the query implies it.
    if needs_price_retrieval(query):
        for focus, doc_list in final_results_for_generation.items():
            # Determine the appropriate "as of" date for the price data. If the query
            # specifies a year, use the end of that year to provide historical context.
            asof_date_str = f"{focus.year}-12-31" if focus.year else datetime.utcnow().strftime("%Y-%m-%d")
            asof_date = datetime.strptime(asof_date_str, "%Y-%m-%d")
            price_payload = fetch_price_chunk(ticker=focus.ticker, asof=asof_date)
            if price_payload:
                # Prepend the market data to the list of documents so it appears first in the context.
                doc_list.insert(0, make_price_search_result(price_payload))

    if final_results_for_generation:
        response = generate_final_answer(
            original_query=query, retrieved_results=final_results_for_generation,
            llm_model=generation_model, llm_tokenizer=generation_tokenizer
        )
        return remove_chunk_references(response)
    else:
        return "❌ No relevant information found."


def get_rag_response(
    query: str, qdrant_client: QdrantClient, collection_name: str, embedding_model: SentenceTransformer,
    reranker_model: CrossEncoder, intent_model: AutoModelForCausalLM, intent_tokenizer: AutoTokenizer,
    metadata_model: AutoModelForCausalLM, metadata_tokenizer: AutoTokenizer, metadata_terminators: list,
    generation_model: AutoModelForCausalLM, generation_tokenizer: AutoTokenizer, top_k: int
) -> str:
    """
    Main entry point for the RAG pipeline, routing queries based on user intent.

    This function acts as the central controller. It first classifies the user's
    query to determine its nature (market data only, document search only, or hybrid).
    Based on this classification, it delegates the query to the appropriate specialized
    processing function to generate a response.

    @param query: The user's query.
    ... (params include all necessary models, clients, and settings for the full pipeline) ...
    @return: The final response string to be shown to the user.
    """
    # Step 1: Classify the user's intent.
    intent = classify_query_intent(query, intent_model, intent_tokenizer)
    print(f"Query classified as: {intent}")

    # Step 2: Route the query to the correct handler based on the intent.
    if intent == "MARKET_DATA_ONLY":
        return process_market_data_only(query, metadata_model, metadata_tokenizer, metadata_terminators)

    elif intent == "DOCUMENT_SEARCH":
        return process_document_search_only(
            query, qdrant_client, collection_name, embedding_model, reranker_model,
            metadata_model, metadata_tokenizer, metadata_terminators,
            generation_model, generation_tokenizer, top_k
        )

    elif intent == "HYBRID":
        return process_hybrid_query(
            query, qdrant_client, collection_name, embedding_model, reranker_model,
            metadata_model, metadata_tokenizer, metadata_terminators,
            generation_model, generation_tokenizer, top_k
        )

    else:
        # Fallback case if the intent classification returns an unexpected value.
        print(f"Warning: Received unexpected intent '{intent}'. Defaulting to HYBRID.")
        return process_hybrid_query(
            query, qdrant_client, collection_name, embedding_model, reranker_model,
            metadata_model, metadata_tokenizer, metadata_terminators,
            generation_model, generation_tokenizer, top_k
        )