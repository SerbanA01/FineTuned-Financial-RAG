from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from .process import SearchResult
from .query_parser_utils.schemas import QueryFocus

def format_context_for_llm(
    retrieved_results: Dict["QueryFocus", List["SearchResult"]]
) -> str:
    """
    Serializes the retrieved search results into a single, formatted string.

    This function takes the structured search results and transforms them into a
    human-readable text block. Each piece of retrieved text ("chunk") is clearly
    labeled with a unique index (e.g., "[CHUNK 1]") and its source metadata.
    This structured format is crucial for the downstream LLM to trace information
    back to its source, enabling accurate citations in the final answer.

    @param retrieved_results: A dictionary mapping a `QueryFocus` to the list
                              of `SearchResult` objects found for it.
    @return: A single string containing all the formatted context, or a message
             indicating that no documents were found.
    """
    if not retrieved_results:
        return "No relevant documents were found."

    context_str = "CONTEXT:\n"
    context_str += "The following document chunks were retrieved as relevant to the user's query:\n\n"

    chunk_index = 1
    # Group results by the original query focus to maintain logical context.
    for focus, results in retrieved_results.items():
        if not results:
            continue

        context_str += f"--- Documents related to: {focus} ---\n"
        for res in results:
            payload = res.point.payload
            tier_info = res.tier

            # Construct a detailed source string for clear attribution.
            source_info = f"{payload.get('source_type', '')}"
            if payload.get('filing_category'):
                source_info += f" ({payload.get('filing_category').upper()})"
            if payload.get('year'):
                source_info += f", Year: {payload.get('year')}"
            if payload.get('quarter'):
                 source_info += f", Quarter: {payload.get('quarter')}"

            # Each chunk is explicitly numbered to allow for precise citation (e.g., [Source: CHUNK 1]).
            context_str += f"[CHUNK {chunk_index}] - Source: {source_info}\n"
            context_str += f"Retrieval Method: {tier_info}\n"
            context_str += f"Content: \"\"\"\n{payload.get('chunk_text', '')}\n\"\"\"\n\n"
            chunk_index += 1

    return context_str.strip()


def generate_final_answer(
    original_query: str,
    retrieved_results: Dict[QueryFocus, List[SearchResult]],
    llm_model: AutoModelForCausalLM,
    llm_tokenizer: AutoTokenizer
) -> str:
    """
    Generates a final, synthesized answer using an LLM, based on the retrieved context.

    This function orchestrates the final step of the RAG pipeline. It formats the
    retrieved search results, constructs a detailed prompt with strict instructions
    for the LLM, and calls the model to generate a citable, fact-based answer.
    The prompt engineering is critical here to ensure the LLM adheres strictly to
    the provided context and avoids hallucination.

    @param original_query: The user's original, unmodified query.
    @param retrieved_results: The search results retrieved from the vector database.
    @param llm_model: The loaded Hugging Face Causal LM for generation.
    @param llm_tokenizer: The tokenizer corresponding to the model.
    @return: A string containing the final, generated answer. Returns an error
             message if the model is unavailable or an exception occurs.
    """
    if not llm_model or not llm_tokenizer:
        return "Sorry, the final answer generation model is not available."

    # First, serialize all the retrieved data into a single context string.
    formatted_context = format_context_for_llm(retrieved_results)

    if "No relevant documents" in formatted_context:
        return "I could not find any relevant documents to answer your query."

    # This system prompt is the "constitution" for the LLM. It sets the persona,
    # defines the rules of engagement, and provides a template to structure the output.
    # The goal is to constrain the model to only synthesize and cite, not create.
    system_prompt = """
You are a factual data extraction AI. Your job is to answer the user's query based **only** on the provided context. You must follow a flexible template structure.

**Core Rules:**
1.  **Analyze the Query:** First, determine if the user is asking a simple question about one subject or asking to compare multiple subjects.
2.  **Use the Template Flexibly:** Fill out the template below. **Only use the sections that are relevant to the user's query.**
3.  **Be Direct:** Answer the user's question directly and concisely.
4.  **CRITICAL DATE CHECK:** Before answering, compare the time period (year/quarter) in the user's query with the time period of the provided context.
    *   If the context's date **DOES NOT MATCH** the query's date, your "Primary Answer" **MUST** start by stating this. For example: "Information for 2024 was not found in the documents. However, context from 2022 states that..."
    *   Never present information from an old date as if it is for the date requested.
5.  **Cite Everything:** Every single fact must end with its source, like [Source: CHUNK X, Retrieval Method: Y].
6.  **Acknowledge Missing Data:** If the information is not in the context, state "This information is not available in the provided documents."
7.  **DO NOT HALLUCINATE:** Never invent information. Stick strictly to the provided context.
"""

    # The user prompt combines the formatted context with the user's query and the final template,
    # giving the model all the information it needs in a single block.
    user_prompt = f"""{formatted_context}

---
Based **only** on the context provided above, answer the user's query by flexibly filling in the template below. Only use the sections relevant to the query.

**User's Query:** "{original_query}"

**ANSWER TEMPLATE:**

**Primary Answer:**
*   [Provide a direct answer to the user's question. For a simple question, this might be the only section you need. For a comparison, this will be the high-level summary.]

**(Optional) Detailed Breakdown - Subject 1:**
*   **Subject Name:** [Identify the first subject from the query, e.g., "Apple"]
*   **Relevant Information:** [Provide the detailed facts, quotes, or numbers for this subject. Cite everything.]

**(Optional) Detailed Breakdown - Subject 2:**
*   **Subject Name:** [Identify the second subject from the query, e.g., "NVIDIA"]
*   **Relevant Information:** [Provide the detailed facts, quotes, or numbers for this subject. Cite everything.]

**(Optional) Comparison Summary:**
*   **Direct Comparison:** [If the query was a comparison, explain if a direct comparison is possible and summarize the key differences or similarities based on the available data.]
"""

    print("\n--- Sending final context to LOCAL LLM for synthesis ---")

    try:
        # Prepare the full prompt using the model's specific chat template for correctness.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        inputs = llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(llm_model.device)

        # We need the input length to later decode only the newly generated text.
        input_ids_length = inputs.shape[1]
        terminators = [
            llm_tokenizer.eos_token_id,
            llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Generate the text. Using `do_sample=False` makes the output deterministic
        # and less "creative," which is ideal for a factual, RAG-based task.
        outputs = llm_model.generate(
            inputs,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=False
        )

        # Decode only the response part, not the input prompt.
        response_content = llm_tokenizer.decode(outputs[0, input_ids_length:], skip_special_tokens=True)
        return response_content.strip()

    except Exception as e:
        # Gracefully handle any errors during model inference.
        print(f"An error occurred during final answer generation with local model: {e}")
        return "Sorry, an error occurred while trying to generate the answer."