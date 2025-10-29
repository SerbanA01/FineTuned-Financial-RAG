import json
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from .schemas import QueryFocus

def refine_with_llm(
    original_query: str,
    provisional_rule_based_focuses: List, # Kept for signature consistency, but not used in the new logic
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    terminators: list
) -> List[QueryFocus]:
    """
    Refines a user's natural language query into a structured list of QueryFocus objects
    by leveraging a fine-tuned Large Language Model (LLM).

    This function constructs a detailed prompt instructing the LLM to act as a financial
    analyst. It passes the user's query and expects a structured JSON response containing
    the extracted entities (ticker, year, quarter, document type). This approach allows for
    more nuanced and context-aware extraction than purely rule-based methods, especially
    for complex queries with pronouns or multiple distinct requests.

    If the LLM fails to produce valid JSON or encounters an error, this function will
    fall back to the provided `provisional_rule_based_focuses`.

    @param original_query: The raw natural language query from the user.
    @param provisional_rule_based_focuses: A list of focuses extracted by a preliminary,
                                           non-LLM method. Used as a fallback.
    @param model: The loaded Hugging Face Causal LM for generation.
    @param tokenizer: The tokenizer corresponding to the model.
    @param terminators: A list of token IDs that signify the end of a generated sequence.
    @return: A list of structured QueryFocus objects parsed from the LLM's response.
             Returns the provisional focuses if the LLM fails.
    """
    # This system prompt is engineered to constrain the LLM's output. It provides a persona,
    # detailed instructions, and few-shot examples to guide the model into producing a
    # reliable, structured JSON output. The core instruction is to prioritize the user's
    # original query over any preliminary rule-based analysis, making the LLM the final arbiter.
    system_prompt = f"""
You are an expert financial data analyst assistant. Your task is to re-evaluate a user's query to produce a definitive, corrected list of financial data focuses.
You are provided with a provisional, rule-based extraction; treat it as a HINT, but the user's original query is the absolute source of truth. You MUST override the provisional extraction if it is incorrect or incomplete based on a full reading of the query.

Instructions:
1.  **Analyze the User's Intent:** Carefully read the ENTIRE user query to understand each distinct request. The query is the ultimate source of truth.
2.  **Extract Entities for Each Request:**
    a.  **Ticker**: Identify the company ticker.
    b.  **Year**: Identify the associated year.
    c.  **Raw_Quarter**: Identify the quarter (1, 2, 3, or 4).
    d.  **Normalized_DocumentType**: This is a critical step. Search for document type keywords within the same phrase or clause as the company/ticker.
        - Keywords for "10-K": "10-K", "10K", "annual report", "annual filing".
        - Keywords for "10-Q": "10-Q", "10Q", "quarterly report", "quarterly filing".
        - Keywords for "EARNINGS_TRANSCRIPT": "earnings transcript", "earnings call", etc.
3.  **Handle Pronouns:** Pay close attention to pronouns like "its", "their", "the company's". Resolve them to the most recent preceding company mentioned. For example, in "...Apple... see its 10-Q...", "its" refers to Apple.
4.  **Strict Association:** You MUST be diligent in linking the document type to the correct company. If a query says "...Apple's 2023 Q2 10-Q...", the `normalized_doc_type` for the Apple focus MUST be "10-Q".
5.  **Handle Ambiguity:** If a company is mentioned conversationally (e.g., "I ate an apple"), only create a financial data request if it is clearly linked to financial terms like "10-Q", "annual report", etc.

Output Format:
Your entire response MUST be a single JSON object. Do not add any text before or after it. The structure must be:

{{
"user_query": "The original user query string here",
"metadata": {{
"focuses": [
{{
"ticker": "...",
"year": ...,
"raw_quarter": "...",
"normalized_doc_type": "..."
}}
]
}}
}}
<|eot_id|>
Examples:
Example 1: Multiple Companies, Same Period

user_query: "How did Nvidia's Q2 sales compare to AMD's Q2 results in 2023?"
Expected Output:

{{
"user_query": "How did Nvidia's Q2 sales compare to AMD's Q2 results in 2023?",
"metadata": {{
"focuses": [
{{
"ticker": "NVDA",
"year": 2023,
"raw_quarter": "Q2",
"normalized_doc_type": 10-Q
}},
{{
"ticker": "AMD",
"year": 2023,
"raw_quarter": "Q2",
"normalized_doc_type": 10-Q
}}
]
}}
}}
<|eot_id|>
Example 2: Multiple Companies, Multiple Document Types

user_query: "I need to see MSFT's 2024 Q1 quarterly report and also find GOOG's annual report for 2023."
Expected Output:

{{
"user_query": "I need to see MSFT's 2024 Q1 quarterly report and also find GOOG's annual report for 2023.",
"metadata": {{
"focuses": [
{{
"ticker": "MSFT",
"year": 2024,
"raw_quarter": "Q1",
"normalized_doc_type": "10-Q"
}},
{{
"ticker": "GOOG",
"year": 2023,
"raw_quarter": null,
"normalized_doc_type": "10-K"
}}
]
}}
}}
<|eot_id|>
user_query: "{original_query}"
Expected Output:
"""
    prompt = system_prompt
    response_content = ""
    try:
        # Prepare the prompt for the model using its specific chat template.
        messages = [{"role": "user", "content": prompt}]
        input_data = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        # Keep track of the input length to separate the prompt from the generated response.
        input_ids_length = input_data.shape[1]

        # Generate the response from the model.
        # do_sample=False encourages deterministic, high-confidence output suitable for JSON generation.
        outputs = model.generate(
            input_data,
            max_new_tokens=1500,
            eos_token_id=terminators,
            do_sample=False,
        )

        # Decode the generated tokens, skipping the input prompt part.
        response_content = tokenizer.decode(outputs[0, input_ids_length:], skip_special_tokens=True)

        # Clean the response. Models sometimes wrap JSON in markdown code blocks (` ```json ... ``` `).
        if response_content.startswith("```json"):
            response_content = response_content[7:-3] if response_content.endswith("```") else response_content[7:]
        response_content = response_content.strip()

        # Parse the cleaned string as JSON. This is a critical step where failure is possible.
        llm_data = json.loads(response_content)
        refined_provisional_focuses = []

        # Safely extract the list of 'focuses' from the expected JSON structure.
        focuses_list = llm_data.get("metadata", {}).get("focuses", [])
        if not isinstance(focuses_list, list):
            # If the structure is not as expected, log a warning and return an empty list.
            print(f"Warning: 'focuses' key not found or not a list in LLM response. Response: {llm_data}")
            return []

        # Iterate through the extracted focus items and cast them into our strong-typed QueryFocus schema.
        for item in focuses_list:
            if isinstance(item, dict) and "ticker" in item:
                # Handle quarter conversion from string (e.g., "Q1") to integer (1).
                quarter_str = item.get("raw_quarter")
                quarter_int = None
                if isinstance(quarter_str, str) and quarter_str.upper().startswith('Q'):
                    try:
                        quarter_int = int(quarter_str[1:])
                    except (ValueError, IndexError):
                        # Gracefully handle cases where the quarter format is unexpected.
                        print(f"Warning: Could not parse quarter string: {quarter_str}")
                        quarter_int = None

                # Create and append a validated QueryFocus object.
                refined_provisional_focuses.append(QueryFocus(
                    ticker=item.get("ticker"),
                    year=item.get("year"),
                    quarter=quarter_int,
                    doc_type=item.get("normalized_doc_type")
                ))
            else:
                print(f"Warning: LLM returned an invalid item in focuses list: {item}")

        return refined_provisional_focuses

    except json.JSONDecodeError:
        # This is the primary failure mode: the LLM did not output a parsable JSON string.
        # In this case, we log the failure and fall back to the rule-based results.
        print(f"Error: LLM did not return valid JSON.")
        print(f"Raw Response from model:\n---\n{response_content}\n---")
        return provisional_rule_based_focuses
    except Exception as e:
        # Catch any other unexpected errors during the process and fall back.
        print(f"An unexpected error occurred during LLM refinement: {e}")
        return provisional_rule_based_focuses