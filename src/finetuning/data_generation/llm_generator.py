# src/finetuning/data_generation/llm_generator.py

import os
import re
import json
import time
import random
from tqdm import tqdm
import requests

# This is the master prompt template that instructs the LLM on how to generate a
# synthetic training example. It's a "few-shot" prompt because it includes a
# detailed example of the desired output.
# Key instructions for the LLM include:
# - Synthesizing information across multiple text chunks, not just extracting.
# - Generating a realistic user query.
# - Classifying the query's intent ('routing_decision').
# - Crucially, creating a structured `metadata_extraction` JSON object with a complete,
#   pre-defined schema, using `null` for unavailable information.
# - Providing a final, synthesized answer to the query.
OPTIMIZED_SYNTHESIS_PROMPT = """
### YOUR ROLE ###
You are an expert data generator for fine-tuning a financial chatbot. Your task is to read a GROUP of 5 related text chunks and create complex training examples that require synthesizing information across them.

### YOUR TASK ###
Based on the "CONTEXT GROUP" provided, generate a list of 1 or 2 JSON objects. Your goal is quality and complexity, not quantity.

### INSTRUCTIONS ###
1.  **Synthesize, Don't Just Extract:** Your primary goal is to create questions that REQUIRE combining information from MULTIPLE chunks in the provided group.
2.  **user_query**: Invent a realistic, multi-faceted user question based on the context.
3.  **routing_decision**: Analyze your invented query and classify it as "DOCUMENT_SEARCH" or "HYBRID".
4.  **metadata_extraction**: ### <-- CHANGE: THIS IS THE MOST IMPORTANT INSTRUCTION ###
    - Based on the query, create the JSON metadata.
    - **EVERY** object inside the "focuses" list **MUST** contain the following keys: `ticker`, `year`, `quarter`, `normalized_doc_type`, `source_type`, `original_file_name`, `exchange`, `date`, `time`, `company_name`.
    - If the information for a key is not available in the query or context, you **MUST** use `null` as its value. Do not omit any keys.
    - Standardize all quarter-related fields to use the key `"quarter"`.
5.  **context**: This MUST be the ENTIRE group of 5 chunks I provided.
6.  **final_response**: Write the perfect, concise answer, making sure to explicitly combine facts from different chunks.
7.  **Follow the Example:** Use the structure and quality demonstrated in the example below as your guide.

### EXAMPLE OF PERFECT OUTPUT (WITH FULL SCHEMA) ###
{{
  "user_query": "Based on their 2019 10-K, how does Oracle's Java software platform relate to its hardware offerings like Engineered Systems?",
  "routing_decision": "DOCUMENT_SEARCH",
  "metadata_extraction": {{
    "focuses": [
      {{
        "ticker": "ORCL",
        "year": 2019,
        "quarter": null,
        "normalized_doc_type": "10-K",
        "source_type": null,          // <-- CHANGE: Example now includes all keys
        "original_file_name": null,
        "exchange": null,
        "date": null,
        "time": null,
        "company_name": null
      }}
    ]
  }},
  "context": [
    {{"text": "standard. We believe the Java programming language and platform together represent one of the most popular and powerful development environments...", "metadata": {{"ticker": "ORCL", "year": "2019", "filing_type": "10-K"}}}},
    {{"text": "by enterprise organizations building custom applications or consuming Java-based ISV products. Oracle Infrastructure Technologies - Hardware Business Offerings...", "metadata": {{"ticker": "ORCL", "year": "2019", "filing_type": "10-K"}}}},
    {{"text": "work together to deliver improved performance, scalability, availability, security and operational efficiency relative to our competitors' products...", "metadata": {{"ticker": "ORCL", "year": "2019", "filing_type": "10-K"}}}},
    {{"text": "a wide range of server products that are designed for mission-critical enterprise environments and that are key components of our engineered systems...", "metadata": {{"ticker": "ORCL", "year": "2019", "filing_type": "10-K"}}}},
    {{"text": "requirements. Storage Oracle storage products are engineered for the cloud and designed to securely store, manage, protect and archive customers'...", "metadata": {{"ticker": "ORCL", "year": "2019", "filing_type": "10-K"}}}}
  ],
  "final_response": "Oracle's Java platform is a key technology used to build its Middleware and Applications software. This software is then integrated into its hardware offerings, such as Oracle Engineered Systems, which combine Oracle's software with server, storage, and networking hardware to deliver improved performance and efficiency."
}}

The output MUST be a valid JSON list `[{{}}, ...]` and nothing else.

### CONTEXT GROUP TO USE ###
{context_group}
"""

def extract_json_from_response(response_content: str) -> list | None:
    """
    Finds and parses the main JSON list from the LLM's raw text response.

    LLM responses can sometimes include conversational text before or after the
    actual JSON output. This function uses a robust regex to find the outermost
    `[...]` list structure and attempts to parse only that part, making the
    data extraction process more resilient.

    @param response_content: The raw string response from the LLM.
    @return: A list of parsed JSON objects if successful, otherwise None.
    """
    # This regex greedily finds the first '[' and the last ']' in the string.
    match = re.search(r'\[.*\]', response_content, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Handle cases where the extracted string is still not valid JSON.
            print(f"\\nWarning: Failed to parse JSON from model output. Skipping. Content: {json_str[:200]}...")
            return None
    print(f"\\nWarning: No JSON list found in model output. Skipping. Content: {response_content[:200]}...")
    return None

def generate_dataset_safely(
    results_data: dict,
    model: str = "meta-llama/Llama-3-70b-chat-hf",
    output_file: str = "training_dataset.jsonl",
    test_run_limit: int | None = None
):
    """
    Generates a synthetic dataset by calling an LLM API and saves progress continuously.

    This function orchestrates the entire data generation process. It shuffles the
    input context groups to ensure variety, sends them to the LLM API one by one,
    parses the responses, and appends the results to an output file. Opening the
    file in append mode (`'a'`) makes the process fault-tolerant; if the script is
    interrupted, already generated data is not lost.

    @param results_data: A dictionary where keys are filenames and values are lists of context groups.
    @param model: The identifier for the LLM to be used for generation.
    @param output_file: The path to the output .jsonl file.
    @param test_run_limit: If set, limits generation to a small number of groups for testing.
    """
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("TOGETHER_API_KEY environment variable is not set.")
        return

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = "https://api.together.xyz/v1/chat/completions"
    # A fixed delay between API calls is a simple but effective way to respect rate limits.
    DELAY_PER_REQUEST = 2

    # Flatten the input dictionary into a single list of all context groups.
    all_groups = []
    for doc, groups in results_data.items():
        all_groups.extend(groups)

    print(f"Found {len(all_groups)} unique context groups.")
    # Shuffling is important to avoid any bias related to the original order of documents.
    random.shuffle(all_groups)

    groups_to_process = all_groups
    # A feature for quick development and debugging runs.
    if test_run_limit:
        print(f"--- RUNNING IN TEST MODE: Processing only {test_run_limit} groups. ---")
        groups_to_process = all_groups[:test_run_limit]

    print(f"Starting generation for {len(groups_to_process)} groups...")

    # Open the file in "append" mode to ensure that progress is saved after each API call.
    with open(output_file, 'a', encoding='utf-8') as f:
        # Use tqdm to create a progress bar for better user experience.
        for group in tqdm(groups_to_process, desc="Generating dataset"):
            group_str = json.dumps(group, indent=2)
            payload = {
                "messages": [{"role": "user", "content": OPTIMIZED_SYNTHESIS_PROMPT.format(context_group=group_str)}],
                "model": model,
                "temperature": 0.8, # A higher temperature encourages more creative and diverse responses.
                "max_tokens": 4096   # Ensure the model has enough token budget to respond fully.
            }

            try:
                # Make the API call with a generous timeout.
                response = requests.post(url, headers=headers, json=payload, timeout=90.0)
                response.raise_for_status() # Raise an exception for HTTP errors (e.g., 4xx, 5xx).

                response_content = response.json()['choices'][0]['message']['content']
                generated_examples = extract_json_from_response(response_content)
                
                if generated_examples:
                    for example in generated_examples:
                        # The LLM is not asked to include the context in its output to save tokens.
                        # We re-insert the original context group here for completeness.
                        example['context'] = group
                        f.write(json.dumps(example) + '\n')

            except requests.exceptions.HTTPError as e:
                print(f"\\nHTTP Error on group: {e}")
            except Exception as e:
                # A broad exception handler prevents a single failed request from crashing the whole script.
                print(f"\\nAn unexpected error occurred on group: {e}")
            
            # Pause between requests to be polite to the API server.
            time.sleep(DELAY_PER_REQUEST)

    print(f"\\n✅ Generation complete. Data saved to {output_file}")