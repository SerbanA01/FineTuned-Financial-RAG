# src/finetuning/data_generation/dataset_splitter.py

import json

# This module contains functions to split a single, multi-purpose, raw generated
# dataset into three distinct datasets, each tailored for a specific fine-tuning task:
# 1. Intent Classification: Mapping a user query to a high-level intent.
# 2. Metadata Extraction: Extracting structured search parameters from a user query.
# 3. Final Answer Generation: Generating a coherent, context-aware answer.
# This separation of concerns allows for training specialized models for each step
# in a complex RAG (Retrieval-Augmented Generation) pipeline.

def extract_intent(input_file: str, output_file: str = "intent_classification_dataset.jsonl"):
    """
    Creates a dataset for the intent classification task.

    This function reads a raw generated .jsonl file and extracts only the `user_query`
    and the `routing_decision` (renamed to `intent`) for each record. The resulting
    dataset is used to train a model that can quickly determine the user's high-level
    goal from their query.

    @param input_file: Path to the input .jsonl file containing raw generated data.
    @param output_file: Path to the output .jsonl file for the intent dataset.
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        # Process the file line by line to handle large datasets efficiently.
        for line in f_in:
            line_json = json.loads(line)
            user_query = line_json.get("user_query", "")
            # The 'routing_decision' field from the raw data corresponds to the 'intent'.
            intent = line_json.get("routing_decision", "")

            # Create a new, smaller JSON object with only the relevant fields.
            intent_entry = {
                "user_query": user_query,
                "intent": intent
            }
            f_out.write(json.dumps(intent_entry) + '\n')

def extract_metadata(input_file: str, output_file: str = "metadata_dataset.jsonl"):
    """
    Creates a dataset for the metadata extraction task.

    This function extracts the `user_query` and the `metadata_extraction` fields.
    The goal is to train a model that can parse a natural language query and convert
    it into a structured set of parameters (e.g., ticker, year, document type) that
    can be used to query a vector database or other data sources.

    @param input_file: Path to the input .jsonl file containing raw generated data.
    @param output_file: Path to the output .jsonl file for the metadata dataset.
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line_json = json.loads(line)
            query = line_json.get("user_query", "")
            # The 'metadata_extraction' field contains the structured output we want the model to learn to generate.
            metadata = line_json.get("metadata_extraction", "")
            
            metadata_query = {
                "user_query": query,
                "metadata": metadata
            }
            f_out.write(json.dumps(metadata_query) + '\n')

def extract_final_answer(input_file: str, output_file: str = "final_answer_dataset.jsonl"):
    """
    Creates a dataset for the final answer generation task.

    This function extracts the `user_query`, the `final_response`, and the `metadata_extraction`
    fields. This dataset is used to train a model that takes the user's question and the
    retrieved context (represented by the metadata) to generate a final, synthesized answer.

    @param input_file: Path to the input .jsonl file containing raw generated data.
    @param output_file: Path to the output .jsonl file for the final answer dataset.
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line_json = json.loads(line)
            query = line_json.get("user_query", "")
            final_answer = line_json.get("final_response", "")
            # Including the metadata provides the model with the "context" it should use to form the answer.
            metadata = line_json.get("metadata_extraction", "")
            
            final_answer_entry = {
                "user_query": query,
                "final_answer": final_answer,
                "metadata": metadata
            }
            f_out.write(json.dumps(final_answer_entry) + '\n')