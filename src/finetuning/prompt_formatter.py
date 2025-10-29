# src/finetuning/prompt_formatter.py

import json

# This module provides functions to format raw data examples into the specific
# string-based prompt structures required for supervised fine-tuning (SFT).
# Each function creates a template that clearly delineates the instruction,
# the input (e.g., a user query), and the desired output for the model to learn.
# Consistent formatting is critical for successful fine-tuning.

def format_intent_training_example(example: dict) -> str:
    """
    Formats a data record for the intent classification fine-tuning task.

    This function creates a simple, clear prompt that asks the model to classify
    a user's query. The model learns to associate the text of the `user_query`
    with the corresponding `intent` label.

    @param example: A dictionary containing 'user_query' and 'intent' keys.
    @return: A formatted string ready for the SFTTrainer.
    """
    user_query = example.get('user_query', '')
    intent = example.get('intent', '')

    # The use of "###" is a common convention to structure prompts, helping the
    # model distinguish between different parts of the input.
    formatted_string = f"""### Instruction:
Classify the following user query.

### User Query:
{user_query}

### Classification:
{intent}"""
    return formatted_string

def format_metadata_extraction_example(example: dict) -> str:
    """
    Formats a data record for the metadata extraction fine-tuning task.

    This prompt instructs the model to act as a structured data extractor. It learns
    to take a natural language `user_query` and convert it into a machine-readable
    JSON object containing the specific parameters (like ticker, year, etc.)
    mentioned in the query.

    @param example: A dictionary with 'user_query' and 'metadata' keys.
    @return: A formatted string ready for the SFTTrainer.
    """
    user_query = example.get('user_query', '')
    metadata = example.get('metadata', {})

    # The target output for the model is a JSON string. We serialize the
    # ground-truth dictionary here to create the training example.
    metadata_string = json.dumps(metadata)

    formatted_string = f"""### Instruction:
Extract the key financial entities and document specifications from the user query into a structured JSON format.

### User Query:
{user_query}

### Structured Metadata:
{metadata_string}"""
    return formatted_string


def format_final_answer_example(example: dict) -> str:
    """
    Formats a data record for the final answer generation fine-tuning task.

    This prompt structure is designed to teach the model how to generate a final,
    synthesized answer based on a user's question. In a full RAG implementation,
    this prompt would typically also include the retrieved context that the model
    should use to formulate the answer. For this simplified version, it maps a
    direct question to its ideal answer.

    @param example: A dictionary with 'user_query' and 'final_answer' keys.
    @return: A formatted string ready for the SFTTrainer.
    """
    user_query = example.get('user_query', '')
    final_answer = example.get('final_answer', '')
    
    # This format assumes a direct question-to-answer mapping.
    # A more advanced version for a RAG system would inject the retrieved context
    # between the query and the answer, instructing the model to synthesize an
    # answer based on the provided context.
    formatted_string = f"""### User Query:
    {user_query}
    
    ### Final Answer:
    {final_answer}"""
    return formatted_string