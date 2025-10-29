from transformers import AutoModelForCausalLM, AutoTokenizer

def classify_query_intent(
    query: str,
    intent_model: AutoModelForCausalLM,
    intent_tokenizer: AutoTokenizer
) -> str:
    """
    Classifies the user's query into a predefined category using a local LLM.

    This function sends the user's query to a fine-tuned language model
    specifically trained for this classification task. The goal is to determine
    the primary nature of the user's request, which helps route the query
    through the appropriate processing pipeline.

    The possible intents are:
    - DOCUMENT_SEARCH: The user is asking for information found within documents
      (e.g., "What did Apple say about AI in their 10-K?").
    - MARKET_DATA_ONLY: The user is asking for quantitative market data
      (e.g., "What is the stock price of TSLA?").
    - HYBRID: The query is a mix of both or is ambiguous.

    @param query: The raw user query string.
    @param intent_model: The loaded Hugging Face model for classification.
    @param intent_tokenizer: The tokenizer corresponding to the classification model.
    @return: A string representing the classified intent. Defaults to "HYBRID"
             in case of errors or unexpected model output.
    """
    try:
        possible_intents = ["HYBRID", "DOCUMENT_SEARCH", "MARKET_DATA_ONLY"]

        # This prompt is structured for a fine-tuned model that expects a simple instruction
        # and a placeholder for the user's text.
        prompt = f"""### Instruction:
Classify the user query into one of the following categories: DOCUMENT_SEARCH, HYBRID, MARKET_DATA_ONLY.

### User Query:
{query}

### Classification:
"""
        # Prepare the prompt according to the model's required chat template.
        messages = [
            {"role": "system", "content": "You are an expert at classifying user queries."},
            {"role": "user", "content": prompt.strip()},
        ]
        inputs = intent_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(intent_model.device)

        # We need the input length to later decode only the newly generated text.
        input_ids_length = inputs.shape[1]

        # Generate the classification. `max_new_tokens` is kept low as we expect a short, single-word answer.
        # `do_sample=False` ensures the most likely, deterministic output.
        outputs = intent_model.generate(
            inputs,
            max_new_tokens=10,
            eos_token_id=intent_tokenizer.eos_token_id,
            do_sample=False,
        )

        # Decode the generated tokens, skipping the prompt part.
        predicted_intent = intent_tokenizer.decode(outputs[0, input_ids_length:], skip_special_tokens=True).strip()

        # Check if the model's output contains one of the valid intents.
        # This is a robust way to parse the output, as the model might occasionally
        # add extra characters or formatting.
        for intent in possible_intents:
            if intent.lower() in predicted_intent.lower():
                return intent

        # If the model returns something completely unexpected, log it and default.
        print(f"Warning: Model returned an unexpected value: '{predicted_intent}'. Defaulting.")
        return "HYBRID"

    except Exception as e:
        # A broad exception handler ensures the system remains stable. If the classification
        # model fails for any reason, we default to the "HYBRID" intent, which is the
        # most flexible and safest fallback.
        print(f"Error during local intent classification: {e}. Defaulting to HYBRID.")
        return "HYBRID"