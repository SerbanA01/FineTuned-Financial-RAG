from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from .query_parser_utils.schemas import QueryFocus, DocType
from .query_parser_utils.rule_based_parser import extract_structured_metadata
from .query_parser_utils.llm_refiner import refine_with_llm

def format_provisional_output_for_prompt(focuses: List[QueryFocus]) -> str:
    """
    Formats the output of the rule-based parser into a string.

    This string is not used in the final logic but is intended to be injected
    into an LLM prompt as a "hint" or a starting point. It provides the LLM
    with a pre-processed, structured guess that it can then refine or correct.

    @param focuses: A list of QueryFocus objects from the rule-based parser.
    @return: A formatted string summarizing the provisional extraction.
    """
    if not focuses: return "Rule-based extraction found no provisional focuses."
    formatted_str = "Provisional Rule-Based Extraction (this is just a HINT, the Original Query is the source of truth):\n"
    for i, focus in enumerate(focuses):
        year_str = str(focus.year) if focus.year is not None else "Not specified"
        quarter_str = "Q" + str(focus.quarter) if focus.quarter is not None else "Not specified"
        doc_type_str = focus.doc_type if focus.doc_type is not None else "Not specified (no keyword)"
        formatted_str += f"- Focus {i+1}: Ticker={focus.ticker}, Year={year_str}, Raw Quarter={quarter_str}, Normalized DocType (from keywords)={doc_type_str}\n"
    return formatted_str.strip()


def apply_final_rules_to_focuses(provisional_focuses: List[QueryFocus]) -> List[QueryFocus]:
    """
    Applies strict, non-negotiable business logic to the parsed query focuses.

    This function acts as a final sanity check and enforcement layer after the LLM has
    produced its refined output. LLMs can sometimes make minor logical errors, and these
    hard-coded rules ensure the final `QueryFocus` objects are valid according to
    financial reporting standards.

    @param provisional_focuses: A list of QueryFocus objects, typically from the LLM refiner.
    @return: A new list of finalized QueryFocus objects with rules applied.
    """
    finalized_focuses = []
    for focus in provisional_focuses:
        final_focus = QueryFocus(focus.ticker, focus.year, focus.quarter, focus.doc_type)

        # Rule: A 10-K is an annual report and thus cannot be associated with a specific quarter.
        # If the LLM mistakenly assigns one, we remove it.
        if final_focus.doc_type == DocType.K10:
            final_focus.quarter = None
        # Rule: A 10-Q is a quarterly report for the first three quarters. The fourth quarter's
        # results are rolled into the annual 10-K.
        elif final_focus.doc_type == DocType.Q10:
            if final_focus.quarter not in [1, 2, 3]:
                final_focus.quarter = None
        finalized_focuses.append(final_focus)
    return finalized_focuses

def process_query_with_llm_refinement(
    query: str,
    metadata_model: AutoModelForCausalLM,
    metadata_tokenizer: AutoTokenizer,
    metadata_terminators: list
) -> Tuple[List[QueryFocus], str]:
    """
    Processes a user query using a hybrid rule-based and LLM-driven approach.

    This function orchestrates the query parsing process by combining the speed and
    predictability of a rule-based system with the contextual nuance and flexibility
    of a Large Language Model.

    The process is as follows:
    1.  A fast, rule-based parser performs an initial pass to extract entities. This
        serves as a baseline but may miss complex associations.
    2.  An LLM then re-evaluates the original query, using the rule-based output only
        as a hint. The LLM's primary instruction is to treat the user's query as the
        absolute source of truth, allowing it to correct errors or omissions from the
        first step.
    3.  A final set of hard-coded business rules are applied to the LLM's output to
        ensure the results are consistent and logical (e.g., a 10-K cannot have a quarter).

    @param query: The raw natural language query from the user.
    @param metadata_model: The fine-tuned LLM for entity extraction and refinement.
    @param metadata_tokenizer: The tokenizer corresponding to the metadata model.
    @param metadata_terminators: A list of token IDs to stop generation.
    @return: A tuple containing:
             - A list of finalized, structured QueryFocus objects.
             - A "modified" query string where the text of extracted entities has been removed.
    """
    # Step 1: Perform an initial, fast extraction using hand-crafted rules.
    # This provides a quick but potentially inaccurate first guess.
    provisional_focuses_rules, modified_query_by_rules = extract_structured_metadata(query)
    print(f"Provisional rule-based output (pre-LLM, pre-final-rules): {provisional_focuses_rules}")

    # Step 2: Use an LLM to refine the initial extraction. The LLM gets the original
    # query and is told to use it as the "source of truth", effectively overriding
    # the rule-based output if it detects a more nuanced interpretation.
    print(f"\n--- Calling LOCAL METADATA LLM for query: \"{query}\" ---")
    provisional_focuses_llm = refine_with_llm(
        original_query=query,
        provisional_rule_based_focuses=provisional_focuses_rules,
        model=metadata_model,
        tokenizer=metadata_tokenizer,
        terminators=metadata_terminators
    )
    print(f"Provisional LLM output (pre-final-rules): {provisional_focuses_llm}")

    # Step 3: Apply a final layer of non-negotiable business logic. This corrects
    # any small, predictable errors the LLM might make regarding financial reporting rules.
    final_focuses = apply_final_rules_to_focuses(provisional_focuses_llm)
    print(f"Final focuses after applying heuristic rules: {final_focuses}")

    # The modified query from the rule-based parser is returned because it's a simple,
    # deterministic string manipulation, which is sufficient for its downstream purpose.
    return final_focuses, modified_query_by_rules