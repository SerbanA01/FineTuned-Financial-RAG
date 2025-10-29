from typing import List
from sentence_transformers import CrossEncoder
from .process import SearchResult


def _get_tier_priority(tier_string: str) -> int:
    """
    Assigns a numerical priority to a search result's tier string for sorting.

    The principle is that a lower number indicates a higher priority. This allows us
    to enforce a strict order of preference in the final results, ensuring that
    results from more precise search methods are always ranked higher than those
    from more speculative, broader searches.

    @param tier_string: The descriptive tier, e.g., "Exact Match".
    @return: An integer representing the priority (0 is best).
    """
    if not tier_string:
        return 99 # Lowest priority for untiered results.

    # 0: Highest priority. These results perfectly matched the user's specific request.
    if tier_string == "Exact Match":
        return 0
    # 1: Second priority. These results were found by slightly relaxing the initial criteria.
    if tier_string.startswith("Relaxed:"):
        return 1
    # 2: Third priority. These were added to pad out the context and may be less relevant.
    if tier_string.startswith("Augmented:"):
        return 2

    return 99 # Default to lowest priority.


def finalize_and_rerank_results(
    query: str,
    candidate_results: List["SearchResult"],
    reranker: "CrossEncoder",
    final_k: int
) -> List["SearchResult"]:
    """
    Refines and ranks a list of candidate results using a powerful CrossEncoder model.

    This function is a critical step for improving the quality of the final context
    provided to the LLM. It takes a potentially large and noisy list of retrieved
    documents and applies a two-stage sorting process:

    1.  **Tier-Based Priority:** It first sorts results based on how they were found.
        "Exact Match" results are always preferred over "Relaxed" or "Augmented" ones,
        regardless of their semantic score.
    2.  **Semantic Reranking:** Within each tier, it uses a CrossEncoder model to
        calculate a much more accurate relevance score between the query and each
        document's text.

    The final output is a single, sorted list of the top `final_k` results, ordered
    by this combined logic.

    @param query: The original user query string.
    @param candidate_results: The list of SearchResult objects retrieved from the DB.
    @param reranker: An initialized CrossEncoder model.
    @param final_k: The desired number of results in the final list.
    @return: A sorted and truncated list of the best SearchResult objects.
    """

    if not candidate_results:
        print("  -> Reranker received no candidates. Returning empty list.")
        return []

    print(f"  -> Reranking {len(candidate_results)} candidates to select top {final_k}...")

    # 1. Prepare the input for the CrossEncoder model. It requires pairs of
    #    [query, document_text] to perform its detailed comparison.
    sentence_pairs = [[query, res.point.payload.get('chunk_text', '')] for res in candidate_results]

    # 2. Predict new, more accurate relevance scores. This is a computationally
    #    intensive step compared to the initial vector search, but yields higher quality.
    try:
        rerank_scores = reranker.predict(sentence_pairs, show_progress_bar=False)
    except Exception as e:
        # As a fallback, if the reranker model fails, return the original results
        # without reranking to prevent the entire pipeline from crashing.
        print(f"  -> ERROR during reranking: {e}. Returning original top-k without reranking.")
        return candidate_results[:final_k]

    # 3. Attach the new score and the tier priority to each result object.
    #    This enriches the objects with the necessary data for our multi-level sort.
    for res, score in zip(candidate_results, rerank_scores):
        res.rerank_score = score
        res.tier_priority = _get_tier_priority(res.tier)

    # 4. Perform the crucial multi-level sort. The `key` lambda function is the core
    #    of this strategy. Python's `sorted` will use the first element of the tuple
    #    (`x.tier_priority`) as the primary sort key. It will only use the second
    #    element (`-x.rerank_score`) to break ties between items with the same tier priority.
    sorted_results = sorted(
        candidate_results,
        key=lambda x: (x.tier_priority, -x.rerank_score)
    )

    # 5. Truncate the perfectly sorted list to the desired final size.
    final_top_k = sorted_results[:final_k]
    print(f"  -> Reranking complete. Final list has {len(final_top_k)} items.")

    return final_top_k