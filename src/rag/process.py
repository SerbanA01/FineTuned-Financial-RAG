import copy
from collections import OrderedDict
from typing import List, Tuple, Dict
from qdrant_client import QdrantClient, models

from .query_parser import process_query_with_llm_refinement
from .query_parser_utils.schemas import QueryFocus
from .retriever import search_qdrant_per_focus, post_filter_results


class SearchResult:
    """
    A container for search results that wraps a Qdrant ScoredPoint and adds metadata.

    This class allows us to attach additional context to each search result, such as
    the 'tier', which explains how the result was obtained (e.g., as an exact match
    or through a relaxed, augmented search). This is crucial for debugging and for
    potentially weighting results differently downstream.
    """
    def __init__(self, point, tier: str):
        self.point = point
        self.tier = tier # e.g., "Exact Match", "Augmented: Other Years"

    def __repr__(self):
        return f"SearchResult(Tier: '{self.tier}', Score: {self.point.score:.4f})"

def search_and_filter(
    query_vector,
    focus: "QueryFocus",
    client: "QdrantClient",
    collection_name: str,
    k: int
) -> List["models.ScoredPoint"]:
    """
    A helper function to run a single search-and-filter operation against Qdrant.

    Encapsulates the two-step process of performing a vector search based on a query
    and a set of filters (derived from the `focus`), and then applying a secondary
    filtering stage to the results.

    @param query_vector: The embedding of the user's query.
    @param focus: The QueryFocus object defining the filters for this search.
    @param client: The Qdrant client instance.
    @param collection_name: The name of the collection to search within.
    @param k: The number of results to retrieve.
    @return: A list of ScoredPoint objects that match the search criteria.
    """
    initial_search_res = search_qdrant_per_focus(
        query_vector=query_vector,
        focuses=[focus],
        client=client,
        collection_name=collection_name,
        k=k,
    )
    final_filtered_res = post_filter_results(initial_search_res)
    return final_filtered_res.get(focus, [])


def process_and_retrieve_with_augmentation(
    query: str,
    client: "QdrantClient",
    collection_name: str,
    query_model,
    metadata_model,
    metadata_tokenizer,
    metadata_terminators,
    min_results_k: int = 5,
) -> Tuple[Dict["QueryFocus", List[SearchResult]], str]:
    """
    Executes a multi-phase retrieval strategy to ensure robust and comprehensive results.

    This is the core retrieval pipeline. It first parses the user query into structured
    `QueryFocus` objects. For each focus, it employs two main strategies:
    1.  **Full Relaxation:** If an exact search yields no results, the search criteria are
        progressively "relaxed" (e.g., by dropping the quarter, then the year) until a
        baseline set of documents is found. This prioritizes finding *something* relevant
        over finding nothing at all.
    2.  **Result Augmentation:** If the initial search (even after relaxation) returns
        fewer results than a desired minimum (`min_results_k`), the pipeline will
        "augment" the results by performing broader searches to pad the context with
        additional, related information.

    @param query: The raw user query string.
    @param client: The Qdrant client instance.
    @param collection_name: The name of the collection to search.
    @param query_model: The sentence-transformer model for embedding the query.
    @param metadata_model: The LLM used for parsing query metadata.
    @param metadata_tokenizer: The tokenizer for the metadata model.
    @param metadata_terminators: Terminator tokens for the metadata model.
    @param min_results_k: The desired minimum number of results per query focus.
    @return: A tuple containing:
             - An OrderedDict mapping each original `QueryFocus` to its list of `SearchResult` objects.
             - The modified query string after entity removal.
    """
    # First, parse the natural language query into structured `QueryFocus` objects.
    original_focuses, modified_query = process_query_with_llm_refinement(
        query,
        metadata_model=metadata_model,
        metadata_tokenizer=metadata_tokenizer,
        metadata_terminators=metadata_terminators
    )
    
    if not original_focuses:
        return OrderedDict(), modified_query

    print("\n--- Original Parsed Focuses ---")
    for f in original_focuses: print(f)

    # The modified query (with entities removed) is used for embedding to focus on the user's core intent.
    query_vector = query_model.encode(modified_query, normalize_embeddings=True)
    final_results_for_query = OrderedDict()

    # Process each distinct request from the user's query individually.
    for original_focus in original_focuses:
        print(f"\n--- Processing Focus with Augmentation: {original_focus} ---")
        
        # --- PHASE 1: Find a base set of results using full relaxation ---
        # The goal here is to find the best possible set of documents, even if it's not an exact match.
        base_results = []
        best_focus_found = None
        
        # Define the order of search attempts, from most specific to most general.
        relaxation_sequence = []
        relaxation_sequence.append((original_focus, "Exact Match"))
        if original_focus.quarter:
            focus_no_q = copy.deepcopy(original_focus); focus_no_q.quarter = None
            relaxation_sequence.append((focus_no_q, f"Relaxed: All quarters for {original_focus.year}"))
        if original_focus.year:
            focus_no_y = copy.deepcopy(original_focus); focus_no_y.year = None; focus_no_y.quarter = None
            relaxation_sequence.append((focus_no_y, "Relaxed: Most relevant year"))
        if original_focus.doc_type:
             focus_no_dt = copy.deepcopy(original_focus); focus_no_dt.doc_type = None; focus_no_dt.year = None; focus_no_dt.quarter = None
             relaxation_sequence.append((focus_no_dt, "Relaxed: Any document type"))

        print("  -> Phase 1: Finding best possible results via relaxation...")
        for focus_to_try, message in relaxation_sequence:
            print(f"    - Attempting search with: {focus_to_try}")
            points = search_and_filter(query_vector, focus_to_try, client, collection_name, k=min_results_k)
            if points:
                print(f"    - SUCCESS: Found {len(points)} results for '{message}'. This is our base.")
                base_results = [SearchResult(p, tier=message) for p in points]
                best_focus_found = focus_to_try
                break # Stop relaxing as soon as we find any results.
        
        if not base_results:
            print("  -> Phase 1 FAILED: No results found even after full relaxation.")
            final_results_for_query[original_focus] = []
            continue

        # --- PHASE 2: Augment results if we have fewer than the desired minimum ---
        # This ensures the LLM has enough context to form a good answer.
        if len(base_results) < min_results_k:
            print(f"  -> Phase 2: Augmenting results (found {len(base_results)} of {min_results_k})...")
            
            # Keep track of existing result IDs to avoid adding duplicates.
            existing_ids = {res.point.id for res in base_results}
            
            # Define a sequence of augmentation searches, similar to the relaxation sequence.
            augmentation_sequence = []
            if best_focus_found and best_focus_found.quarter:
                focus_aug = copy.deepcopy(best_focus_found); focus_aug.quarter = None
                augmentation_sequence.append((focus_aug, f"Augmented: Other quarters from {best_focus_found.year}"))
            if best_focus_found and best_focus_found.year:
                focus_aug = copy.deepcopy(best_focus_found); focus_aug.year = None; focus_aug.quarter = None
                augmentation_sequence.append((focus_aug, "Augmented: Other relevant years"))
            if best_focus_found and best_focus_found.doc_type:
                 focus_aug = copy.deepcopy(best_focus_found); focus_aug.doc_type = None
                 augmentation_sequence.append((focus_aug, "Augmented: Other document types"))

            for focus_to_try, message in augmentation_sequence:
                if len(base_results) >= min_results_k: break

                needed = min_results_k - len(base_results)
                print(f"    - Attempting to find {needed} more results with: {focus_to_try}")
                
                # Fetch more than needed initially to account for filtering out duplicates.
                points = search_and_filter(query_vector, focus_to_try, client, collection_name, k=min_results_k * 2)
                
                new_points_added = 0
                for p in points:
                    if len(base_results) >= min_results_k: break
                    if p.id not in existing_ids:
                        base_results.append(SearchResult(p, tier=message))
                        existing_ids.add(p.id)
                        new_points_added += 1
                
                if new_points_added > 0:
                    print(f"    - SUCCESS: Added {new_points_added} new results.")

        final_results_for_query[original_focus] = base_results

    # This final printout is for logging/debugging to clearly show what was retrieved for each part of the query.
    print("\n\n--- FINAL TIERED RESULTS ---")
    for focus, results in final_results_for_query.items():
        print(f"\nResults for Original Request: {focus}")
        if results:
            for res in results:
                payload = res.point.payload
                print(f"  - Tier: '{res.tier}' | "
                      f"{payload.get('filing_category') or payload.get('source_type')} "
                      f"({payload.get('year')}, Date: {payload.get('date')}) | "
                      f"Score: {res.point.score:.4f}")
        else:
            print("  No relevant documents could be found.")
            
    return final_results_for_query, modified_query