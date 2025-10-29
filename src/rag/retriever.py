from qdrant_client import QdrantClient, models
from dateutil.parser import parse as date_parse
from collections import OrderedDict
from typing import List, Optional

from .query_parser_utils.schemas import QueryFocus, DocType


def get_quarter_from_date(date_str: Optional[str]) -> Optional[int]:
    """
    Infers the calendar quarter (1, 2, 3, or 4) from a date string.

    This is a crucial helper for post-filtering 10-Q documents, as their payloads
    contain a 'date' field but not an explicit 'quarter' field.

    @param date_str: A string representing a date (e.g., "2023-05-10").
    @return: The integer quarter (1-4) or None if the date string is invalid or missing.
    """
    if not date_str:
        return None
    try:
        # `dateutil.parser` is robust and can handle a wide variety of date formats
        # without needing to specify the format string explicitly.
        dt = date_parse(date_str)
        # Standard calculation to map a month to its calendar quarter.
        return (dt.month - 1) // 3 + 1
    except (ValueError, TypeError):
        # Gracefully handle cases where the date is None, not a string, or un-parseable.
        return None

# --- Step 1: Building the Qdrant Filter (Broad Search) ---

def build_single_focus_filter(focus: QueryFocus) -> models.Filter:
    """
    Builds a robust Qdrant filter for a single QueryFocus object.

    This filter is the "broad" part of a two-phase filtering strategy. It's designed
    to retrieve a superset of potentially relevant documents from the vector database,
    even with known inconsistencies in the payload data. More precise filtering,
    especially for 10-Q quarters, is handled in a later post-filtering step.

    @param focus: The QueryFocus object containing the criteria (ticker, year, etc.).
    @return: A Qdrant Filter object to be used in a search query.
    """
    must_conditions = []

    # Ticker is a reliable field, so we create a simple match condition.
    must_conditions.append(
        models.FieldCondition(key="ticker", match=models.MatchValue(value=focus.ticker))
    )

    # The 'year' field in the database can be either a string or an integer.
    # To handle this inconsistency, we use a 'should' clause which acts as an OR.
    # The document's year must match either the integer version OR the string version.
    if focus.year:
        must_conditions.append(
            models.Filter(
                should=[
                    models.FieldCondition(key="year", match=models.MatchValue(value=focus.year)),
                    models.FieldCondition(key="year", match=models.MatchValue(value=str(focus.year)))
                ]
            )
        )

    # Document type and quarter filtering is the most nuanced part due to data inconsistencies.
    if focus.doc_type == DocType.K10:
        # 10-K filings are reliably identified by these two payload fields.
        must_conditions.append(models.FieldCondition(key="source_type", match=models.MatchValue(value="sec_filing")))
        must_conditions.append(models.FieldCondition(key="filing_category", match=models.MatchValue(value="10k")))
        # A 10-K is an annual report, so we intentionally do not filter on quarter.

    elif focus.doc_type == DocType.EARNINGS_TRANSCRIPT:
        # Earnings transcripts are reliably identified by their source_type.
        must_conditions.append(models.FieldCondition(key="source_type", match=models.MatchValue(value="earnings_transcript")))
        # For transcripts, the 'quarter' field in the payload (e.g., "Q1") is reliable, so we can filter on it directly.
        if focus.quarter:
            must_conditions.append(
                models.FieldCondition(key="quarter", match=models.MatchValue(value=f"Q{focus.quarter}"))
            )

    elif focus.doc_type == DocType.Q10:
        # 10-Q filings are identified by these fields.
        must_conditions.append(models.FieldCondition(key="source_type", match=models.MatchValue(value="sec_filing")))
        must_conditions.append(models.FieldCondition(key="filing_category", match=models.MatchValue(value="10q")))
        # CRITICAL STRATEGY: We deliberately DO NOT filter by quarter at this stage.
        # The 10-Q payloads lack a reliable 'quarter' field but have a 'date' field.
        # So, we retrieve all 10-Qs for the given year and then use the 'date' field
        # for precise quarter filtering in the post-processing step.

    return models.Filter(must=must_conditions)


def search_qdrant_per_focus(
    query_vector,
    focuses: List[QueryFocus],
    client: QdrantClient,
    collection_name: str,
    k: int = 10
) -> OrderedDict[QueryFocus, list[models.ScoredPoint]]:
    """
    Runs one vector similarity search against Qdrant for each QueryFocus.

    This function orchestrates the initial, broad retrieval. It iterates through each
    parsed user request (focus), builds the appropriate filter for it, and executes
    the search. The results are returned in an ordered dictionary, keyed by the
    original focus that produced them.

    @param query_vector: The embedding of the user's query.
    @param focuses: A list of QueryFocus objects representing the user's requests.
    @param client: The Qdrant client instance.
    @param collection_name: The name of the collection to search.
    @param k: The number of initial candidates to retrieve.
    @return: An OrderedDict mapping each focus to its list of retrieved ScoredPoints.
    """
    vec = query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector
    initial_results = OrderedDict()

    for focus in focuses:
        query_filter = build_single_focus_filter(focus)

        try:
            res = client.search(
                collection_name=collection_name,
                query_vector=vec,
                limit=k,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                timeout=10
            )
            initial_results[focus] = res
        except Exception as e:
            # Ensure that a failure for one focus doesn't crash the entire process.
            print(f"Error searching Qdrant for focus {focus}: {e}")
            initial_results[focus] = []

    return initial_results


# --- Step 2: Applying the Post-Filter (Precise Filtering) ---

def post_filter_results(
    initial_results_dict: OrderedDict[QueryFocus, list[models.ScoredPoint]]
) -> OrderedDict[QueryFocus, list[models.ScoredPoint]]:
    """
    Applies precise, in-memory filtering rules after the initial Qdrant search.

    This is the "precise" part of the two-phase filtering strategy. Its main purpose
    is to handle the case of 10-Q documents where the quarter can only be reliably
    determined by parsing the 'date' field from the payload. This logic is too
    complex for a direct Qdrant filter, so it's applied here in Python.

    @param initial_results_dict: The dictionary of results from `search_qdrant_per_focus`.
    @return: A new dictionary with the results precisely filtered.
    """
    final_results_dict = OrderedDict()

    for focus, points in initial_results_dict.items():
        # The only case that requires post-filtering is a 10-Q with a specific quarter.
        # All other document types were either fully filtered in Qdrant or don't need this logic.
        if not (focus.doc_type == DocType.Q10 and focus.quarter is not None):
            final_results_dict[focus] = points # Pass through without changes.
            continue

        # This is the core logic: iterate through the retrieved 10-Q points and keep
        # only those where the date corresponds to the requested quarter.
        filtered_points = []
        for point in points:
            # The payload 'date' field is the source of truth for the quarter.
            payload_date = point.payload.get("date")
            inferred_quarter = get_quarter_from_date(payload_date)

            if inferred_quarter == focus.quarter:
                filtered_points.append(point)

        final_results_dict[focus] = filtered_points

    return final_results_dict