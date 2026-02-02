def precision_at_k(retrieved_ids, relevant_ids, k):
    """
    Compute Precision@K.

    Parameters:
    - retrieved_ids: list of retrieved document IDs (in ranked order)
    - relevant_ids: set of relevant document IDs for the query
    - k: int, the cutoff rank

    Returns:
    - float, precision@k score
    """
    retrieved_at_k = set(retrieved_ids[:k])
    relevant_retrieved = retrieved_at_k & set(relevant_ids)
    return len(relevant_retrieved) / k
