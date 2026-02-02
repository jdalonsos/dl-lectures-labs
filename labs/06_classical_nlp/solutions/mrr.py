def reciprocal_rank(retrieved_ids, relevant_ids):
    """
    Compute the reciprocal rank for a single query.

    Parameters:
    - retrieved_ids: list of retrieved document IDs (in ranked order)
    - relevant_ids: set of relevant document IDs

    Returns:
    - float, reciprocal rank (1/rank of first relevant doc, or 0 if none found)
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(results):
    """
    Compute Mean Reciprocal Rank over multiple queries.

    Parameters:
    - results: list of (retrieved_ids, relevant_ids) tuples

    Returns:
    - float, MRR score
    """
    rr_scores = [reciprocal_rank(ret, rel) for ret, rel in results]
    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0
