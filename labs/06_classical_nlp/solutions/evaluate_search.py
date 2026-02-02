# Sample 100 unique queries for evaluation
sample_queries = list(set(queries))[:100]

# Evaluate the search system
k = 10
precision_scores = []
mrr_results = []

for query in tqdm(sample_queries, desc="Evaluating"):
    # Get relevant documents for this query
    relevant_ids = set(doc_ids[i] for i, q in enumerate(queries) if q == query)

    # Run search
    results = search(query, top_k=k)
    retrieved_ids = [doc_id for doc_id, score in results]

    # Compute metrics
    p_at_k = precision_at_k(retrieved_ids, relevant_ids, k)
    precision_scores.append(p_at_k)
    mrr_results.append((retrieved_ids, relevant_ids))

# Compute averages
avg_precision = sum(precision_scores) / len(precision_scores)
mrr = mean_reciprocal_rank(mrr_results)

print(f"Average Precision@{k}: {avg_precision:.4f}")
print(f"Mean Reciprocal Rank: {mrr:.4f}")
