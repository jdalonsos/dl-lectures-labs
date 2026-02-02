from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Build TF-IDF vectorizer and matrix
vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
tfidf_matrix = vectorizer.fit_transform(doc_texts)


def search(query, top_k=10):
    """
    Search for documents matching the query.

    Parameters:
    - query: str, the search query
    - top_k: int, number of top results to return

    Returns:
    - list of (doc_id, score) tuples, sorted by score descending
    """
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    return [(doc_ids[i], scores[i]) for i in top_indices]
