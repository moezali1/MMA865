from database import connect_to_database
from embedding import get_embedding
from cosine import cosine_similarity
import heapq


def retrieve_similar_documents(input_query, top_n):
    """
    Retrieve the most similar documents to the input query using cosine similarity.

    Args:
        input_query (str): The search query text
        top_n (int): Number of most similar documents to return (default=5)

    Returns:
        list: List of top_n documents sorted by similarity score (highest first)
    """
    # Get database connection
    db = connect_to_database()
    liberal_collection = db["liberal"]

    # Get embedding for input query using get_embedding function
    query_embedding = get_embedding(input_query)
    if not query_embedding:
        return []

    # Use a min heap to efficiently track top_n results
    top_results = []

    # Process each document
    for doc in liberal_collection.find():
        doc_embedding = doc.get("embedding")
        if doc_embedding:
            # Calculate similarity
            similarity = cosine_similarity(query_embedding, doc_embedding)

            # Use negative similarity for min heap to get highest scores
            if len(top_results) < top_n:
                heapq.heappush(top_results, (similarity, doc))
            else:
                heapq.heappushpop(top_results, (similarity, doc))

    # Sort results by similarity score (highest first)
    results = sorted(top_results, key=lambda x: x[0], reverse=True)

    # Return list of documents with their similarity scores
    return [
        {"score": score, "page": doc["page"], "text": doc["text"]}
        for score, doc in results
    ]


if __name__ == "__main__":
    # Test query
    sample_query = "Housing crisis"

    # Get similar documents
    results = retrieve_similar_documents(sample_query, top_n=25)

    # Print results
    print(f"\nTop results for query: '{sample_query}'\n")
    for i, result in enumerate(results, 1):
        print(f"Result {i}")
        print(f"Page: {result['page']}")
        print(f"Similarity Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:200]}...")  # Show first 200 chars
        print()
