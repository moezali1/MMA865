import numpy as np


def cosine_similarity(vector1, vector2):
    """
    Calculate cosine similarity between two vectors.

    Args:
        vector1 (list): First vector
        vector2 (list): Second vector

    Returns:
        float: Cosine similarity score between 0 and 1
    """
    # Convert lists to numpy arrays
    v1 = np.array(vector1)
    v2 = np.array(vector2)

    # Calculate dot product
    dot_product = np.dot(v1, v2)

    # Calculate magnitudes
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)

    # Calculate cosine similarity
    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    similarity = dot_product / (magnitude1 * magnitude2)
    return similarity


if __name__ == "__main__":
    # Test vectors
    vec1 = [1, 2, 3]
    vec2 = [4, 5, 6]

    similarity = cosine_similarity(vec1, vec2)
    print(f"Cosine similarity between test vectors: {similarity:.4f}")
