import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_embedding(text):
    """
    Get embedding vector for input text using OpenAI's embedding model

    Args:
        text (str): Input text to get embedding for

    Returns:
        list: Embedding vector from OpenAI model
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Get embedding from OpenAI
        response = client.embeddings.create(model="text-embedding-ada-002", input=text)

        # Return the embedding vector
        return response.data[0].embedding

    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        return None


if __name__ == "__main__":
    sample_text = "This is sample text"
    embedding = get_embedding(sample_text)
    print(embedding)
    if embedding:
        print(f"Generated embedding vector of length {len(embedding)} for sample text")
    else:
        print("Failed to generate embedding")
