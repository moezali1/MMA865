from pypdf import PdfReader
import os

from database import connect_to_database
from embedding import get_embedding

db = connect_to_database()


def load_and_chunk_pdf(filename="Liberal.pdf"):
    # Construct path to PDF file in data directory
    # Since we're in Week 5/smartvote, we need to adjust the path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "data", filename)
    print(f"Attempting to read from: {pdf_path}")

    # Check if file exists before trying to read it
    if not os.path.exists(pdf_path):
        # Try alternate path in case data folder is at project root
        pdf_path = os.path.join(current_dir, "..", "data", filename)
        print(f"First path failed, trying: {pdf_path}")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(
                f"PDF file not found at either path. Please ensure {filename} exists in the data directory"
            )

    # Initialize PDF reader
    reader = PdfReader(pdf_path)

    # Create chunks where each chunk is one page
    chunks = []
    for page in reader.pages:
        text = page.extract_text()
        chunks.append(text)

    # Get embeddings and store chunks in MongoDB
    liberal_collection = db["liberal"]
    for i, chunk in enumerate(chunks):
        # Get embedding vector for the chunk
        embedding = get_embedding(chunk)

        # Create document with text, page number and embedding
        document = {"page": i + 1, "text": chunk, "embedding": embedding}
        liberal_collection.insert_one(document)

        # Print progress
        print(f"Processed and stored page {i + 1} of {len(chunks)}")

    return chunks


if __name__ == "__main__":
    try:
        chunks = load_and_chunk_pdf()
        print(f"Loaded and stored {len(chunks)} pages from PDF")
        print("Pages have been stored in the 'liberal' collection in MongoDB")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the PDF file exists in the 'data' directory")
