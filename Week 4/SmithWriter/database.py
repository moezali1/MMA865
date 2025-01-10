from pymongo import MongoClient


def connect_to_database():
    """
    Establishes connection to MongoDB running on localhost
    Returns MongoDB database connection object
    """
    try:
        # Connect to MongoDB running on localhost:27017
        client = MongoClient("mongodb://localhost:27017/")

        # Get database object
        db = client["mmai891"]

        return db

    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return None
