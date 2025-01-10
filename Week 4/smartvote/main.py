from retriever import retrieve_similar_documents
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class Party:
    def __init__(self, query):
        """
        Initialize Party with a search query.

        Args:
            query (str): The search query text
        """
        self.query = query

    def retrieve(self, top_n=25):
        """
        Retrieve similar documents for the query.

        Args:
            top_n (int): Number of results to return (default=5)

        Returns:
            list: List of documents sorted by similarity score
        """
        self.similar_documents = retrieve_similar_documents(self.query, top_n)
        return self.similar_documents

    def generate_analysis(self):

        system_prompt = """You are a helpful assistant explaining political party policies to voters. Your task is to take voter questions about specific issues and explain, in simple and clear language, what the political party's position and plans are on that topic.

        For each question, you will receive relevant extracts from the party's policy documents. Your role is to:
        1. Understand the voter's concern
        2. Review the party's position from the provided extracts
        3. Explain it in straightforward, everyday language that any voter can understand
        4. Focus on the practical impact - what the party plans to do and how it might affect voters

        Return your explanation as a JSON object with this structure:
        {'response': 'Your clear and simple yet very detailed explanation of the party's position and plans on the given issue'}

        If the provided extracts don't contain relevant information about the voter's concern, respond with:
        {'response': 'I cannot find this party's position on this specific issue in the provided information.'}
        
        Remember: Voters will use your explanation to help make their voting decisions, so be accurate, unbiased, and as clear as possible.
        """

        try:
            client = OpenAI()
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"User query: {self.query}\nRelevant extracts: {self.similar_documents}",
                    },
                ],
                response_format={"type": "json_object"},
                seed=33,
            )

            # Extract and parse the JSON response
            analysis = json.loads(completion.choices[0].message.content)
            return analysis

        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    liberal = Party("what are your plans for growing population crisis")
    print(liberal.retrieve())
    print(liberal.generate_analysis())
