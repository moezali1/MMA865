from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import Party

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class Query(BaseModel):
    text: str


@app.post("/query")
async def query(query: Query):
    # Create Party instance with query text
    party = Party(query.text)

    # Get similar documents
    similar_docs = party.retrieve()

    # Generate analysis
    analysis = party.generate_analysis()

    return {"similar_documents": similar_docs, "analysis": analysis}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
