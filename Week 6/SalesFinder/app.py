from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import SalesAgent
import json

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class Profile(BaseModel):
    text: str


@app.post("/generate-email")
async def generate_email(profile: Profile):
    try:
        # Create SalesAgent instance with profile text directly
        agent = SalesAgent(profile.text)

        # Generate email
        email = agent.generate_email()

        # Return raw email response without JSON parsing
        return email

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
