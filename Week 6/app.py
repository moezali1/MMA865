from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Load the trained pipeline
with open("train_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)


class DiamondFeatures(BaseModel):
    carat_weight: float
    cut: str
    color: str
    clarity: str
    polish: str
    symmetry: str
    report: str


@app.post("/predict")
async def predict_price(features: DiamondFeatures):
    try:
        # Convert input to pandas DataFrame
        input_df = pd.DataFrame(
            {
                "Carat Weight": [features.carat_weight],
                "Cut": [features.cut],
                "Color": [features.color],
                "Clarity": [features.clarity],
                "Polish": [features.polish],
                "Symmetry": [features.symmetry],
                "Report": [features.report],
            }
        )

        # Make prediction
        prediction = pipeline.predict(input_df)

        return {"predicted_price": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
