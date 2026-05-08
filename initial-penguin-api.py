from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# Input format
class Penguin(BaseModel):
    bill_length_mm: float
    flipper_length_mm: float


# Home page
@app.get("/")
def home():
    return {"message": "Penguin API is running!"}


# Prediction endpoint
@app.post("/predict")
def predict(penguin: Penguin):

    # SUPER basic fake prediction logic
    if penguin.flipper_length_mm > 200:
        species = "Gentoo"
    else:
        species = "Adelie"

    return {
        "predicted_species": species,
        "input_data": penguin
    }