# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


app = FastAPI(title="Penguin Species Prediction API")

penguins: Dict[int, dict] = {}
next_id = 1
model = None


class PenguinCreate(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    island: Optional[str] = None
    sex: Optional[str] = None
    species: Optional[str] = None


class PenguinUpdate(BaseModel):
    bill_length_mm: Optional[float] = None
    bill_depth_mm: Optional[float] = None
    flipper_length_mm: Optional[float] = None
    body_mass_g: Optional[float] = None
    island: Optional[str] = None
    sex: Optional[str] = None
    species: Optional[str] = None


class PredictionInput(BaseModel):
    bill_length_mm: float = Field(..., example=39.1)
    bill_depth_mm: float = Field(..., example=18.7)
    flipper_length_mm: float = Field(..., example=181)
    body_mass_g: float = Field(..., example=3750)


@app.on_event("startup")
def train_model():
    global model

    try:
        df = pd.read_csv("penguins_raw.csv")

        df = df.rename(columns={
            "Culmen Length (mm)": "bill_length_mm",
            "Culmen Depth (mm)": "bill_depth_mm",
            "Flipper Length (mm)": "flipper_length_mm",
            "Body Mass (g)": "body_mass_g",
            "Species": "species"
        })

        features = [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g"
        ]

        df = df.dropna(subset=["species"])

        X = df[features]
        y = df["species"]

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000))
        ])

        model.fit(X, y)

        print("Model trained successfully.")

    except Exception as e:
        print("Model could not be trained:", e)
        model = None


@app.get("/")
def home():
    return {
        "message": "Penguin Species Prediction API is running.",
        "endpoints": [
            "POST /penguins",
            "GET /penguins",
            "GET /penguins/{penguin_id}",
            "PUT /penguins/{penguin_id}",
            "DELETE /penguins/{penguin_id}",
            "POST /predict"
        ]
    }


@app.get("/health")
def health_check():
    return {
        "api_status": "running",
        "model_loaded": model is not None,
        "stored_penguins": len(penguins)
    }


@app.post("/penguins")
def create_penguin(penguin: PenguinCreate):
    global next_id

    penguin_id = next_id
    penguins[penguin_id] = penguin.model_dump()
    next_id += 1

    return {
        "message": "Penguin record created.",
        "penguin_id": penguin_id,
        "data": penguins[penguin_id]
    }


@app.get("/penguins")
def read_all_penguins():
    return penguins


@app.get("/penguins/{penguin_id}")
def read_penguin(penguin_id: int):
    if penguin_id not in penguins:
        raise HTTPException(status_code=404, detail="Penguin not found")

    return penguins[penguin_id]


@app.put("/penguins/{penguin_id}")
def update_penguin(penguin_id: int, updated_penguin: PenguinUpdate):
    if penguin_id not in penguins:
        raise HTTPException(status_code=404, detail="Penguin not found")

    update_data = updated_penguin.model_dump(exclude_unset=True)
    penguins[penguin_id].update(update_data)

    return {
        "message": "Penguin record updated.",
        "penguin_id": penguin_id,
        "data": penguins[penguin_id]
    }


@app.delete("/penguins/{penguin_id}")
def delete_penguin(penguin_id: int):
    if penguin_id not in penguins:
        raise HTTPException(status_code=404, detail="Penguin not found")

    deleted = penguins.pop(penguin_id)

    return {
        "message": "Penguin record deleted.",
        "deleted_data": deleted
    }


@app.post("/predict")
def predict_species(data: PredictionInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Make sure penguins_raw.csv is in the project folder."
        )

    input_df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    probability_output = {
        species: round(float(prob), 4)
        for species, prob in zip(classes, probabilities)
    }

    return {
        "predicted_species": prediction,
        "probabilities": probability_output,
        "input_data": data.model_dump()
    }