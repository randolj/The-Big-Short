from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
from typing import Optional

app = FastAPI()

# Allow frontend to access the backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data and model artifacts
csv_data = pd.read_csv("Liberty_tax_assessor_cleaned.csv", low_memory=False)
model = tf.keras.models.load_model("liberty_model.h5")
scaler = joblib.load("scaler.pkl")
column_order = joblib.load("model_columns.pkl")  # List of features used during training


class PredictionRequest(BaseModel):
    size: Optional[float] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    yearBuilt: Optional[int] = None
    stories: Optional[int] = None
    basement: Optional[int] = None
    hotWaterHeating: Optional[int] = None
    airConditioning: Optional[int] = None
    mainroad: Optional[int] = None
    frontage: Optional[float] = None
    depth: Optional[float] = None
    backyardSize: Optional[float] = None
    garage: Optional[int] = None
    distanceFromOcean: Optional[float] = None


def prepare_input(row: pd.DataFrame) -> np.ndarray:
    """Ensure correct column order, fill missing values, and scale input."""
    row = row.reindex(columns=column_order, fill_value=0).fillna(0)
    return scaler.transform(row)


@app.get("/predict-by-address")
def predict_by_address(address: str):
    row = csv_data[csv_data["PropertyAddressFull"] == address]

    if row.empty:
        return {"error": "Address not found"}

    features_df = (
        row[column_order]
        if set(column_order).issubset(row.columns)
        else row.reindex(columns=column_order, fill_value=0)
    )
    scaled = prepare_input(features_df)
    prediction = model.predict(scaled)[0][0]

    # Only include selected useful features
    important_keys = [
        "YearBuilt",
        "Pool",
        "AreaLotSF",
        "BathCount",
        "BedroomsCount",
        "StoriesCount",
    ]
    filtered_features = {
        key: features_df.iloc[0][key]
        for key in important_keys
        if key in features_df.columns
    }

    return {
        "predicted_value": round(float(prediction), 2),
        "features": filtered_features,
    }


# TODO: If the user doesn't give a value for predict, use average of all, or something along those lines (kinda working, just returns the same value)


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        input_data = request.dict()
        df = pd.DataFrame([input_data])
        df = df.reindex(columns=column_order)
        df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns

        for col in df.columns:
            val = df[col].iloc[0]
            if pd.isna(val):
                if col in csv_data.columns and pd.api.types.is_numeric_dtype(
                    csv_data[col]
                ):
                    df.at[0, col] = csv_data[col].dropna().mean()
                else:
                    df.at[0, col] = 0

        print("✅ Final model input:", df.to_dict(orient="records")[0])

        scaled = prepare_input(df)
        prediction = model.predict(scaled)[0][0]

        return {"predicted_value": round(float(prediction), 2)}
    except Exception as e:
        print("🔥 Prediction error:", e)
        return {"error": "Prediction failed."}


@app.get("/search-addresses")
def search_addresses(query: str = "", limit: int = 10):
    filtered = csv_data[
        csv_data["PropertyAddressFull"].str.contains(query, case=False, na=False)
    ]
    results = filtered[["PropertyAddressFull"]].drop_duplicates().head(limit)
    return results["PropertyAddressFull"].tolist()


@app.get("/address-details")
def get_address_details(address: str = Query(...)):
    row = csv_data[csv_data["PropertyAddressFull"] == address]
    if row.empty:
        return {"error": "Address not found"}
    return row.to_dict(orient="records")[0]


@app.get("/zip-properties")
def get_properties_by_zip(zipcode: str = Query(...)):
    try:
        zipcode_int = int(zipcode)
        filtered = csv_data[csv_data["PropertyAddressZIP"] == zipcode_int]

        if filtered.empty:
            return {"properties": []}

        results = []
        for _, row in filtered.iterrows():
            try:
                current_value = float(row.get("TaxMarketValueTotal", 0) or 0)
                predicted_value = current_value * 1.1  # Simple placeholder logic
                results.append(
                    {
                        "address": row["PropertyAddressFull"],
                        "current": current_value,
                        "predicted": round(predicted_value, 2),
                    }
                )
            except Exception:
                continue  # Skip bad rows

        return {"properties": results}
    except ValueError:
        return {"properties": [], "error": "Invalid ZIP code format"}
