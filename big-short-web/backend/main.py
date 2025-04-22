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
csv_data = pd.read_csv("Miami-Dade_tax_assessor_cleaned.csv", low_memory=False)
model = joblib.load("Miami-Dade_GradientBoosting_final_model_pipeline.pkl")

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

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PropertyAge"] = 2023 - df["YearBuilt"]
    df["TotalBathCount"] = df["BathCount"] + 0.5 * df["BathPartialCount"].fillna(0)
    df["AreaPerBedroom"] = df["AreaBuilding"] / df["BedroomsCount"].replace(0, 1)
    df["PricePerSqFt"] = df["AssessorLastSaleAmount"] / df["AreaBuilding"].replace(0, np.nan)
    df["AmenityScore"] = 0  # Simplified — or reimplement your full scoring logic
    df["GeographicCluster"] = 0  # Simplified — or use KMeans
    df["PriceLocationCluster"] = 0  # Simplified — or use KMeans
    df["AreaBuilding_log"] = np.log1p(df.get("AreaBuilding", 0))
    df["AreaLotSF_log"] = np.log1p(df.get("AreaLotSF", 0))
    return df


@app.get("/predict-by-address")
def predict_by_address(address: str):
    row = csv_data[csv_data["PropertyAddressFull"] == address]

    if row.empty:
        return {"error": "Address not found"}

    row = row.loc[:, ~row.columns.duplicated()]
    row = engineer_features(row)

    try:
        prediction = model.predict(row)[0]

        important_keys = [
            "YearBuilt",
            "Pool",
            "AreaLotSF",
            "BathCount",
            "BedroomsCount",
            "StoriesCount",
        ]
        filtered_features = {
            key: row.iloc[0][key]
            for key in important_keys
            if key in row.columns
        }

        return {
            "predicted_value": round(float(prediction), 2),
            "features": {
                k: v.item() if isinstance(v, np.generic) else v
                for k, v in filtered_features.items()
            },
        }
    except Exception as e:
        print("🔥 Prediction error:", e)
        return {"error": "Prediction failed."}



@app.post("/predict")
def predict(request: PredictionRequest):
    input_data = request.dict()
    df = pd.DataFrame([input_data])
    df = df.loc[:, ~df.columns.duplicated()]

    for col in df.columns:
        if pd.isna(df.at[0, col]):
            df.at[0, col] = 0

    # ✅ Add this line:
    df = engineer_features(df)

    print("✅ Final model input:", df.to_dict(orient="records")[0])
    prediction = model.predict(df)[0]

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
        filtered = filter_valid_properties(csv_data)
        filtered = filtered[filtered["PropertyAddressZIP"] == zipcode_int]
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

import random

@app.get("/random-properties")
def get_random_properties(count: int = 3):
    filtered = filter_valid_properties(csv_data)
    sample = filtered.sample(n=min(count, len(filtered)))

    results = []
    for _, row in sample.iterrows():
        try:
            # Convert single row to DataFrame for processing
            row_df = pd.DataFrame([row])
            row_df = row_df.loc[:, ~row_df.columns.duplicated()]  # clean up

            # Apply feature engineering
            row_df = engineer_features(row_df)

            # Get model prediction
            predicted_value = model.predict(row_df)[0]

            results.append({
                "address": row["PropertyAddressFull"],
                "city": row["PropertyAddressCity"],
                "zip": row["PropertyAddressZIP"],
                "current": float(row["TaxMarketValueTotal"]),
                "predicted": round(float(predicted_value), 2),
            })
        except Exception as e:
            print(f"🔥 Error predicting row: {e}")
            continue

    return {"properties": results}

def filter_valid_properties(df):
    return df[
        (df["TaxMarketValueTotal"] > 10000) &
        (df["PropertyAddressFull"].notna()) &
        (df["PropertyAddressFull"].str.strip().str.upper().isin(["", "0", "UNKNOWN"]) == False) &
        (df["PropertyAddressCity"].notna()) &
        (df["PropertyAddressZIP"].notna())
    ]