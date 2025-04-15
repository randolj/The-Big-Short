from fastapi import FastAPI, Query
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend to access the backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to ["http://localhost:3000"] if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the CSV at startup
csv_data = pd.read_csv("Liberty_tax_assessor_cleaned.csv", low_memory=False)

# Load model, scaler, and columns
model = tf.keras.models.load_model("liberty_model.h5")
scaler = joblib.load("scaler.pkl")
column_order = joblib.load("model_columns.pkl")  # list of columns used during training

class PredictionRequest(BaseModel):
    size: float
    bedrooms: int
    bathrooms: int
    age: int
    stories: int
    basement: int
    hotWaterHeating: int
    airConditioning: int
    mainroad: int
    frontage: float
    depth: float
    backyardSize: float
    garage: int
    distanceFromOcean: float

# Input schema
class HouseFeatures(BaseModel):
    features: dict  # expects full feature dict

@app.get("/predict-by-address")
def predict_by_address(address: str):
    # Locate the property by address
    row = csv_data[csv_data["PropertyAddressFull"] == address]

    if row.empty:
        return {"error": "Address not found"}

    # Only use feature columns
    features_df = row[column_order] if set(column_order).issubset(row.columns) else row.reindex(columns=column_order, fill_value=0)

    # Ensure proper column order and fill missing columns
    features_df = features_df.fillna(0).reindex(columns=column_order, fill_value=0)

    # Scale
    scaled = scaler.transform(features_df)

    # Predict
    prediction = model.predict(scaled)[0][0]
    return {"predicted_value": round(float(prediction), 2)}

# TODO: Show what attributes contributed to that predicted value
# TODO: If the user doesn't give a value for predict, use average of all, or something along those lines

@app.post("/predict")
def predict(request: PredictionRequest):
    input_dict = request.dict()
    df = pd.DataFrame([input_dict])
    df = df.reindex(columns=column_order, fill_value=0)  # Match training data
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0][0]
    return {"predicted_value": round(float(prediction), 2)}

@app.get("/search-addresses")
def search_addresses(query: str = "", limit: int = 10):
    filtered = csv_data[csv_data["PropertyAddressFull"].str.contains(query, case=False, na=False)]
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
        # Convert zipcode to integer for comparison with integer values in dataset
        zipcode_int = int(zipcode)
        
        # Filter data by ZIP code (compare as integers)
        filtered = csv_data[csv_data["PropertyAddressZIP"] == zipcode_int]
        
        print(f"ZIP code search: {zipcode_int}")
        print(f"Found {len(filtered)} properties")
        
        if filtered.empty:
            return {"properties": []}
        
        # Prepare the results
        results = []
        for _, row in filtered.iterrows():
            try:
                current_value = float(row["TaxMarketValueTotal"]) if pd.notna(row["TaxMarketValueTotal"]) else 0
                predicted_value = current_value * 1.1  # Simple placeholder
                
                results.append({
                    "address": row["PropertyAddressFull"],
                    "current": current_value,
                    "predicted": predicted_value
                })
            except Exception as e:
                print(f"Error processing row: {e}")
        
        return {"properties": results}
    except ValueError:
        return {"properties": [], "error": "Invalid ZIP code format"}