from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

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
csv_data = pd.read_csv("Miami-Dade_tax_assessor_cleaned.csv", low_memory=False)

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