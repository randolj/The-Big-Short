import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load datasets
property_df = pd.read_csv("Miami-Dade_recorder_cleaned.csv")
transaction_df = pd.read_csv("Miami-Dade_tax_assessor_cleaned.csv")

# Inflation adjustment using CPI (rough estimates)
cpi_map = {
    1980: 82.4, 1981: 90.9, 1982: 96.5, 1983: 99.6, 1984: 103.9,
    1985: 107.6, 1986: 109.6, 1987: 113.6, 1988: 118.3, 1989: 124.0,
    1990: 130.7, 1991: 136.2, 1992: 140.3, 1993: 144.5, 1994: 148.2,
    1995: 152.4, 1996: 156.9, 1997: 160.5, 1998: 163.0, 1999: 166.6,
    2000: 172.2, 2001: 177.1, 2002: 179.9, 2003: 184.0, 2004: 188.9,
    2005: 195.3, 2006: 201.6, 2007: 207.3, 2008: 215.3, 2009: 214.5,
    2010: 218.056, 2011: 224.939, 2012: 229.594, 2013: 232.957, 2014: 236.736,
    2015: 237.017, 2016: 240.007, 2017: 245.120, 2018: 251.107, 2019: 255.657,
    2020: 258.811, 2021: 270.970, 2022: 292.655, 2023: 305.322
}
latest_cpi = max(cpi_map.values())

# Adjust TransferAmount for inflation
if "AssessorLastSaleDate" in property_df.columns:
    property_df["SaleYear"] = pd.to_datetime(property_df["AssessorLastSaleDate"], errors='coerce').dt.year
    property_df["CPI"] = property_df["SaleYear"].map(cpi_map)
    property_df["CPI"].fillna(latest_cpi, inplace=True)
    property_df["TransferAmount"] = property_df["TransferAmount"] * (latest_cpi / property_df["CPI"])

# Merge property and transaction data
df = property_df.merge(transaction_df, on="AttomID", how="inner")

# Enhanced Feature Engineering
if "YearBuilt" in df.columns:
    df["PropertyAge"] = 2023 - df["YearBuilt"]

if "BathCount" in df.columns and "BathPartialCount" in df.columns:
    df["TotalBathCount"] = df["BathCount"] + 0.5 * df["BathPartialCount"].fillna(0)

if "AreaBuilding" in df.columns and "BedroomsCount" in df.columns:
    df["AreaPerBedroom"] = df["AreaBuilding"] / df["BedroomsCount"].replace(0, 1)

# Calculate price per square foot
if "AreaBuilding" in df.columns and "TransferAmount" in df.columns:
    df["PricePerSqFt"] = df["TransferAmount"] / df["AreaBuilding"].replace(0, np.nan)

# Create an "amenities score"
amenity_cols = [col for col in df.columns if any(x in col.lower() for x in ["flag", "count"])]
numeric_amenity_score = 0
for col in amenity_cols:
    try:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_amenity_score += df[col].fillna(0)
        else:
            temp = df[col].map({'Y': 1, 'N': 0, 'Yes': 1, 'No': 0, 'TRUE': 1, 'FALSE': 0,
                               'True': 1, 'False': 0, 'true': 1, 'false': 0,
                               't': 1, 'f': 0})
            if temp.notna().all():
                numeric_amenity_score += temp
    except:
        pass

df["AmenityScore"] = numeric_amenity_score

# Add geographic clusters
if "PropertyLatitude" in df.columns and "PropertyLongitude" in df.columns:
    from sklearn.cluster import KMeans
    geo_df = df[["PropertyLatitude", "PropertyLongitude"]].copy()
    geo_df = geo_df.dropna()
    if len(geo_df) > 0:
        try:
            kmeans = KMeans(n_clusters=5, random_state=42)
            df.loc[geo_df.index, "GeographicCluster"] = kmeans.fit_predict(geo_df)
        except:
            df["GeographicCluster"] = 0

# Add location quality score using clustering of price data
if "PropertyLatitude" in df.columns and "PropertyLongitude" in df.columns and "TransferAmount" in df.columns:
    geo_price_df = df[["PropertyLatitude", "PropertyLongitude", "TransferAmount"]].copy()
    geo_price_df = geo_price_df.dropna()
   
    if len(geo_price_df) > 0:
        try:
            # Create weighted coordinates based on price
            price_mean = geo_price_df["TransferAmount"].mean()
            geo_price_df["weighted_lat"] = geo_price_df["PropertyLatitude"] * (geo_price_df["TransferAmount"] / price_mean)
            geo_price_df["weighted_lon"] = geo_price_df["PropertyLongitude"] * (geo_price_df["TransferAmount"] / price_mean)
           
            kmeans = KMeans(n_clusters=8, random_state=42)
            df.loc[geo_price_df.index, "PriceLocationCluster"] = kmeans.fit_predict(
                geo_price_df[["weighted_lat", "weighted_lon"]])
        except:
            df["PriceLocationCluster"] = 0

# Log transform skewed numerical features
skewed_features = ["AreaBuilding", "AreaLotSF", "TransferAmount"]
for feature in skewed_features:
    if feature in df.columns:
        df[f"{feature}_log"] = np.log1p(df[feature].fillna(0))

# Feature columns and target
features = ["TaxMarketValueTotal", "YearBuilt", "PropertyAge", "MinorCivilDivisionCode", "NeighborhoodCode",
    "CensusTract", "PropertyLatitude", "PropertyLongitude", "GeographicCluster", "PriceLocationCluster",
    "TaxAssessedValueTotal", "TaxRateArea", "TaxExemptionHomeownerFlag",
    "PropertyUseStandardized", "AssessorLastSaleAmount", "AreaBuilding", "AreaLotAcres", "AreaLotSF",
    "AreaPerBedroom", "TotalBathCount", "AmenityScore", "PricePerSqFt",
    "ParkingGarage", "ParkingGarageArea", "ParkingCarport", "ParkingCarportArea",
    "UtilitiesSewageUsage", "UtilitiesWaterSource", "Foundation", "Construction",
    "InteriorStructure", "PlumbingFixturesCount", "ConstructionFireResistanceClass", "SafetyFireSprinklersFlag",
    "FlooringMaterialPrimary", "BathCount", "BathPartialCount", "BedroomsCount", "RoomsCount", "StoriesCount",
    "UnitsCount", "RoomsBonusRoomFlag", "RoomsBreakfastNookFlag", "RoomsCellarFlag", "RoomsCellarWineFlag",
    "RoomsExerciseFlag", "RoomsFamilyCode", "RoomsGameFlag", "RoomsGreatFlag", "RoomsHobbyFlag", "RoomsLaundryFlag",
    "RoomsMediaFlag", "RoomsMudFlag", "RoomsOfficeArea", "RoomsOfficeFlag", "RoomsSafeRoomFlag", "RoomsSittingFlag",
    "RoomsStormShelter", "RoomsStudyFlag", "RoomsSunroomFlag", "RoomsUtilityArea", "RoomsUtilityCode", "Fireplace",
    "FireplaceCount", "AccessabilityElevatorFlag", "AccessabilityHandicapFlag", "EscalatorFlag", "CentralVacuumFlag",
    "RoofMaterial", "RoofConstruction", "ContentStormShutterFlag", "ContentOverheadDoorFlag", "ViewDescription",
    "PorchArea", "PatioArea", "DeckFlag", "DeckArea", "FeatureBalconyFlag", "BalconyArea", "BreezewayFlag",
    "ParkingRVParkingFlag", "ParkingSpaceCount", "DrivewayArea", "DrivewayMaterial", "Pool", "PoolArea",
    "ContentSaunaFlag", "TopographyCode", "FenceArea", "CourtyardFlag", "CourtyardArea", "ArborPergolaFlag",
    "GolfCourseGreenFlag", "TennisCourtFlag", "SportsCourtFlag", "ArenaFlag", "WaterFeatureFlag", "PondFlag",
    "BoatLiftFlag", "BuildingsCount", "BathHouseArea", "BathHouseFlag", "BoatAccessFlag", "BoatHouseArea",
    "BoatHouseFlag", "GazeboArea", "GazeboFlag",
    # Add log-transformed features
    "AreaBuilding_log", "AreaLotSF_log"
]
target = "TransferAmount"

# Keep only columns present in the dataset
available_features = [col for col in features if col in df.columns]

# Filter dataframe to only include available features and target
df = df[available_features + [target]]

# Handle outliers in the target variable
q1 = df[target].quantile(0.01)
q3 = df[target].quantile(0.99)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df_filtered = df[(df[target] >= lower_bound) & (df[target] <= upper_bound)]

# Drop rows where target is missing
df_filtered = df_filtered.dropna(subset=[target])

# Force low-cardinality and all other categorical columns to strings
for col in df_filtered.columns:
    if col in available_features:
        unique_vals = df_filtered[col].nunique(dropna=True)
        if df_filtered[col].dtype not in ['object', 'category', 'bool']:
            if unique_vals < 25:
                df_filtered[col] = df_filtered[col].astype(str)

# Re-identify categorical and numeric columns after coercion
categorical_cols = []
numeric_cols = []

for col in available_features:
    if df_filtered[col].dtype in ['object', 'category', 'bool']:
        categorical_cols.append(col)
    else:
        try:
            df_filtered[col] = df_filtered[col].astype(float)  # Ensure it's float or int
            numeric_cols.append(col)
        except:
            df_filtered[col] = df_filtered[col].astype(str)
            categorical_cols.append(col)

# Explicitly convert ALL categorical columns to string to avoid mixed types
for col in categorical_cols:
    df_filtered[col] = df_filtered[col].astype(str)

# Identify numeric and categorical columns
numeric_cols = df_filtered[available_features].select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df_filtered[available_features].select_dtypes(include=['object', 'category']).columns.tolist()

# Split features and target
X = df_filtered[available_features]
y = df_filtered[target]

# Create a train-test split to evaluate model performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='drop'
)

# Define base models with optimized parameters
rf_model = RandomForestRegressor(
    n_estimators=300,
    min_samples_split=5,
    min_samples_leaf=4,
    max_features='sqrt',
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=4,
    subsample=1.0,
    random_state=42
)

xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.005,
    max_depth=3,
    colsample_bytree=0.7,
    subsample=0.7,
    random_state=42,
    n_jobs=-1
)

nn_model = MLPRegressor(
    hidden_layer_sizes=(200, 100, 50, 25),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

# Define individual models
models = {
    'RandomForest': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', rf_model)
    ]),
    'GradientBoosting': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', gb_model)
    ]),
    'XGBoost': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb_model)
    ]),
    'NeuralNetwork': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', nn_model)
    ])
}

# Function to calculate and print metrics
def calculate_metrics(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
   
    print(f"\n{model_name} Metrics:")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE: ${mae:.2f}")
    print(f"R²: {r2:.4f}")
   
    return rmse, mae, r2

for col in categorical_cols:
    types = df_filtered[col].map(type).value_counts()
    if len(types) > 1:
        print(f"⚠️ Column '{col}' has mixed types: {types}")

# Train models and calculate metrics
results = {}
print("Training models with optimized parameters...")

for model_name, pipeline in models.items():
    print(f"\nTraining {model_name}...")
   
    # Train the model
    try:
        pipeline.fit(X_train, y_train)
       
        # Calculate metrics on test set
        y_pred = pipeline.predict(X_test)
        rmse, mae, r2 = calculate_metrics(y_test, y_pred, model_name)
       
        # Store results
        results[model_name] = {
            'pipeline': pipeline,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
       
        # Save the model
        joblib.dump(pipeline, f"Miami-Dade_{model_name}_model_pipeline.pkl")
        print(f"{model_name} model saved successfully!")
    except Exception as e:
        print(f"Error training {model_name}: {str(e)}")

# Find the best performing model based on R²
if results:
    best_model = max(results.items(), key=lambda x: x[1]['r2'])[0]
    print(f"\nBest model based on R²: {best_model} with R² = {results[best_model]['r2']:.4f}")

    # Train the final models on the full dataset
    print("\nTraining final models on full dataset...")
    for model_name, pipeline in models.items():
        if model_name in results:  # Only retrain models that completed successfully
            print(f"Training final {model_name} model...")
           
            try:
                # Train on full dataset
                pipeline.fit(X, y)
               
                # Save the final model
                joblib.dump(pipeline, f"Miami-Dade_{model_name}_final_model_pipeline.pkl")
                print(f"Final {model_name} model saved successfully!")
            except Exception as e:
                print(f"Error training final {model_name} model: {str(e)}")
else:
    print("No models completed training successfully.")

print("\nAll models trained and saved!")

# Save final artifacts for backend
print("\nSaving artifacts for backend...")

# Select the best performing model pipeline (already selected earlier)
final_pipeline = results[best_model]['pipeline']

# Fit on full dataset again (to ensure it's trained on everything)
final_pipeline.fit(X, y)

# Extract preprocessor and model
preprocessor_fitted = final_pipeline.named_steps['preprocessor']
model_fitted = final_pipeline.named_steps['model']

# Save the scaler (numeric part of preprocessor)
scaler = preprocessor_fitted.named_transformers_['num'].named_steps['scaler']
joblib.dump(scaler, "miami_scaler.pkl")

# Save final artifacts for backend
print("\n🔧 Saving artifacts for backend (TensorFlow format)...")

# Refit the best model pipeline on the full dataset
final_pipeline = results[best_model]['pipeline']
final_pipeline.fit(X, y)

# Extract preprocessor and model
preprocessor_fitted = final_pipeline.named_steps['preprocessor']
mlp_model = final_pipeline.named_steps['model']

# Save scaler from numeric pipeline
scaler = preprocessor_fitted.named_transformers_['num'].named_steps['scaler']
joblib.dump(scaler, "miami_scaler.pkl")

# Save column order
joblib.dump(X.columns.tolist(), "miami_model_columns.pkl")

# Convert sklearn MLP to TensorFlow model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# Rebuild the architecture
tf_model = Sequential()
# Get true input dimension after preprocessing
X_processed = preprocessor_fitted.transform(X)
input_dim = X_processed.shape[1]
layer_sizes = mlp_model.hidden_layer_sizes

for i, size in enumerate(layer_sizes):
    if i == 0:
        tf_model.add(Dense(size, input_dim=input_dim, activation='relu'))
    else:
        tf_model.add(Dense(size, activation='relu'))
tf_model.add(Dense(1))  # Output layer

# Set weights
weights = []
for coef, intercept in zip(mlp_model.coefs_, mlp_model.intercepts_):
    weights.append(coef)
    weights.append(intercept)
tf_model.set_weights(weights)

# Compile and save model
tf_model.compile(optimizer='adam', loss='mean_squared_error')
tf_model.save("miami_model.h5")

print("✅ Backend artifacts saved: miami_model.h5, miami_scaler.pkl, miami_model_columns.pkl")

