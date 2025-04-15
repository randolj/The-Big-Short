import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError

# Load datasets
property_df = pd.read_csv("Liberty_recorder_cleaned.csv")
transaction_df = pd.read_csv("Liberty_tax_assessor_cleaned.csv")

# Inflation adjustment using CPI (rough estimates)
cpi_map = {
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

# Feature columns and target
features = [
    "TaxMarketValueTotal", "YearBuilt", "AreaBuilding", "MinorCivilDivisionCode", "NeighborhoodCode", "CensusTract",
    "PropertyLatitude", "PropertyLongitude", "LegalSubdivision", 
    "TaxAssessedValueTotal", "TaxAssessedValueImprovements", "TaxAssessedValueLand", "TaxAssessedImprovementsPerc",
    "TaxRateArea", "TaxExemptionHomeownerFlag", "ZonedCodeLocal",
    "PropertyUseStandardized", "AssessorLastSaleAmount", "AreaBuilding", "AreaLotAcres", "AreaLotSF", "ParkingGarage",
    "ParkingGarageArea", "ParkingCarport", "ParkingCarportArea", "HVACCoolingDetail", "HVACHeatingDetail", "HVACHeatingFuel",
    "UtilitiesSewageUsage", "UtilitiesWaterSource", "UtilitiesMobileHomeHookupFlag", "Foundation", "Construction",
    "InteriorStructure", "PlumbingFixturesCount", "ConstructionFireResistanceClass", "SafetyFireSprinklersFlag",
    "FlooringMaterialPrimary", "BathCount", "BathPartialCount", "BedroomsCount", "RoomsCount", "StoriesCount",
    "UnitsCount", "RoomsBonusRoomFlag", "RoomsBreakfastNookFlag", "RoomsCellarFlag", "RoomsCellarWineFlag",
    "RoomsExerciseFlag", "RoomsFamilyCode", "RoomsGameFlag", "RoomsGreatFlag", "RoomsHobbyFlag", "RoomsLaundryFlag",
    "RoomsMediaFlag", "RoomsMudFlag", "RoomsOfficeArea", "RoomsOfficeFlag", "RoomsSafeRoomFlag", "RoomsSittingFlag",
    "RoomsStormShelter", "RoomsStudyFlag", "RoomsSunroomFlag", "RoomsUtilityArea", "RoomsUtilityCode", "Fireplace",
    "FireplaceCount", "AccessabilityElevatorFlag", "AccessabilityHandicapFlag", "EscalatorFlag", "CentralVacuumFlag",
    "ContentIntercomFlag", "ContentSoundSystemFlag", "WetBarFlag", "SecurityAlarmFlag", "StructureStyle", "Exterior1Code",
    "RoofMaterial", "RoofConstruction", "ContentStormShutterFlag", "ContentOverheadDoorFlag", "ViewDescription",
    "PorchCode", "PorchArea", "PatioArea", "DeckFlag", "DeckArea", "FeatureBalconyFlag", "BalconyArea", "BreezewayFlag",
    "ParkingRVParkingFlag", "ParkingSpaceCount", "DrivewayArea", "DrivewayMaterial", "Pool", "PoolArea",
    "ContentSaunaFlag", "TopographyCode", "FenceCode", "FenceArea", "CourtyardFlag", "CourtyardArea", "ArborPergolaFlag",
    "GolfCourseGreenFlag", "TennisCourtFlag", "SportsCourtFlag", "ArenaFlag", "WaterFeatureFlag", "PondFlag",
    "BoatLiftFlag", "BuildingsCount", "BathHouseArea", "BathHouseFlag", "BoatAccessFlag", "BoatHouseArea",
    "BoatHouseFlag", "CabinArea", "CabinFlag", "CanopyArea", "CanopyFlag", "GazeboArea", "GazeboFlag", "GraineryArea",
    "GraineryFlag", "GreenHouseArea", "GreenHouseFlag", "GuestHouseArea", "GuestHouseFlag", "KennelArea", "KennelFlag",
    "LeanToArea", "LeanToFlag", "LoadingPlatformArea", "LoadingPlatformFlag", "MilkHouseArea", "MilkHouseFlag",
    "OutdoorKitchenFireplaceFlag", "PoolHouseArea", "PoolHouseFlag", "PoultryHouseArea", "QuonsetArea", "QuonsetFlag",
    "ShedArea", "ShedCode", "SiloArea", "SiloFlag", "StableArea", "StableFlag", "StorageBuildingArea", "StorageBuildingFlag",
    "UtilityBuildingArea", "UtilityBuildingFlag", "PoleStructureArea", "PoleStructureFlag", "CommunityRecRoomFlag"
]

target = "TransferAmount"

# Drop missing values
df = df[features + [target]].dropna()

# One-hot encode selected categorical columns
categorical_cols = ["LegalSubdivision", "ZonedCodeLocal", "Exterior1Code", "PorchCode", "FenceCode"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Split features and target
X = df.drop(columns=[target])
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

### Neural Network ###
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss=MeanSquaredError(),
    metrics=[MeanAbsoluteError()]
)

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate neural network
nn_loss, nn_mae = model.evaluate(X_test, y_test)
print(f"Neural Network MAE: {nn_mae:.2f}")

### Baseline Models ###

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_preds = lin_reg.predict(X_test)
lin_mae = mean_absolute_error(y_test, lin_preds)
print(f"Linear Regression MAE: {lin_mae:.2f}")

# Decision Tree
tree = DecisionTreeRegressor(max_depth=10, random_state=42)
tree.fit(X_train, y_train)
tree_preds = tree.predict(X_test)
tree_mae = mean_absolute_error(y_test, tree_preds)
print(f"Decision Tree MAE: {tree_mae:.2f}")

### Sample Predictions ###
nn_preds = model.predict(X_test)
print(f"Sample Neural Net Predictions:\n{nn_preds[:5]}")

model.save("liberty_model.h5")

import joblib
joblib.dump(scaler, "scaler.pkl")
# Assuming df is your full DataFrame with features + target
X = df.drop(columns=["TransferAmount"])  # drop the target
y = df["TransferAmount"]

# Save only feature column order
joblib.dump(X.columns.tolist(), "model_columns.pkl")  # or "column_order.pkl"
