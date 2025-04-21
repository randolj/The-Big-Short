import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2


# Load datasets
property_df = pd.read_csv("Liberty_recorder_cleaned.csv")
transaction_df = pd.read_csv("Liberty_tax_assessor_cleaned.csv")

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

# Feature columns and target
"""
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
    #"GraineryFlag", "GreenHouseArea", "GreenHouseFlag", "GuestHouseArea", "GuestHouseFlag", "KennelArea", "KennelFlag",
    #"LeanToArea", "LeanToFlag", "LoadingPlatformArea", "LoadingPlatformFlag", "MilkHouseArea", "MilkHouseFlag",
    #"OutdoorKitchenFireplaceFlag", "PoolHouseArea", "PoolHouseFlag", "PoultryHouseArea", "QuonsetArea", "QuonsetFlag",
    #"ShedArea", "ShedCode", "SiloArea", "SiloFlag", "StableArea", "StableFlag", "StorageBuildingArea", "StorageBuildingFlag",
    #"UtilityBuildingArea", "UtilityBuildingFlag", "PoleStructureArea", "PoleStructureFlag", "CommunityRecRoomFlag"
]
"""
features = ["TaxMarketValueTotal", "YearBuilt", "AreaBuilding", "MinorCivilDivisionCode", "NeighborhoodCode", "CensusTract",
    "PropertyLatitude", "PropertyLongitude",
    "TaxAssessedValueTotal",
    "TaxRateArea", "TaxExemptionHomeownerFlag",
    "PropertyUseStandardized", "AssessorLastSaleAmount", "AreaBuilding", "AreaLotAcres", "AreaLotSF", "ParkingGarage",
    "ParkingGarageArea", "ParkingCarport", "ParkingCarportArea",
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
    "BoatHouseFlag",  "GazeboArea", "GazeboFlag"
    ]
target = "TransferAmount"

# Drop missing values
df = df[features + [target]].dropna()

# One-hot encode selected categorical columns
#categorical_cols = [col for col in ["PropertyAddressZIP","LegalSubdivision", "ZonedCodeLocal", "Exterior1Code", "PorchCode", "FenceCode"] if col in df.columns]
categorical_cols = [col for col in ["PropertyAddressZIP"] if col in df.columns]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Split features and target
X = df.drop(columns=[target])
y = df[target]

# Preprocessing
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

k_best = 50  # you can tune this number
selector = SelectKBest(score_func=f_regression, k=k_best)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = X.columns[selector.get_support()].tolist()
print(f"Selected features: {selected_features}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

nn_rmse_scores = []
nn_mae_scores = []
nn_r2_scores = []

for fold, (train_index, val_index) in enumerate(kf.split(X_selected)):
    print(f"Training fold {fold+1}...")

    X_train, X_val = X_selected[train_index], X_selected[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model = Sequential([
    keras.Input(shape=(X_selected.shape[1],)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1)
])

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )

    val_predictions = model.predict(X_val).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    mae = mean_absolute_error(y_val, val_predictions)
    r2 = r2_score(y_val, val_predictions)

    nn_rmse_scores.append(rmse)
    nn_mae_scores.append(mae)
    nn_r2_scores.append(r2)

    print(f"Fold {fold+1} RMSE: {rmse:.4f} | MAE: {mae:.2f} | R²: {r2:.4f}")


### Baseline Models ###

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_preds = lin_reg.predict(X_test)
lin_mae = mean_absolute_error(y_test, lin_preds)
print(f"Linear Regression MAE: {lin_mae:.2f}")

# Decision Tree
#tree = DecisionTreeRegressor(max_depth=10, random_state=42)
#tree.fit(X_train, y_train)
#tree_preds = tree.predict(X_test)
#tree_mae = mean_absolute_error(y_test, tree_preds)
#print(f"Decision Tree MAE: {tree_mae:.2f}")

print(f"Selected features: {selected_features}")

### Sample Predictions ###
nn_preds = model.predict(X_test)
print(f"Sample Neural Net Predictions:\n{nn_preds[:5]}")

# R² for Neural Network
print(f"Neural Net K-Fold Avg MAE: {np.mean(nn_mae_scores):.2f}")
print(f"Neural Net K-Fold Avg R²: {np.mean(nn_r2_scores):.4f}")

nn_r2 = r2_score(y_test, nn_preds)
print(f"Neural Network R²: {nn_r2:.4f}")

# R² for Linear Regression
lin_r2 = r2_score(y_test, lin_preds)
print(f"Linear Regression R²: {lin_r2:.4f}")

# R² for Decision Tree
#tree_r2 = r2_score(y_test, tree_preds)
#print(f"Decision Tree R²: {tree_r2:.4f}")

model.save("liberty_model.h5")

import joblib
joblib.dump(scaler, "scaler.pkl")
# Assuming df is your full DataFrame with features + target
X = df.drop(columns=["TransferAmount"])  # drop the target
y = df["TransferAmount"]

# Save only feature column order
joblib.dump(X.columns.tolist(), "model_columns.pkl")  # or "column_order.pkl"
