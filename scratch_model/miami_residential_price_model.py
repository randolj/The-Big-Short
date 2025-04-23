import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping



import joblib


import time
start = time.time()



# load the actual cleaned data files (assessor + recorder)
tax_df = pd.read_csv('Miami-Dade_tax_assessor_cleaned.csv')
recorder_df = pd.read_csv('Miami-Dade_recorder_cleaned.csv')

# merge the datasets on AttomID (assuming 1:1 or many:1 mapping)
merged_df = pd.merge(tax_df, recorder_df, on='AttomID', how='left')







###### define the final selected features and the target variable
selected_features = [
    'AreaBuilding', 'BedroomsCount', 'BathCount', 'YearBuilt', 'AreaLotSF',
    'ParkingSpaceCount', 'Pool', 'Fireplace', 'StoriesCount',
    'PropertyUseStandardized', 'RoomsCount'
]
target_variable = 'TransferAmount'






# drop rows where any selected feature or the target is missing
merged_df = merged_df.dropna(subset=selected_features + [target_variable])

# Only keep houses priced between $50K and $1.5M
merged_df = merged_df[
    (merged_df['TransferAmount'] >= 50000) &
    (merged_df['TransferAmount'] <= 1500000)
]


# Sample just 10,000 rows for quick testing
merged_df = merged_df.sample(10000, random_state=42)



# separate features and target
X = merged_df[selected_features]
y = np.log1p(merged_df[target_variable])  # log(1 + value)

# define preprocessing: scale numeric, one-hot encode categorical
numeric_features = [
    'AreaBuilding', 'BedroomsCount', 'BathCount', 'YearBuilt',
    'AreaLotSF', 'ParkingSpaceCount', 'StoriesCount', 'RoomsCount'
]
categorical_features = ['PropertyUseStandardized']
binary_features = ['Pool', 'Fireplace']  # these will be passed through

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
], remainder='passthrough')  # Pool and Fireplace passed through





# apply preprocessing
X_processed = preprocessor.fit_transform(X)

# split data for training/testing
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)





# build and compile the neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# train with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stop], verbose=1)

print(f"Training took {time.time() - start:.2f} seconds")

# evaluate model
y_pred = model.predict(X_test).flatten()  # Make 1D
y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred)

# Ensure shapes match
y_test_actual = np.array(y_test_actual).flatten()

mae = np.mean(np.abs(y_pred_actual - y_test_actual))
print(f"Test MAE (actual dollars): ${mae:,.2f}")








# save model and preprocessor
model.save("miami_value_model.h5")
joblib.dump(preprocessor, "preprocessor.pkl")
