import numpy as np
import joblib
import tensorflow as tf

# load pre-trained model and preprocessor
model = tf.keras.models.load_model('miami_value_model.h5')
preprocessor = joblib.load('preprocessor.pkl')

# feature order — must match training
feature_order = [
    'AreaBuilding', 'BedroomsCount', 'BathCount', 'YearBuilt', 'AreaLotSF',
    'ParkingSpaceCount', 'Pool', 'Fireplace', 'StoriesCount',
    'PropertyUseStandardized', 'RoomsCount'
]

def predict_home_value(user_input: dict) -> float:
    """
    accepts a dict of user input (like from React frontend) & returns predicted market value
    """
    # ensure order of features
    input_df = {key: [user_input[key]] for key in feature_order}
    input_df = pd.DataFrame(input_df)
    
    #apply saved preprocessing (scaling, encoding, passthrough)
    transformed_input = preprocessor.transform(input_df)
    
    # predict using the trained model
    predicted_value = model.predict(transformed_input)[0][0]
    
    return round(predicted_value, 2)
