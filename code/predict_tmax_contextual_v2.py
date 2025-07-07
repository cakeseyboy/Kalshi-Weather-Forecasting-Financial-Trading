
import torch
import pickle
import numpy as np
import pandas as pd
from train_contextual_transformer import TransformerModel

def infer(model, data, scaler):
    model.eval()
    
    # Define the expanded feature set, matching the training order
    features = [
        'TMAX', 'TMIN', 'avg_wind_speed_6h', 'avg_wind_direction_6h', 'morning_cloud_cover_avg', 
        'ceiling_slope', 'temp_slope_7_10am', 'sun_flag_8am', 'dewpoint_spread_5am'
    ]
    
    # Ensure data is a pandas DataFrame and has the correct columns
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=features) # Assuming data is passed as a numpy array or list of lists

    # Ensure all required features are present in the input data
    for col in features:
        if col not in data.columns:
            # This should ideally not happen if data preparation is correct
            # For now, we'll raise an error, but in a production system, you might impute or handle differently
            raise ValueError(f"Missing feature '{col}' in input data for inference.")

    # Select and order features as expected by the scaler and model
    data_to_scale = data[features].values

    # Normalize the input data
    data_scaled = scaler.transform(data_to_scale)
    
    # Reshape for the model (add batch dimension)
    data_tensor = torch.from_numpy(data_scaled).float().unsqueeze(0) 

    with torch.no_grad():
        prediction_scaled = model(data_tensor).squeeze().numpy()

    # Inverse transform the prediction
    # Create a dummy array for inverse transform. The number of columns must match the scaler's original fit.
    num_features = len(features)
    dummy_pred = np.zeros((1, num_features))
    dummy_pred[0, 0] = prediction_scaled # Assuming TMAX is the first feature in the scaler's fit
    prediction_inv = scaler.inverse_transform(dummy_pred)[0, 0]
    
    return prediction_inv

if __name__ == '__main__':
    # Load the new model and scaler
    model_path = 'model/klax_contextual_transformer_v2.pth'
    scaler_path = 'model/contextual_scaler_v2.pkl'

    # Update input_dim to match the retrained model (9 features)
    input_dim = 9 
    model_dim = 64
    nhead = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 128
    dropout = 0.1

    model = TransformerModel(input_dim, model_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
    model.load_state_dict(torch.load(model_path))
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Create some dummy data for inference with all 9 features
    # Ensure the order matches the 'features' list in the infer function
    dummy_data = pd.DataFrame({
        'TMAX': np.random.randint(60, 80, 14),
        'TMIN': np.random.randint(40, 60, 14),
        'avg_wind_speed_6h': np.random.uniform(0, 10, 14),
        'avg_wind_direction_6h': np.random.uniform(0, 360, 14),
        'morning_cloud_cover_avg': np.random.uniform(0, 100, 14),
        'ceiling_slope': np.random.uniform(-100, 100, 14),
        'temp_slope_7_10am': np.random.uniform(0, 5, 14),
        'sun_flag_8am': np.random.randint(0, 2, 14),
        'dewpoint_spread_5am': np.random.uniform(0, 20, 14)
    })
    
    prediction = infer(model, dummy_data, scaler)
    print(f'Example Inference Prediction: {prediction:.2f}')
