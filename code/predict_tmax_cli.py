import torch
import pickle
import numpy as np
import pandas as pd
import asyncio
import os
import libsql_client
from datetime import datetime, timedelta
from train_contextual_transformer_cli import CliTransformerModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Turso database configuration
DB_URL = os.getenv("TURSO_DATABASE_URL").replace("libsql", "https")
AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

async def load_recent_features(days=14):
    """
    Load the most recent 14 days of features from the database for prediction.
    """
    async with libsql_client.create_client(url=DB_URL, auth_token=AUTH_TOKEN) as client:
        # Load recent data from all tables
        rs_nowcast = await client.execute("SELECT * FROM historical_nowcast_features ORDER BY DATE DESC LIMIT ?", (days,))
        nowcast_df = pd.DataFrame(rs_nowcast.rows, columns=[col for col in rs_nowcast.columns])
        
        rs_marine = await client.execute("SELECT * FROM historical_marine_features ORDER BY DATE DESC LIMIT ?", (days,))
        marine_df = pd.DataFrame(rs_marine.rows, columns=[col for col in rs_marine.columns])
    
    # Convert dates and sort ascending (oldest first)
    nowcast_df['DATE'] = pd.to_datetime(nowcast_df['DATE'])
    marine_df['DATE'] = pd.to_datetime(marine_df['DATE'])
    
    nowcast_df = nowcast_df.sort_values('DATE')
    marine_df = marine_df.sort_values('DATE')
    
    # Clean marine features
    marine_df['sun_flag_8am'] = pd.to_numeric(marine_df['sun_flag_8am'], errors='coerce').fillna(0)
    
    # Handle dewpoint duplication
    if 'dewpoint_spread_5am' in nowcast_df.columns:
        nowcast_df = nowcast_df.drop(columns=['dewpoint_spread_5am'])
    
    # Join features
    df = pd.merge(nowcast_df, marine_df, on='DATE', how='inner')
    
    if len(df) < days:
        print(f"Warning: Only {len(df)} days of recent data available (requested {days})")
    
    return df

def predict_cli_tmax(model, recent_data, scaler, target_date=None):
    """
    Predict CLI TMAX for the next day using the transformer model.
    
    Args:
        model: Trained CLI transformer model
        recent_data: DataFrame with recent 14 days of features
        scaler: Fitted StandardScaler from training
        target_date: Optional date for prediction (for logging)
    
    Returns:
        dict: Prediction results with raw and rounded values
    """
    model.eval()
    
    # Define features in the exact order used during training
    features = [
        'avg_wind_speed_6h', 'avg_wind_direction_6h', 'morning_cloud_cover_avg',
        'ceiling_slope', 'temp_slope_7_10am', 'sun_flag_8am', 'dewpoint_spread_5am'
    ]
    
    # Validate input data
    for feature in features:
        if feature not in recent_data.columns:
            raise ValueError(f"Missing required feature: {feature}")
    
    if len(recent_data) < 14:
        raise ValueError(f"Need 14 days of data, got {len(recent_data)}")
    
    # Use the most recent 14 days
    recent_features = recent_data[features].tail(14)
    
    # Normalize features using the training scaler
    features_scaled = scaler.transform(recent_features.values)
    
    # Create tensor for model input
    input_tensor = torch.from_numpy(features_scaled).float().unsqueeze(0)  # Shape: (1, 14, 7)
    
    # Make prediction
    with torch.no_grad():
        raw_prediction = model(input_tensor).squeeze().item()
    
    # Round to nearest integer (Kalshi settlement style)
    cli_prediction = round(raw_prediction)
    
    # Prepare results
    result = {
        'raw_prediction': raw_prediction,
        'cli_settlement': cli_prediction,
        'target_date': target_date or (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
        'confidence_level': _calculate_confidence(raw_prediction),
        'features_used': features,
        'input_date_range': f"{recent_data['DATE'].min().strftime('%Y-%m-%d')} to {recent_data['DATE'].max().strftime('%Y-%m-%d')}"
    }
    
    return result

def _calculate_confidence(raw_prediction):
    """
    Calculate confidence level based on distance from rounding boundary.
    """
    rounded_pred = round(raw_prediction)
    distance_from_boundary = abs(raw_prediction - rounded_pred)
    
    # Higher confidence when farther from 0.5 boundary
    confidence = (0.5 - distance_from_boundary) / 0.5 * 100
    confidence = max(0, min(100, confidence))  # Clamp to 0-100
    
    return confidence

def predict_cli_range(model, recent_data, scaler, target_temp_range):
    """
    Predict probability that CLI TMAX falls within a specific range (for Kalshi markets).
    
    Args:
        target_temp_range: tuple (min_temp, max_temp) for the range
    
    Returns:
        dict: Probability and recommendation
    """
    prediction = predict_cli_tmax(model, recent_data, scaler)
    cli_temp = prediction['cli_settlement']
    raw_temp = prediction['raw_prediction']
    
    min_temp, max_temp = target_temp_range
    
    # Check if rounded prediction falls in range
    in_range = min_temp <= cli_temp <= max_temp
    
    # Calculate probability based on raw prediction and uncertainty
    uncertainty = abs(raw_temp - cli_temp)  # Distance from rounded value
    
    if in_range:
        # If rounded prediction is in range, probability depends on how close to boundary
        prob = 75 + (25 * (1 - uncertainty))  # 75-100% if in range
    else:
        # If outside range, probability depends on how close to range
        distance_to_range = min(abs(cli_temp - min_temp), abs(cli_temp - max_temp))
        if distance_to_range == 1 and uncertainty > 0.3:
            prob = 25 + (25 * uncertainty)  # 25-50% if close and uncertain
        else:
            prob = max(5, 25 * (1 - distance_to_range/5))  # Lower if far from range
    
    prob = max(0, min(100, prob))  # Clamp to 0-100
    
    return {
        'target_range': target_temp_range,
        'predicted_cli': cli_temp,
        'probability': prob,
        'recommendation': 'YES' if prob > 55 else 'NO',
        'confidence': prediction['confidence_level'],
        'raw_prediction': raw_temp
    }

async def main():
    """
    Main function to demonstrate CLI prediction capabilities.
    """
    print("=== MARLIN v3: CLI Settlement Predictor ===")
    
    # Load the CLI model and scaler
    model_path = 'model/klax_contextual_transformer_cli.pth'
    scaler_path = 'model/contextual_cli_scaler.pkl'
    
    print(f"Loading CLI model from {model_path}")
    print(f"Loading CLI scaler from {scaler_path}")
    
    # Model architecture (must match training)
    input_dim = 7
    model_dim = 64
    nhead = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 128
    dropout = 0.1
    
    # Load model
    model = CliTransformerModel(
        input_dim, model_dim, nhead, 
        num_encoder_layers, num_decoder_layers, 
        dim_feedforward, dropout
    )
    model.load_state_dict(torch.load(model_path))
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print("✅ Model and scaler loaded successfully")
    
    # Load recent features for prediction
    print("\nLoading recent weather features...")
    recent_data = await load_recent_features(days=14)
    
    if len(recent_data) == 0:
        print("❌ No recent data available for prediction")
        return
    
    print(f"✅ Loaded {len(recent_data)} days of recent data")
    print(f"Date range: {recent_data['DATE'].min()} to {recent_data['DATE'].max()}")
    
    # Make CLI prediction
    print("\n=== CLI TMAX Prediction ===")
    prediction = predict_cli_tmax(model, recent_data, scaler)
    
    print(f"Target Date: {prediction['target_date']}")
    print(f"Raw Prediction: {prediction['raw_prediction']:.2f}°F")
    print(f"CLI Settlement: {prediction['cli_settlement']}°F")
    print(f"Confidence: {prediction['confidence_level']:.1f}%")
    print(f"Data Range: {prediction['input_date_range']}")
    
    # Example Kalshi market predictions
    print("\n=== Kalshi Market Analysis ===")
    
    # Common temperature ranges for KLAX markets
    test_ranges = [
        (70, 75),  # 70-75°F range
        (75, 80),  # 75-80°F range
        (65, 70),  # 65-70°F range
    ]
    
    for temp_range in test_ranges:
        market_pred = predict_cli_range(model, recent_data, scaler, temp_range)
        print(f"\nMarket: Will KLAX high be {temp_range[0]}-{temp_range[1]}°F?")
        print(f"  Predicted CLI: {market_pred['predicted_cli']}°F")
        print(f"  Probability: {market_pred['probability']:.1f}%")
        print(f"  Recommendation: {market_pred['recommendation']}")
    
    print(f"\n🎯 Ready for Kalshi trading!")

if __name__ == '__main__':
    asyncio.run(main()) 