

import pandas as pd
import pickle
from train_contextual_transformer import TransformerModel
from predict_tmax_contextual_v2 import infer
import torch
import argparse
import numpy as np # Import numpy
import asyncio
import os
import libsql_client
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import feature creation functions from marine_features.py
from marine_features import create_nowcast_features, create_new_marine_features

# Load environment variables from .env file
load_dotenv()

# Turso database configuration
DB_URL = os.getenv("TURSO_DATABASE_URL").replace("libsql", "https")
AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

async def predict_today():
    # Load historical data from Turso database
    async with libsql_client.create_client(url=DB_URL, auth_token=AUTH_TOKEN) as client:
        rs_nowcast = await client.execute("SELECT * FROM historical_nowcast_features")
        historical_context = pd.DataFrame(rs_nowcast.rows, columns=[col for col in rs_nowcast.columns])
        
        rs_marine = await client.execute("SELECT * FROM historical_marine_features")
        historical_marine_features = pd.DataFrame(rs_marine.rows, columns=[col for col in rs_marine.columns])

    # Load daily temperature data (still from CSV)
    daily_temp = pd.read_csv('data/klax_daily.csv')
    
    # Convert DATE columns to datetime objects for historical data
    daily_temp['DATE'] = pd.to_datetime(daily_temp['DATE'])
    historical_context['DATE'] = pd.to_datetime(historical_context['DATE'])
    historical_marine_features['DATE'] = pd.to_datetime(historical_marine_features['DATE'])

    # Drop 'dewpoint_spread_5am' from historical_context to avoid duplication during merge
    if 'dewpoint_spread_5am' in historical_context.columns:
        historical_context = historical_context.drop(columns=['dewpoint_spread_5am'])

    # Merge all historical data
    historical_data_combined = pd.merge(historical_context, daily_temp, on='DATE', how='inner')
    historical_data_combined = pd.merge(historical_data_combined, historical_marine_features, on='DATE', how='inner')
    
    # Sort by date to ensure correct sequence selection
    historical_data_combined = historical_data_combined.sort_values(by='DATE').reset_index(drop=True)

    # --- Prepare current day's data ---
    # Get today's date for feature generation
    today_date = datetime.utcnow().date()
    
    # Generate current day's nowcast features directly
    nowcast_enriched_today_df = create_nowcast_features(target_date=datetime.utcnow()) # Pass current datetime
    if nowcast_enriched_today_df is None or nowcast_enriched_today_df.empty:
        print("Could not generate current day's nowcast features. Exiting.")
        return

    # Generate current day's marine features directly
    marine_features_today_df = create_new_marine_features(target_date=datetime.utcnow()) # Pass current datetime
    if marine_features_today_df is None or marine_features_today_df.empty:
        print("Could not generate current day's marine features. Exiting.")
        return

    # Combine current day's features into a single DataFrame row
    # Ensure DATE column is consistent for merging
    nowcast_enriched_today_df['DATE'] = pd.to_datetime(nowcast_enriched_today_df['DATE'])
    marine_features_today_df['DATE'] = pd.to_datetime(marine_features_today_df['DATE'])

    # Drop dewpoint_spread_5am from nowcast_enriched_today_df if it exists, to use the one from marine_features_today_df
    if 'dewpoint_spread_5am' in nowcast_enriched_today_df.columns:
        nowcast_enriched_today_df = nowcast_enriched_today_df.drop(columns=['dewpoint_spread_5am'])

    current_day_features = pd.merge(nowcast_enriched_today_df, marine_features_today_df, on='DATE', how='left')

    # Add TMAX and TMIN columns to current_day_features and set to NaN for prediction
    current_day_features['TMAX'] = np.nan
    current_day_features['TMIN'] = np.nan

    # Define the full set of features for the model input, in the correct order
    model_features = [
        'TMAX', 'TMIN', 'avg_wind_speed_6h', 'avg_wind_direction_6h', 'morning_cloud_cover_avg',
        'ceiling_slope', 'temp_slope_7_10am', 'sun_flag_8am', 'dewpoint_spread_5am'
    ]

    # Select and order columns for concatenation from the comprehensive historical data
    historical_data_for_concat = historical_data_combined[model_features + ['DATE']]
    current_day_features_for_concat = current_day_features[model_features + ['DATE']]

    # Combine data
    last_14_days = historical_data_for_concat.tail(14)
    prediction_input_df = pd.concat([last_14_days, current_day_features_for_concat], ignore_index=True)
    
    # Fill any remaining NaNs that might result from merging or missing data
    prediction_input_df.ffill(inplace=True)
    prediction_input_df.bfill(inplace=True)

    

    # Load the new model and scaler
    model_path = 'model/klax_contextual_transformer_v2.pth'
    scaler_path = 'model/contextual_scaler_v2.pkl'

    # The input_dim is now 9
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
        
    # Predict
    prediction = infer(model, prediction_input_df, scaler)
    
    print(f"Predicted TMAX for today (with marine features): {prediction:.2f}°F")

if __name__ == '__main__':
    asyncio.run(predict_today())
