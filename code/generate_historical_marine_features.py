
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import os
import numpy as np
import asyncio
import libsql_client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Assuming marine_features.py is in the same directory or accessible in PYTHONPATH
from marine_features import create_nowcast_features, create_new_marine_features

# Turso database configuration
DB_URL = os.getenv("TURSO_DATABASE_URL").replace("libsql", "https")
AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

async def init_db():
    async with libsql_client.create_client(url=DB_URL, auth_token=AUTH_TOKEN) as client:
        # Create historical_nowcast_features table
        await client.execute('''
            CREATE TABLE IF NOT EXISTS historical_nowcast_features (
                DATE TEXT PRIMARY KEY,
                avg_wind_speed_6h REAL,
                avg_wind_direction_6h REAL,
                morning_cloud_cover_avg REAL,
                dewpoint_spread_5am REAL
            )
        ''')

        # Create historical_marine_features table
        await client.execute('''
            CREATE TABLE IF NOT EXISTS historical_marine_features (
                DATE TEXT PRIMARY KEY,
                ceiling_slope REAL,
                temp_slope_7_10am REAL,
                sun_flag_8am REAL,
                dewpoint_spread_5am REAL
            )
        ''')

async def generate_historical_features(start_year, end_year):
    await init_db()
    async with libsql_client.create_client(url=DB_URL, auth_token=AUTH_TOKEN) as client:
        # Get existing dates from the database to avoid re-fetching
        rs_nowcast = await client.execute("SELECT DATE FROM historical_nowcast_features")
        existing_nowcast_dates = {row[0] for row in rs_nowcast.rows}

        rs_marine = await client.execute("SELECT DATE FROM historical_marine_features")
        existing_marine_dates = {row[0] for row in rs_marine.rows}

        for year in range(start_year, end_year + 1):
            print(f"Processing year: {year}")
            current_date = datetime(year, 1, 1)
            while current_date.year == year:
                date_str = current_date.strftime('%Y-%m-%d')

                # Only fetch if data for this date is missing in either table
                if date_str not in existing_nowcast_dates or date_str not in existing_marine_dates:
                    print(f"  Fetching data for: {date_str}")
                    
                    # Fetch nowcast features
                    nowcast_df_for_day = create_nowcast_features(target_date=current_date)
                    if nowcast_df_for_day is None or nowcast_df_for_day.empty:
                        nowcast_df_for_day = pd.DataFrame({
                            'avg_wind_speed_6h': [np.nan],
                            'avg_wind_direction_6h': [np.nan],
                            'morning_cloud_cover_avg': [np.nan],
                            'dewpoint_spread_5am': [np.nan],
                            'DATE': [date_str]
                        })
                    
                    nowcast_numeric_cols = [col for col in nowcast_df_for_day.columns if col != 'DATE']
                    nowcast_data_processed = nowcast_df_for_day[nowcast_numeric_cols].astype(float).fillna(0).iloc[0]
                    
                    await client.execute(
                        "INSERT OR REPLACE INTO historical_nowcast_features (DATE, avg_wind_speed_6h, avg_wind_direction_6h, morning_cloud_cover_avg, dewpoint_spread_5am) VALUES (?, ?, ?, ?, ?)",
                        (date_str, nowcast_data_processed['avg_wind_speed_6h'], nowcast_data_processed['avg_wind_direction_6h'], nowcast_data_processed['morning_cloud_cover_avg'], nowcast_data_processed['dewpoint_spread_5am'])
                    )
                    existing_nowcast_dates.add(date_str)

                    # Fetch marine features
                    marine_df_for_day = create_new_marine_features(target_date=current_date)
                    if marine_df_for_day is None or marine_df_for_day.empty:
                        marine_df_for_day = pd.DataFrame({
                            'ceiling_slope': [np.nan],
                            'temp_slope_7_10am': [np.nan],
                            'sun_flag_8am': [np.nan],
                            'dewpoint_spread_5am': [np.nan],
                            'DATE': [date_str]
                        })

                    marine_numeric_cols = [col for col in marine_df_for_day.columns if col != 'DATE']
                    marine_data_processed = marine_df_for_day[marine_numeric_cols].astype(float).fillna(0).iloc[0]

                    await client.execute(
                        "INSERT OR REPLACE INTO historical_marine_features (DATE, ceiling_slope, temp_slope_7_10am, sun_flag_8am, dewpoint_spread_5am) VALUES (?, ?, ?, ?, ?)",
                        (date_str, marine_data_processed['ceiling_slope'], marine_data_processed['temp_slope_7_10am'], marine_data_processed['sun_flag_8am'], marine_data_processed['dewpoint_spread_5am'])
                    )
                    existing_marine_dates.add(date_str)

                current_date += timedelta(days=1)
                await asyncio.sleep(0.1) # Small delay between daily requests
            
            await asyncio.sleep(5) # Longer delay between yearly requests

    print(f"Historical features generated and saved to Turso.")

if __name__ == '__main__':
    # Determine the earliest date from existing data files
    klax_daily_df = pd.read_csv('data/klax_daily.csv')
    klax_historical_context_df = pd.read_csv('data/klax_historical_context.csv')

    min_date_daily = pd.to_datetime(klax_daily_df['DATE']).min()
    min_date_context = pd.to_datetime(klax_historical_context_df['DATE']).min()

    start_year = min(min_date_daily, min_date_context).year
    
    # End year is the current year
    end_year = datetime.utcnow().year

    print(f"Generating historical features from {start_year} to {end_year}")
    asyncio.run(generate_historical_features(start_year, end_year))
