import requests
import time
import os
from datetime import datetime, timedelta
import json
import asyncio
import libsql_client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Turso database configuration
DB_URL = os.getenv("TURSO_DATABASE_URL").replace("libsql", "https")
AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

# LAX coordinates for Open-Meteo API
LAX_LAT = 33.9425
LAX_LON = -118.4081

async def init_db():
    """Initializes the Turso database and creates the historical_cli_reports table."""
    async with libsql_client.create_client(url=DB_URL, auth_token=AUTH_TOKEN) as client:
        await client.execute('''
            CREATE TABLE IF NOT EXISTS historical_cli_reports(
                DATE TEXT PRIMARY KEY,
                tmax_cli_f REAL,
                tmax_cli_c REAL,
                source_url TEXT
            )
        ''')

def fetch_historical_temps_batch(start_date, end_date):
    """Fetches historical daily max temperatures from Open-Meteo API for a date range."""
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    url = (f'https://archive-api.open-meteo.com/v1/archive'
           f'?latitude={LAX_LAT}&longitude={LAX_LON}'
           f'&start_date={start_str}&end_date={end_str}'
           f'&daily=temperature_2m_max'
           f'&temperature_unit=fahrenheit'
           f'&timezone=America/Los_Angeles')
    
    print(f'Fetching data from {start_str} to {end_str}...')
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'daily' in data and 'time' in data['daily'] and 'temperature_2m_max' in data['daily']:
            dates = data['daily']['time']
            temps_f = data['daily']['temperature_2m_max']
            
            results = []
            for date_str, temp_f in zip(dates, temps_f):
                if temp_f is not None:
                    temp_c = (temp_f - 32) * 5 / 9
                    results.append((date_str, temp_f, temp_c, url))
            
            print(f'Successfully fetched {len(results)} temperature records')
            return results
        else:
            print(f'Warning: Unexpected API response format: {data}')
            return []
            
    except requests.exceptions.RequestException as e:
        print(f'Error fetching data for {start_str} to {end_str}: {e}')
        return []
    except json.JSONDecodeError as e:
        print(f'Error parsing JSON response: {e}')
        return []

async def scrape_klax_cli_reports():
    """Scrapes KLAX historical temperature data using Open-Meteo API and stores in Turso."""
    print("Starting KLAX CLI temperature data collection using Open-Meteo API...")
    
    await init_db()
    async with libsql_client.create_client(url=DB_URL, auth_token=AUTH_TOKEN) as client:
        rs_dates = await client.execute("SELECT DATE FROM historical_cli_reports")
        existing_dates = {row[0] for row in rs_dates.rows}
        print(f"Found {len(existing_dates)} existing records in database")

        start_date = datetime(2000, 1, 1).date()
        end_date = datetime.now().date() - timedelta(days=1)
        
        print(f"Target date range: {start_date} to {end_date}")
        print(f"Total days to process: {(end_date - start_date).days + 1}")

        current_date = start_date
        total_new_records = 0
        total_api_calls = 0
        
        while current_date <= end_date:
            month_end = min(
                current_date.replace(day=1) + timedelta(days=32),
                end_date
            ).replace(day=1) - timedelta(days=1)
            
            if month_end > end_date:
                month_end = end_date
            
            month_dates = []
            check_date = current_date
            while check_date <= month_end:
                if check_date.strftime('%Y-%m-%d') not in existing_dates:
                    month_dates.append(check_date)
                check_date += timedelta(days=1)
            
            if month_dates:
                print(f"\nProcessing {len(month_dates)} missing dates from {current_date.strftime('%Y-%m')}...")
                
                batch_results = fetch_historical_temps_batch(current_date, month_end)
                total_api_calls += 1
                
                new_records_this_batch = 0
                for date_str, temp_f, temp_c, source_url in batch_results:
                    if date_str not in existing_dates:
                        if 30 <= temp_f <= 130:
                            await client.execute(
                                "INSERT OR REPLACE INTO historical_cli_reports (DATE, tmax_cli_f, tmax_cli_c, source_url) VALUES (?, ?, ?, ?)",
                                (date_str, temp_f, temp_c, source_url)
                            )
                            new_records_this_batch += 1
                            total_new_records += 1
                            print(f"  {date_str}: {temp_f:.1f}°F ({temp_c:.1f}°C)")
                        else:
                            print(f"  Warning: Skipping {date_str} - invalid temperature {temp_f}°F")
                
                if new_records_this_batch > 0:
                    print(f"  Inserted {new_records_this_batch} new records for {current_date.strftime('%Y-%m')}")
                
                await asyncio.sleep(0.5)
            else:
                print(f"Skipping {current_date.strftime('%Y-%m')} - all data already in database")
            
            current_date = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1)

    print(f"\n=== KLAX CLI Data Collection Complete ===")
    print(f"Date range processed: {start_date} to {end_date}")
    print(f"Total new records added: {total_new_records}")
    print(f"Total API calls made: {total_api_calls}")
    
    async with libsql_client.create_client(url=DB_URL, auth_token=AUTH_TOKEN) as client:
        rs_summary = await client.execute("SELECT COUNT(*), MIN(DATE), MAX(DATE) FROM historical_cli_reports")
        count, min_date, max_date = rs_summary.rows[0]
        print(f"Database now contains {count} total records from {min_date} to {max_date}")

if __name__ == '__main__':
    asyncio.run(scrape_klax_cli_reports())
