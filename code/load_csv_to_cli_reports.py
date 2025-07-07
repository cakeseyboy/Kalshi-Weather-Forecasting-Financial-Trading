import pandas as pd
import os
import asyncio
import libsql_client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CSV_FILE = 'data/klax_contextual_training_data.csv'

# Turso database configuration
DB_URL = os.getenv("TURSO_DATABASE_URL").replace("libsql", "https")
AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

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

async def load_csv_to_cli_reports():
    """Load temperature data from CSV into the historical_cli_reports table in Turso."""
    print(f"Loading temperature data from {CSV_FILE}...")
    
    df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(df)} rows from CSV")
    
    await init_db()
    async with libsql_client.create_client(url=DB_URL, auth_token=AUTH_TOKEN) as client:
        rs_count = await client.execute('SELECT COUNT(*) FROM historical_cli_reports')
        existing_count = rs_count.rows[0][0]
        print(f"Existing records in CLI reports table: {existing_count}")
        
        rs_dates = await client.execute('SELECT DATE FROM historical_cli_reports')
        existing_dates = {row[0] for row in rs_dates.rows}
        
        records_added = 0
        records_updated = 0
        
        for _, row in df.iterrows():
            date_str = row['DATE']
            tmax_f = row['TMAX']
            tmax_c = (tmax_f - 32) * 5/9
            source_url = "klax_contextual_training_data.csv"
            
            await client.execute(
                'INSERT OR REPLACE INTO historical_cli_reports (DATE, tmax_cli_f, tmax_cli_c, source_url) VALUES (?, ?, ?, ?)',
                (date_str, tmax_f, tmax_c, source_url)
            )
            
            if date_str not in existing_dates:
                records_added += 1
            else:
                records_updated += 1
        
        rs_final_count = await client.execute('SELECT COUNT(*) FROM historical_cli_reports')
        final_count = rs_final_count.rows[0][0]
        
        rs_date_range = await client.execute('SELECT MIN(DATE), MAX(DATE) FROM historical_cli_reports')
        min_date, max_date = rs_date_range.rows[0]
        
        print(f"\n=== Summary ===")
        print(f"Records in database: {final_count}")
        print(f"Date range: {min_date} to {max_date}")
        print(f"Database updated successfully!")
        
        rs_recent_data = await client.execute('SELECT * FROM historical_cli_reports ORDER BY DATE DESC LIMIT 5')
        
        print(f"\n=== Recent Data Sample ===")
        for row in rs_recent_data.rows:
            print(f"  {row[0]}: {row[1]}°F ({row[2]:.1f}°C)")

if __name__ == '__main__':
    asyncio.run(load_csv_to_cli_reports()) 