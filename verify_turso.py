import os
import asyncio
import libsql_client
from dotenv import load_dotenv

load_dotenv()

async def verify_turso_data():
    """Connects to Turso and verifies the migrated data."""
    db_url = os.getenv("TURSO_DATABASE_URL").replace("libsql", "https")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")

    if not db_url or not auth_token:
        print("Error: TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be set in .env")
        return

    try:
        async with libsql_client.create_client(url=db_url, auth_token=auth_token) as client:
            # 1. List tables
            rs = await client.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row["name"] for row in rs.rows]
            print(f"Found tables: {tables}")

            # 2. Count rows in each table
            for table in tables:
                if not table.startswith("libsql_"): # Skip internal tables
                    rs = await client.execute(f"SELECT COUNT(*) FROM {table}")
                    count = rs.rows[0][0]
                    print(f"- Table '{table}' has {count} rows.")
            
            # 3. Check recent training logs
            print("\n--- Recent Training Logs (training_runs_log) ---")
            try:
                rs_logs = await client.execute("SELECT * FROM training_runs_log ORDER BY id DESC LIMIT 5")
                if rs_logs and hasattr(rs_logs, 'rows') and rs_logs.rows:
                    for row in rs_logs.rows:
                        print(row)
                else:
                    print(f"No training logs found or empty result set. Type: {type(rs_logs)}")
            except Exception as e:
                print(f"Error fetching training logs: {e}")

            # 4. Check recent model metadata
            print("\n--- Recent Model Metadata (model_metadata) ---")
            try:
                rs_metadata = await client.execute("SELECT * FROM model_metadata ORDER BY training_date DESC LIMIT 5")
                if rs_metadata and hasattr(rs_metadata, 'rows') and rs_metadata.rows:
                    for row in rs_metadata.rows:
                        print(row)
                else:
                    print(f"No model metadata found or empty result set. Type: {type(rs_metadata)}")
            except Exception as e:
                print(f"Error fetching model metadata: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(verify_turso_data())