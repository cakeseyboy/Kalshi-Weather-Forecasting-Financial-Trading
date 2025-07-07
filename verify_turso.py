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

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(verify_turso_data())