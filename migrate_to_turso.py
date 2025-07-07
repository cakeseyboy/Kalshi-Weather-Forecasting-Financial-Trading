import os
import asyncio
import libsql_client
from dotenv import load_dotenv

load_dotenv()

async def migrate():
    """Migrates data from a local SQL dump to a Turso database."""
    db_url = os.getenv("TURSO_DATABASE_URL").replace("libsql", "https")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")

    if not db_url or not auth_token:
        print("Error: TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be set in .env")
        return

    try:
        async with libsql_client.create_client(url=db_url, auth_token=auth_token) as client:
            with open("/Users/matt/Documents/Kalshi-Weather-Forecasting-Financial-Trading/data/klax_weather.sql", "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            await client.execute(line.strip())
                        except KeyError:
                            pass
        print("Data migration successful!")
    except Exception as e:
        print(f"Error during data migration: {e}")

if __name__ == "__main__":
    asyncio.run(migrate())
