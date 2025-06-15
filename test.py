import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
import logging

logging.basicConfig(level=logging.DEBUG) # Set logging level to DEBUG
logger = logging.getLogger(__name__)

async def test_connection():
    load_dotenv()
    connection_string = os.getenv("URl")

    if not connection_string:
        logger.error("MongoDB URL (URl) not found in .env file.")
        print("Failed: MongoDB URL not configured.")
        return

    logger.info(f"Attempting to connect to MongoDB with URL: {connection_string[:30]}...") # Mask part of URL for security

    try:
        # This is the line that's problematic in your app, let's test it directly
        client = AsyncIOMotorClient(connection_string, serverSelectionTimeoutMS=5000, tls=True)

        # Attempt to ping the admin database to verify connection
        await client.admin.command('ping')
        print("Success: MongoDB connection established and ping successful!")
    except Exception as e:
        print(f"Failed: Could not connect to MongoDB. Error: {e}")
        logger.exception("Detailed error during MongoDB connection test:")

if __name__ == "__main__":
    asyncio.run(test_connection())