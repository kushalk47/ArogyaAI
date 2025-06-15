# app/config.py
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

load_dotenv()

connectionstring = os.getenv("URl")
if not connectionstring:
    logger.error("MongoDB URL (URl) not found in .env file. Please check your .env configuration.")
    raise ValueError("MongoDB URL not configured in .env file. Please set the 'URl' environment variable.")

MONGO_URL = connectionstring
logger.debug(f"Connecting to MongoDB with URL: {MONGO_URL[:30]}...") # Mask URL for logs

# --- START NEW DEBUGGING LOGIC ---
print("\n--- DEBUGGING AsyncIOMotorClient INITIALIZATION ---")
print(f"Current file: {__file__}")
print(f"Directory: {os.path.dirname(os.path.abspath(__file__))}")
print(f"os.getcwd(): {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not Set')}")
print(f"MONGODB_URL value being used: {MONGO_URL[:30]}...")
print(f"Type of MONGODB_URL: {type(MONGO_URL)}")
print(f"Value of serverSelectionTimeoutMS: {5000}")
print(f"Type of serverSelectionTimeoutMS: {type(5000)}")
print(f"Value of tls: {True}")
print(f"Type of tls: {type(True)}")
print("Attempting to initialize AsyncIOMotorClient...")
# --- END NEW DEBUGGING LOGIC ---

try:
    client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=5000, tls=True)

    # Note: client.admin.command('ping') is an async operation.
    # While it might work here due to Uvicorn's event loop,
    # it's generally better to await it or move it to an async startup event.
    # For now, keep it as is for debugging the *initialization* specifically.
    client.admin.command('ping')
    logger.info("MongoDB connection successful!")
except Exception as e:
    # --- START NEW DEBUGGING LOGIC ---
    print(f"!!! ERROR DURING AsyncIOMotorClient INITIALIZATION !!!")
    print(f"Error type: {type(e)}")
    print(f"Error message: {e}")
    # --- END NEW DEBUGGING LOGIC ---
    logger.error(f"Failed to connect to MongoDB: {e}", exc_info=True)
    raise ConnectionError(f"Could not connect to MongoDB: {e}")

db = client["healthcare_platform_db"]
logger.debug("Database 'healthcare_platform_db' selected.")

def init_db():
    logger.info("Database initialization function called (init_db in config.py).")
    pass

def get_db():
    return db