# run.py
from app.main import app # This line is correct, it imports 'app' from 'app.main'
import uvicorn

if __name__ == "__main__":
    # Change "run:app" to "app.main:app"
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)