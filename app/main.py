from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.routes import router as api_router

app = FastAPI()

# Mount static files and template engine
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.templates = Jinja2Templates(directory="app/templates")

# Include routers from the routes package
app.include_router(api_router)  # Includes all the routes defined in routes/__init__.py

# You can add any other app-wide middleware, event handlers, etc., here if necessary.
    