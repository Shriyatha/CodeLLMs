"""Main entry point for the Code-Gym backend API using FastAPI."""
import logging

from app.routes import courses, problems
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Code-Gym API",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(courses.router, prefix="/api/courses")
app.include_router(problems.router, prefix="/api/problems")

@app.get("/")
async def root() -> dict:
    """Return a welcome message indicating the API is running."""
    return {"message": "Code-Gym API is running"}

@app.get("/health")
async def health_check() -> dict:
    """Return the health status of the API."""
    return {"status": "healthy"}
