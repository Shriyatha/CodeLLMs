from fastapi import FastAPI
from .routes import courses, problems

app = FastAPI(title="Code-Gym API")

# Include routers with proper prefixes
app.include_router(
    courses.router,
    prefix="/api/courses",
    tags=["courses"]
)

app.include_router(
    problems.router,
    prefix="/api/problems",
    tags=["problems"]
)