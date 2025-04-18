"""Courses API endpoints for managing course information."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel

from app.services.course_loader import CourseLoader

router = APIRouter(prefix="", tags=["courses"])
logger = logging.getLogger(__name__)


# Response Models
class CourseSummary(BaseModel):
    """Basic course information summary model."""

    id: str
    title: str
    description: str


class TopicSummary(BaseModel):
    """Basic topic information summary model."""

    id: str
    title: str
    description: str


class CourseDetails(BaseModel):
    """Detailed course information model including topics."""

    id: str
    title: str
    description: str
    topics: list[TopicSummary]


# Dependency
def get_course_loader() -> CourseLoader:
    """Get an instance of the course loader service."""
    return CourseLoader()


@router.get("/", summary="List all available courses")
async def list_courses(
    loader: Annotated[CourseLoader, Depends(get_course_loader)],
) -> dict[str, CourseSummary]:
    """Return a dictionary of all available courses.

    - Key is the course ID
    - Value contains basic course information
    """
    try:
        courses = await loader.load_all_courses()
        if not courses:
            logger.warning("No courses found in the system")
            return {}

        return {
            course_id: CourseSummary(
                id=course["id"],
                title=course["title"],
                description=course["description"],
            )
            for course_id, course in courses.items()
        }
    except Exception as e:
        logger.exception("Failed to list courses")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve course list",
        ) from e


@router.get("/{course_id}", summary="Get detailed course information")
async def get_course_details(
    course_id: Annotated[str, Path(..., description="The ID of the course to retrieve")],
    loader: Annotated[CourseLoader, Depends(get_course_loader)],
) -> CourseDetails:
    """Return detailed information about a specific course.

    - Course title and description
    - List of all topics in the course
    """
    try:
        course = await loader.get_course(course_id)
        if not course:
            logger.warning("Course not found: %s", course_id)
            _raise_not_found(course_id)

        return CourseDetails(
            id=course["id"],
            title=course["title"],
            description=course["description"],
            topics=[
                {
                    "id": topic["id"],
                    "title": topic["title"],
                    "description": topic.get("description", ""),
                }
                for topic in course.get("topics", [])
            ],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get course details for %s", course_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve course details",
        ) from e


@router.get(
    "/{course_id}/topics",
    summary="List all topics in a course",
)
async def list_course_topics(
    course_id: Annotated[str, Path(..., description="The ID of the course")],
    loader: Annotated[CourseLoader, Depends(get_course_loader)],
) -> list[TopicSummary]:
    """Return a list of all topics available in the specified course."""
    try:
        course = await loader.get_course(course_id)
        if not course:
            _raise_not_found(course_id)

        return [
            TopicSummary(
                id=topic["id"],
                title=topic["title"],
                description=topic.get("description", ""),
            )
            for topic in course.get("topics", [])
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to list topics for course %s", course_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve course topics",
        ) from e


def _raise_not_found(course_id: str) -> None:
    """Raise a 404 not found exception for the specified course ID."""
    raise HTTPException(
        status_code=404,
        detail=f"Course {course_id} not found",
    )
