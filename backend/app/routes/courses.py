from fastapi import APIRouter, HTTPException, Path, Depends
from typing import Dict, List
from pydantic import BaseModel
from app.services.course_loader import CourseLoader
import logging


router = APIRouter(prefix="", tags=["courses"])
logger = logging.getLogger(__name__)

# Response Models
class CourseSummary(BaseModel):
    id: str
    title: str
    description: str

class TopicSummary(BaseModel):
    id: str
    title: str
    description: str

class CourseDetails(BaseModel):
    id: str
    title: str
    description: str
    topics: List[TopicSummary]

# Dependency
def get_course_loader():
    return CourseLoader()

@router.get("/", response_model=Dict[str, CourseSummary], summary="List all available courses")
async def list_courses(loader: CourseLoader = Depends(get_course_loader)):
    """
    Returns a dictionary of all available courses where:
    - Key is the course ID
    - Value contains basic course information
    """
    try:
        courses = await loader.load_all_courses()
        if not courses:
            logger.warning("No courses found in the system")
            return {}
        
        return {
            course_id: CourseSummary(**{
                "id": course["id"],
                "title": course["title"],
                "description": course["description"]
            })
            for course_id, course in courses.items()
        }
    except Exception as e:
        logger.error(f"Failed to list courses: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve course list"
        )

@router.get("/{course_id}", response_model=CourseDetails, summary="Get detailed course information")
async def get_course_details(
    course_id: str = Path(..., description="The ID of the course to retrieve"),
    loader: CourseLoader = Depends(get_course_loader)
):
    """
    Returns detailed information about a specific course including:
    - Course title and description
    - List of all topics in the course
    """
    try:
        course = await loader.get_course(course_id)
        if not course:
            logger.warning(f"Course not found: {course_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Course {course_id} not found"
            )
        
        return CourseDetails(**{
            "id": course["id"],
            "title": course["title"],
            "description": course["description"],
            "topics": [
                {
                    "id": topic["id"],
                    "title": topic["title"],
                    "description": topic.get("description", "")
                }
                for topic in course.get("topics", [])
            ]
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get course details for {course_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve course details"
        )

@router.get("/{course_id}/topics", response_model=List[TopicSummary], summary="List all topics in a course")
async def list_course_topics(
    course_id: str = Path(..., description="The ID of the course"),
    loader: CourseLoader = Depends(get_course_loader)
):
    """
    Returns a list of all topics available in the specified course
    """
    try:
        course = await loader.get_course(course_id)
        if not course:
            raise HTTPException(
                status_code=404,
                detail=f"Course {course_id} not found"
            )
        
        return [
            TopicSummary(**{
                "id": topic["id"],
                "title": topic["title"],
                "description": topic.get("description", "")
            })
            for topic in course.get("topics", [])
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list topics for course {course_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve course topics"
        )