"""Module for handling coding problem API endpoints.

This module provides endpoints for managing coding problems, including
code execution, hint generation, error explanation, and optimization suggestions.
"""

import logging
import re
from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import uuid4

from app.services.code_execution import CodeExecutionService
from app.services.course_loader import CourseLoader
from app.services.grading_service import SubmissionStorage, grade_submission_workflow
from app.services.llm_service import LLMService
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    status,
)
from pydantic import BaseModel, Field

router = APIRouter(prefix="", tags=["problems"])
logger = logging.getLogger(__name__)

# Constants
MAX_CODE_LENGTH = 10000
VALID_ID_PATTERN = r"^[a-zA-Z0-9_-]+$"
VALID_LANGUAGES = ["python", "javascript"]


# Models
class SubmissionRequest(BaseModel):
    """Request model for code submission."""

    code: str = Field(..., min_length=1)
    language: str = Field("python", pattern="^(python|javascript)$")
    user_id: str = Field(...)


class SubmissionResponse(BaseModel):
    """Response model for submission status and results."""

    submission_id: str = Field(...)
    status: str = Field(...)
    timestamp: datetime = Field(...)
    result: dict[str, Any] | None = Field(None)


class GradingResult(BaseModel):
    """Model representing the grading result of a submission."""

    passed: bool
    score: float = Field(..., ge=0, le=100)
    feedback: str
    execution_time: str
    test_results: list[dict[str, Any]]


class CodeSubmission(BaseModel):
    """Model for submitting code for execution, not grading."""

    code: str = Field(..., min_length=1)
    language: str = Field("python", pattern="^(python|javascript)$")


class ProblemResponse(BaseModel):
    """Response model for problem details."""

    id: str
    title: str
    description: str
    complexity: str = Field(..., pattern="^(easy|medium|hard)$")
    starter_code: str
    visible_test_cases: list[dict[str, Any]]


class TopicResponse(BaseModel):
    """Response model for topic details including available problems."""

    id: str
    title: str
    description: str
    problems: list[dict[str, Any]]


class HintResponse(BaseModel):
    """Response model for progressive hints."""

    hint_level: int = Field(..., ge=0)
    hint_text: str
    max_level: int = Field(..., ge=1)


class ErrorExplanation(BaseModel):
    """Response model for error explanations with suggestions."""

    error_type: str
    explanation: str
    suggested_fixes: list[str]
    original_error: str
    relevant_line: int | None = Field(None, ge=1)


class Complexity(BaseModel):
    """Model representing time and space complexity."""

    time: str
    space: str


class OptimizationSuggestion(BaseModel):
    """Response model for code optimization suggestions."""

    current_complexity: Complexity
    suggested_complexity: Complexity
    optimization_suggestions: list[str]
    readability_suggestions: list[str]
    best_practice_suggestions: list[str]
    edge_cases: list[str]
    explanation: str
    code_snippet: str


class ConceptualSteps(BaseModel):
    """Response model for conceptual problem-solving steps."""

    steps: list[str]
    current_step: int | None = Field(None, ge=0)


class DebuggingSuggestion(BaseModel):
    """Response model for debugging suggestions."""

    strategy: str
    variables_to_track: list[str]


class ExecutionResult(BaseModel):
    """Response model for code execution results."""

    visible_results: list[dict[str, Any]]
    hidden_passed: bool
    execution_time: str


class PseudocodeResponse(BaseModel):
    """Response model for pseudocode generation."""

    pseudocode: str
    explanation: str


# Helper functions
def raise_not_found(detail: str = "Resource not found") -> None:
    """Raise a 404 Not Found error with custom detail message.

    Args:
        detail: Custom error message to include in response.

    """
    raise HTTPException(status_code=404, detail=detail)


def raise_internal_error(detail: str = "Internal server error") -> None:
    """Raise a 500 Internal Server Error with custom detail message.

    Args:
        detail: Custom error message to include in response.

    """
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=detail,
    )


def raise_bad_request(detail: str) -> None:
    """Raise a 400 Bad Request error with custom detail message.

    Args:
        detail: Custom error message to include in response.

    """
    raise HTTPException(status_code=400, detail=detail)


def validate_id_format(id_value: str, id_name: str) -> None:
    """Validate ID format against allowed pattern.

    Args:
        id_value: The ID string to validate.
        id_name: Name of the ID for error messages.

    Raises:
        HTTPException: If ID format is invalid.

    """
    if not re.match(VALID_ID_PATTERN, id_value):
        raise HTTPException(status_code=400, detail=f"Invalid {id_name} ID format")


async def validate_problem_exists(
    course_id: str,
    topic_id: str,
    problem_id: str,
    loader: CourseLoader,
) -> dict[str, Any]:
    """Validate that a problem exists and return its data.

    Args:
        course_id: The course identifier.
        topic_id: The topic identifier.
        problem_id: The problem identifier.
        loader: Course loader service instance.

    Returns:
        Problem data dictionary if found.

    Raises:
        HTTPException: If validation fails or problem doesn't exist.

    """
    validate_id_format(course_id, "course")
    validate_id_format(topic_id, "topic")
    validate_id_format(problem_id, "problem")

    try:
        problem = await loader.get_problem(course_id, topic_id, problem_id)
        if not problem:
            logger.error("Problem not found: %s/%s/%s", course_id, topic_id, problem_id)
            raise_not_found("Problem not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error loading problem")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load problem",
        ) from e
    else:
        return problem


async def validate_code_submission(code: str, language: str = "python") -> None:
    """Validate code submission against basic requirements.

    Args:
        code: The code string to validate.
        language: Programming language of submitted code.

    Raises:
        HTTPException: If validation fails.

    """
    if not code.strip():
        raise HTTPException(status_code=400, detail="Empty code submission")
    if len(code) > MAX_CODE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Code too long (max {MAX_CODE_LENGTH:,} characters)",
        )
    if language not in VALID_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")


# Dependency injections
def get_execution_service() -> CodeExecutionService:
    """Create and return a CodeExecutionService instance.

    Returns:
        An initialized CodeExecutionService for code execution.

    Raises:
        HTTPException: If service initialization fails.

    """
    try:
        return CodeExecutionService()
    except Exception as e:
        logger.exception("Failed to initialize CodeExecutionService")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Code execution service unavailable",
        ) from e


def get_course_loader() -> CourseLoader:
    """Create and return a CourseLoader instance.

    Returns:
        An initialized CourseLoader for accessing course content.

    Raises:
        HTTPException: If service initialization fails.

    """
    try:
        return CourseLoader()
    except Exception as e:
        logger.exception("Failed to initialize CourseLoader")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Course loading service unavailable",
        ) from e


def get_llm_service() -> LLMService:
    """Create and return an LLM service instance.

    Returns:
        An initialized LLMService for AI-powered features.

    Raises:
        HTTPException: If service initialization fails.

    """
    try:
        return LLMService()
    except Exception as e:
        logger.exception("Failed to initialize LLMService")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service unavailable",
        ) from e


# Endpoints
@router.get(
    "/courses/{course_id}/topics/{topic_id}",
    responses={
        404: {"description": "Topic not found"},
        400: {"description": "Invalid ID format"},
        500: {"description": "Internal server error"},
    },
    response_model=TopicResponse,
)
async def get_topic_details(
    course_id: Annotated[str, Path(..., description="ID of the course", min_length=1)],
    topic_id: Annotated[str, Path(..., description="ID of the topic", min_length=1)],
    loader: Annotated[CourseLoader, Depends(get_course_loader)],
) -> TopicResponse:
    """Get topic information including description and list of problems.

    Args:
        course_id: The course identifier.
        topic_id: The topic identifier.
        loader: Injected course loader instance.

    Returns:
        TopicResponse with topic details and problems.

    Raises:
        HTTPException: If topic not found or other errors occur.

    """
    try:
        validate_id_format(course_id, "course")
        validate_id_format(topic_id, "topic")

        topic = await loader.get_topic(course_id, topic_id)
        if not topic:
            logger.error("Topic not found: %s/%s", course_id, topic_id)
            raise_not_found(f"Topic {topic_id} not found")

        return TopicResponse(
            id=topic_id,
            title=topic.get("title", ""),
            description=topic.get("description", ""),
            problems=topic.get("problems", []),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting topic details")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e


@router.get(
    "/courses/{course_id}/topics/{topic_id}/problems/{problem_id}",
    responses={
        404: {"description": "Problem not found"},
        400: {"description": "Invalid ID format"},
        500: {"description": "Internal server error"},
    },
    response_model=ProblemResponse,
)
async def get_problem_details(
    course_id: Annotated[str, Path(..., description="ID of the course", min_length=1)],
    topic_id: Annotated[str, Path(..., description="ID of the topic", min_length=1)],
    problem_id: Annotated[str, Path(..., description="ID of the problem", min_length=1)],
    loader: Annotated[CourseLoader, Depends(get_course_loader)],
) -> ProblemResponse:
    """Get detailed problem information including starter code and test cases.

    Args:
        course_id: The course identifier.
        topic_id: The topic identifier.
        problem_id: The problem identifier.
        loader: Injected course loader instance.

    Returns:
        ProblemResponse with problem details.

    Raises:
        HTTPException: If problem not found or other errors occur.

    """
    try:
        problem = await validate_problem_exists(course_id, topic_id, problem_id, loader)

        return ProblemResponse(
            id=problem_id,
            title=problem.get("title", ""),
            description=problem.get("description", ""),
            complexity=problem.get("complexity", "medium"),
            starter_code=problem.get("starter_code", ""),
            visible_test_cases=problem.get("visible_test_cases", []),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting problem details")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e


@router.post(
    "/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/execute",
    responses={
        400: {"description": "Invalid input or no test cases"},
        404: {"description": "Problem not found"},
        500: {"description": "Execution failed"},
    },
    response_model=ExecutionResult,
)
async def execute_problem(
    course_id: Annotated[str, Path(..., description="ID of the course", min_length=1)],
    topic_id: Annotated[str, Path(..., description="ID of the topic", min_length=1)],
    problem_id: Annotated[str, Path(..., description="ID of the problem", min_length=1)],
    submission: CodeSubmission,
    execution_service: Annotated[CodeExecutionService, Depends(get_execution_service)],
    loader: Annotated[CourseLoader, Depends(get_course_loader)],
) -> ExecutionResult:
    """Execute code against test cases in secure Docker container.

    Args:
        course_id: The course identifier.
        topic_id: The topic identifier.
        problem_id: The problem identifier.
        submission: Code submission to execute.
        execution_service: Code execution service instance.
        loader: Course loader service instance.

    Returns:
        ExecutionResult with test results.

    Raises:
        HTTPException: If execution fails or other errors occur.

    """
    try:
        await validate_code_submission(submission.code, submission.language)
        problem = await validate_problem_exists(
            course_id, topic_id, problem_id, loader,
        )

        test_cases = problem.get("visible_test_cases", []) + problem.get("hidden_test_cases", [])
        if not test_cases:
            logger.error("No test cases found for problem: %s", problem_id)
            raise_bad_request("No test cases provided")

        results = await execution_service.execute_code(
            code=submission.code,
            language=submission.language,
            test_cases=test_cases,
        )

        if not isinstance(results, list):
            logger.error("Unexpected results format: %s", type(results))
            raise_bad_request("Unexpected results format")

        visible_count = len(problem.get("visible_test_cases", []))
        visible_results = results[:visible_count]

        all_hidden_passed = True
        if len(results) > visible_count:
            all_hidden_passed = all(r.get("passed", False) for r in results[visible_count:])

        return ExecutionResult(
            visible_results=visible_results,
            hidden_passed=all_hidden_passed,
            execution_time=datetime.now(UTC).isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Execution failed")
        raise HTTPException(
            status_code=500,
            detail="Code execution failed",
        ) from e


@router.post(
    "/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/hints",
    responses={
        400: {"description": "Invalid input"},
        404: {"description": "Problem not found"},
        500: {"description": "Failed to generate hints"},
    },
    response_model=HintResponse,
)
async def get_progressive_hints(
    course_id: Annotated[str, Path(..., description="ID of the course", min_length=1)],
    topic_id: Annotated[str, Path(..., description="ID of the topic", min_length=1)],
    problem_id: Annotated[str, Path(..., description="ID of the problem", min_length=1)],
    submission: CodeSubmission,
    llm: Annotated[LLMService, Depends(get_llm_service)],
    loader: Annotated[CourseLoader, Depends(get_course_loader)],
) -> HintResponse:
    """Get tiered hints to guide student toward solution.

    Args:
        course_id: The course identifier.
        topic_id: The topic identifier.
        problem_id: The problem identifier.
        submission: Code submission to provide hints for.
        llm: LLM service for hint generation.
        loader: Course loader service instance.

    Returns:
        HintResponse with hint text and level information.

    Raises:
        HTTPException: If hint generation fails or other errors occur.

    """
    try:
        problem = await validate_problem_exists(
            course_id, topic_id, problem_id, loader,
        )
        await validate_code_submission(submission.code, submission.language)

        hints, max_level = await llm.get_progressive_hints(
            problem=problem["description"],
            code=submission.code,
            current_level=0,
        )

        return HintResponse(
            hint_level=0,
            hint_text=hints[0] if hints else "No hints available",
            max_level=max_level,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Hint generation failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate hints",
        ) from e


@router.post(
    "/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/explain",
    responses={
        400: {"description": "Invalid input or no errors"},
        404: {"description": "Problem not found"},
        500: {"description": "Failed to explain errors"},
    },
    response_model=ErrorExplanation,
)
async def explain_errors(
    course_id: Annotated[str, Path(..., description="ID of the course", min_length=1)],
    topic_id: Annotated[str, Path(..., description="ID of the topic", min_length=1)],
    problem_id: Annotated[str, Path(..., description="ID of the problem", min_length=1)],
    submission: CodeSubmission,
    execution_service: Annotated[CodeExecutionService, Depends(get_execution_service)],
    llm: Annotated[LLMService, Depends(get_llm_service)],
    loader: Annotated[CourseLoader, Depends(get_course_loader)],
) -> ErrorExplanation:
    """Explain errors in user's code with AI assistance.

    Args:
        course_id: The course identifier.
        topic_id: The topic identifier.
        problem_id: The problem identifier.
        submission: Code submission to explain errors for.
        execution_service: Code execution service instance.
        llm: LLM service for error analysis.
        loader: Course loader service instance.

    Returns:
        ErrorExplanation with detailed error analysis.

    Raises:
        HTTPException: If error analysis fails or other errors occur.

    """
    try:
        problem = await validate_problem_exists(
            course_id, topic_id, problem_id, loader,
        )
        await validate_code_submission(submission.code, submission.language)

        # First try compilation
        compile_result = await execution_service.compile_code(
            code=submission.code,
            language=submission.language,
        )

        if not compile_result["success"]:
            error = compile_result["error"]
        else:
            # If compilation succeeds, try execution
            visible_test_cases = problem.get("visible_test_cases", [])
            test_case = visible_test_cases[0] if visible_test_cases else {}
            exec_result = await execution_service.execute_code(
                code=submission.code,
                language=submission.language,
                test_cases=[test_case],
            )
            error = next((r["error"] for r in exec_result if r.get("error")), None)

        if not error:
            raise_bad_request("No errors found in the code")

        explanation = await llm.explain_errors(
            error=error,
            code=submission.code,
            problem=problem["description"],
        )

        return ErrorExplanation(
            error_type=explanation.get("error_type", "Runtime"),
            explanation=explanation.get("explanation", "No explanation available"),
            suggested_fixes=explanation.get("suggested_fixes", []),
            original_error=error,
            relevant_line=explanation.get("line"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error explanation failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to explain errors",
        ) from e


@router.post(
    "/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/optimize",
    responses={
        400: {"description": "Invalid input or code errors"},
        404: {"description": "Problem not found"},
        500: {"description": "Failed to generate optimizations"},
    },
    response_model=OptimizationSuggestion,
)
async def suggest_optimizations(
    course_id: Annotated[str, Path(..., description="ID of the course", min_length=1)],
    topic_id: Annotated[str, Path(..., description="ID of the topic", min_length=1)],
    problem_id: Annotated[str, Path(..., description="ID of the problem", min_length=1)],
    submission: CodeSubmission,
    execution_service: Annotated[CodeExecutionService, Depends(get_execution_service)],
    llm: Annotated[LLMService, Depends(get_llm_service)],
    loader: Annotated[CourseLoader, Depends(get_course_loader)],
) -> OptimizationSuggestion:
    """Suggest improvements for working or nearly-working code.

    Args:
        course_id: The course identifier.
        topic_id: The topic identifier.
        problem_id: The problem identifier.
        submission: Code submission to optimize.
        execution_service: Code execution service instance.
        llm: LLM service for optimization analysis.
        loader: Course loader service instance.

    Returns:
        OptimizationSuggestion with optimization recommendations.

    Raises:
        HTTPException: If optimization analysis fails or other errors occur.

    """
    try:
        problem = await validate_problem_exists(
            course_id, topic_id, problem_id, loader,
        )
        await validate_code_submission(submission.code, submission.language)

        # First try compilation
        compile_result = await execution_service.compile_code(
            code=submission.code,
            language=submission.language,
        )

        if not compile_result["success"]:
            error_msg = compile_result.get("error", "Unknown error")
            raise_bad_request(f"Code has compilation errors: {error_msg}")

        # Basic execution check
        visible_test_cases = problem.get("visible_test_cases", [])
        test_case = visible_test_cases[0] if visible_test_cases else {}
        exec_result = await execution_service.execute_code(
            code=submission.code,
            language=submission.language,
            test_cases=[test_case],
        )

        if not exec_result or not exec_result[0].get("passed", False):
            error = "Execution failed"
            if exec_result:
                error = exec_result[0].get("error", "Test case failed")
            raise_bad_request(f"Code fails basic test cases: {error}")

        # Get optimization suggestions
        optimization = await llm.analyze_optimizations(
            code=submission.code,
            problem=problem["description"],
            language=submission.language,
        )

        default_complexity = {"time": "Unknown", "space": "Unknown"}
        current_complexity = optimization.get("current_complexity", default_complexity)
        suggested_complexity = optimization.get("suggested_complexity", default_complexity)

        return OptimizationSuggestion(
            current_complexity=Complexity(**current_complexity),
            suggested_complexity=Complexity(**suggested_complexity),
            optimization_suggestions=optimization.get("optimization_suggestions", []),
            readability_suggestions=optimization.get("readability_suggestions", []),
            best_practice_suggestions=optimization.get("best_practice_suggestions", []),
            edge_cases=optimization.get("edge_cases", []),
            explanation=optimization.get("explanation", "No optimization suggestions available"),
            code_snippet=optimization.get("code_snippet", ""),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Optimization suggestion failed")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while generating optimizations",
        ) from e


@router.get(
    "/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/steps",
    responses={
        404: {"description": "Problem not found"},
        500: {"description": "Failed to generate steps"},
    },
    response_model=ConceptualSteps,
)
async def get_conceptual_steps(
    course_id: Annotated[str, Path(...)],
    topic_id: Annotated[str, Path(...)],
    problem_id: Annotated[str, Path(...)],
    llm: Annotated[LLMService, Depends(get_llm_service)],
    loader: Annotated[CourseLoader, Depends(get_course_loader)],
) -> ConceptualSteps:
    """Break down problem into smaller conceptual steps.

    Args:
        course_id: The course identifier.
        topic_id: The topic identifier.
        problem_id: The problem identifier.
        llm: LLM service for conceptual analysis.
        loader: Course loader service instance.

    Returns:
        ConceptualSteps with problem-solving breakdown.

    Raises:
        HTTPException: If step generation fails or other errors occur.

    """
    try:
        problem = await validate_problem_exists(course_id, topic_id, problem_id, loader)

        steps = await llm.generate_conceptual_steps(
            problem=problem["description"],
        )

        return ConceptualSteps(
            steps=steps,
            current_step=None,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to generate steps")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate conceptual steps",
        ) from e


@router.post(
    "/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/pseudocode",
    responses={
        400: {"description": "Invalid input"},
        404: {"description": "Problem not found"},
        500: {"description": "Failed to generate pseudocode"},
    },
    response_model=PseudocodeResponse,
)
async def generate_pseudocode(
    course_id: Annotated[str, Path(..., description="ID of the course", min_length=1)],
    topic_id: Annotated[str, Path(..., description="ID of the topic", min_length=1)],
    problem_id: Annotated[str, Path(..., description="ID of the problem", min_length=1)],
    submission: CodeSubmission,
    llm: Annotated[LLMService, Depends(get_llm_service)],
    loader: Annotated[CourseLoader, Depends(get_course_loader)],
) -> PseudocodeResponse:
    """Help translate student logic into structured pseudocode.

    Args:
        course_id: The course identifier.
        topic_id: The topic identifier.
        problem_id: The problem identifier.
        submission: Code submission to generate pseudocode for.
        llm: LLM service for pseudocode generation.
        loader: Course loader service instance.

    Returns:
        PseudocodeResponse with pseudocode and explanation.

    Raises:
        HTTPException: If pseudocode generation fails or other errors occur.

    """
    try:
        problem = await validate_problem_exists(
            course_id, topic_id, problem_id, loader,
        )
        await validate_code_submission(submission.code, submission.language)

        pseudocode = await llm.generate_pseudocode(
            code=submission.code,
            problem=problem["description"],
        )

        return PseudocodeResponse(
            pseudocode=pseudocode.get("pseudocode", ""),
            explanation=pseudocode.get("explanation", ""),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Pseudocode failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate pseudocode",
        ) from e


@router.post(
    "/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/submit",
    response_model=SubmissionResponse,
)
async def submit_problem(
    course_id: Annotated[str, Path(..., description="ID of the course", min_length=1)],
    topic_id: Annotated[str, Path(..., description="ID of the topic", min_length=1)],
    problem_id: Annotated[str, Path(..., description="ID of the problem", min_length=1)],
    submission: SubmissionRequest,
    execution_service: Annotated[CodeExecutionService, Depends(get_execution_service)],
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
    loader: Annotated[CourseLoader, Depends(get_course_loader)],
) -> SubmissionResponse:
    """Submit problem solution for grading.

    Args:
        course_id: The course identifier.
        topic_id: The topic identifier.
        problem_id: The problem identifier.
        submission: User's code submission.
        execution_service: Code execution service instance.
        llm_service: LLM service for analysis.
        loader: Course loader service instance.

    Returns:
        SubmissionResponse with submission status and results.

    Raises:
        HTTPException: If submission processing fails or other errors occur.

    """
    submission_id = str(uuid4())
    timestamp = datetime.now(UTC)

    # Initialize storage
    submission_data_init = {
        "submission_id": submission_id,
        "user_id": submission.user_id,
        "problem_id": problem_id,
        "course_id": course_id,
        "topic_id": topic_id,
        "status": "processing",  # Start as processing
        "timestamp": timestamp.isoformat(),  # Store as ISO string
        "code": submission.code,
        "language": submission.language,
        "result": None,
        "updated_at": timestamp.isoformat(),  # Store as ISO string
    }

    await SubmissionStorage.create(submission_id, submission_data_init)

    try:
        problem = await loader.get_problem(course_id,
                                                  topic_id, problem_id)
        if not problem:
            raise_not_found("Problem not found")

        result = await grade_submission_workflow(
            submission_id=submission_id,
            code=submission.code,
            language=submission.language,
            problem=problem,
            execution_service=execution_service,
            llm_service=llm_service,
        )

        final_status = "completed" if result.get("passed") else "failed"
        await SubmissionStorage.update(submission_id, {
            "status": final_status,
            "result": result,
            "updated_at": datetime.now(UTC).isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
        })

        return SubmissionResponse(
            submission_id=submission_id,
            status=final_status,
            timestamp=timestamp,
            result=result,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Grading failed")
        await SubmissionStorage.update(submission_id, {
            "status": "failed",
            "result": {"error": str(e)},
            "updated_at": datetime.now(UTC).isoformat(),
        })
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Grading process failed",
        ) from e
