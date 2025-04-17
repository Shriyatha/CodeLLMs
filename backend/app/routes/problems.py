from fastapi import APIRouter, HTTPException, Path, status, Body, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from fastapi.params import Query
from app.services.llm_service import LLMService
from app.services.code_execution import CodeExecutionService
from app.services.course_loader import CourseLoader
import logging
from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from datetime import timedelta
from typing import Dict, Union
from uuid import uuid4
import re
from threading import Lock
import asyncio
from app.services.grading_service import  SubmissionStorage, grade_submission_workflow

router = APIRouter(prefix="", tags=["problems"])
logger = logging.getLogger(__name__)


# Models
class SubmissionRequest(BaseModel):
    code: str = Field(..., min_length=1)
    language: str = Field("python", pattern="^(python|javascript)$")
    user_id: str = Field(...)

class SubmissionResponse(BaseModel):
    submission_id: str = Field(...)
    status: str = Field(...)
    timestamp: datetime = Field(...)
    result: Optional[Dict[str, Any]] = Field(None)

class GradingResult(BaseModel):
    passed: bool
    score: float = Field(..., ge=0, le=100)
    feedback: str
    execution_time: str
    test_results: List[Dict[str, Any]]

class CodeSubmission(BaseModel):
    code: str = Field(..., min_length=1)
    language: str = Field("python", pattern="^(python|javascript)$")

class ProblemResponse(BaseModel):
    id: str
    title: str
    description: str
    complexity: str = Field(..., pattern="^(easy|medium|hard)$")
    starter_code: str
    visible_test_cases: List[Dict[str, Any]]

class TopicResponse(BaseModel):
    id: str
    title: str
    description: str
    problems: List[Dict[str, Any]]

class HintResponse(BaseModel):
    hint_level: int = Field(..., ge=0)
    hint_text: str
    max_level: int = Field(..., ge=1)

class ErrorExplanation(BaseModel):
    error_type: str
    explanation: str
    suggested_fixes: List[str]
    original_error: str
    relevant_line: Optional[int] = Field(None, ge=1)

class Complexity(BaseModel):
    time: str
    space: str

class OptimizationSuggestion(BaseModel):
    current_complexity: Complexity
    suggested_complexity: Complexity
    optimization_suggestions: List[str]
    readability_suggestions: List[str]
    best_practice_suggestions: List[str]
    edge_cases: List[str]
    explanation: str
    code_snippet: str

class ConceptualSteps(BaseModel):
    steps: List[str]
    current_step: Optional[int] = Field(None, ge=0)

class DebuggingSuggestion(BaseModel):
    strategy: str
    variables_to_track: List[str]

class ExecutionResult(BaseModel):
    visible_results: List[Dict[str, Any]]
    hidden_passed: bool
    execution_time: str

class PseudocodeResponse(BaseModel):
    pseudocode: str
    explanation: str

# Dependency injections
def get_execution_service():
    try:
        return CodeExecutionService()
    except Exception as e:
        logger.error(f"Failed to initialize CodeExecutionService: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Code execution service unavailable"
        )

def get_course_loader():
    try:
        return CourseLoader()
    except Exception as e:
        logger.error(f"Failed to initialize CourseLoader: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Course loading service unavailable"
        )

def get_llm_service():
    try:
        return LLMService()
    except Exception as e:
        logger.error(f"Failed to initialize LLMService: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service unavailable"
        )

# Helper functions
async def validate_problem_exists(course_id: str, topic_id: str, problem_id: str, loader: CourseLoader):
    if not re.match(r'^[a-zA-Z0-9_-]+$', course_id):
        raise HTTPException(status_code=400, detail="Invalid course ID format")
    if not re.match(r'^[a-zA-Z0-9_-]+$', topic_id):
        raise HTTPException(status_code=400, detail="Invalid topic ID format")
    if not re.match(r'^[a-zA-Z0-9_-]+$', problem_id):
        raise HTTPException(status_code=400, detail="Invalid problem ID format")

    try:
        problem = await loader.get_problem(course_id, topic_id, problem_id)
        if not problem:
            logger.error(f"Problem not found: {course_id}/{topic_id}/{problem_id}")
            raise HTTPException(status_code=404, detail="Problem not found")
        return problem
    except Exception as e:
        logger.error(f"Error loading problem: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load problem"
        )

async def validate_code_submission(code: str, language: str = "python"):
    if not code.strip():
        raise HTTPException(status_code=400, detail="Empty code submission")
    if len(code) > 10000:
        raise HTTPException(status_code=400, detail="Code too long (max 10,000 characters)")
    if language not in ["python", "javascript"]:
        raise HTTPException(status_code=400, detail="Unsupported language")

# Endpoints
@router.get("/courses/{course_id}/topics/{topic_id}",
           response_model=TopicResponse,
           responses={
               404: {"description": "Topic not found"},
               400: {"description": "Invalid ID format"},
               500: {"description": "Internal server error"}
           })
async def get_topic_details(
    course_id: str = Path(..., description="ID of the course", min_length=1),
    topic_id: str = Path(..., description="ID of the topic", min_length=1),
    loader: CourseLoader = Depends(get_course_loader)
):
    """Get topic information including description and list of problems"""
    try:
        if not re.match(r'^[a-zA-Z0-9_-]+$', course_id):
            raise HTTPException(status_code=400, detail="Invalid course ID format")
        if not re.match(r'^[a-zA-Z0-9_-]+$', topic_id):
            raise HTTPException(status_code=400, detail="Invalid topic ID format")

        topic = await loader.get_topic(course_id, topic_id)
        if not topic:
            logger.error(f"Topic not found: {course_id}/{topic_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Topic {topic_id} not found"
            )
        
        return TopicResponse(
            id=topic_id,
            title=topic.get("title", ""),
            description=topic.get("description", ""),
            problems=topic.get("problems", [])
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting topic details: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/courses/{course_id}/topics/{topic_id}/problems/{problem_id}", 
           response_model=ProblemResponse,
           responses={
               404: {"description": "Problem not found"},
               400: {"description": "Invalid ID format"},
               500: {"description": "Internal server error"}
           })
async def get_problem_details(
    course_id: str = Path(..., description="ID of the course", min_length=1),
    topic_id: str = Path(..., description="ID of the topic", min_length=1),
    problem_id: str = Path(..., description="ID of the problem", min_length=1),
    loader: CourseLoader = Depends(get_course_loader)
):
    """Get detailed problem information including starter code and test cases"""
    try:
        problem = await validate_problem_exists(course_id, topic_id, problem_id, loader)
        
        return ProblemResponse(
            id=problem_id,
            title=problem.get("title", ""),
            description=problem.get("description", ""),
            complexity=problem.get("complexity", "medium"),
            starter_code=problem.get("starter_code", ""),
            visible_test_cases=problem.get("visible_test_cases", [])
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting problem details: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/execute",
            response_model=ExecutionResult,
            responses={
                400: {"description": "Invalid input or no test cases"},
                404: {"description": "Problem not found"},
                500: {"description": "Execution failed"}
            })
async def execute_problem(
    course_id: str = Path(...),
    topic_id: str = Path(...),
    problem_id: str = Path(...),
    submission: CodeSubmission = Body(...),
    execution_service: CodeExecutionService = Depends(get_execution_service),
    loader: CourseLoader = Depends(get_course_loader)
):
    """Execute code against test cases in secure Docker container"""
    try:
        await validate_code_submission(submission.code, submission.language)
        problem = await validate_problem_exists(course_id, topic_id, problem_id, loader)

        test_cases = problem.get('visible_test_cases', []) + problem.get('hidden_test_cases', [])
        if not test_cases:
            logger.error(f"No test cases found for problem: {problem_id}")
            raise HTTPException(status_code=400, detail="No test cases provided")

        results = await execution_service.execute_code(
            code=submission.code,
            language=submission.language,
            test_cases=test_cases
        )

        if not isinstance(results, list):
            logger.error(f"Unexpected results format: {type(results)}")
            raise HTTPException(status_code=500, detail="Unexpected results format")

        visible_count = len(problem.get('visible_test_cases', []))
        visible_results = results[:visible_count]

        return ExecutionResult(
            visible_results=visible_results,
            hidden_passed=all(r.get('passed', False) for r in results[visible_count:]),
            execution_time=datetime.now(timezone.utc).isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Code execution failed"
        )

@router.post("/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/hints",
            response_model=HintResponse,
            responses={
                400: {"description": "Invalid input"},
                404: {"description": "Problem not found"},
                500: {"description": "Failed to generate hints"}
            })
async def get_progressive_hints(
    course_id: str = Path(...),
    topic_id: str = Path(...),
    problem_id: str = Path(...),
    submission: CodeSubmission = Body(...),
    llm: LLMService = Depends(get_llm_service),
    loader: CourseLoader = Depends(get_course_loader)
):
    """Get tiered hints to guide student toward solution"""
    try:
        problem = await validate_problem_exists(course_id, topic_id, problem_id, loader)
        await validate_code_submission(submission.code, submission.language)
        
        hints, max_level = await llm.get_progressive_hints(
            problem=problem['description'],
            code=submission.code,
            current_level=0
        )
        
        return HintResponse(
            hint_level=0,
            hint_text=hints[0] if hints else "No hints available",
            max_level=max_level
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hint generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate hints"
        )

@router.post("/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/explain",
            response_model=ErrorExplanation,
            responses={
                400: {"description": "Invalid input or no errors"},
                404: {"description": "Problem not found"},
                500: {"description": "Failed to explain errors"}
            })
async def explain_errors(
    course_id: str = Path(...),
    topic_id: str = Path(...),
    problem_id: str = Path(...),
    submission: CodeSubmission = Body(...),
    execution_service: CodeExecutionService = Depends(get_execution_service),
    llm: LLMService = Depends(get_llm_service),
    loader: CourseLoader = Depends(get_course_loader)
):
    """Explain errors in user's code with AI assistance"""
    try:
        problem = await validate_problem_exists(course_id, topic_id, problem_id, loader)
        await validate_code_submission(submission.code, submission.language)

        # First try compilation
        compile_result = await execution_service.compile_code(
            code=submission.code,
            language=submission.language
        )
        
        if not compile_result['success']:
            error = compile_result['error']
        else:
            # If compilation succeeds, try execution
            test_case = problem.get('visible_test_cases', [{}])[0] if problem.get('visible_test_cases') else {}
            exec_result = await execution_service.execute_code(
                code=submission.code,
                language=submission.language,
                test_cases=[test_case]
            )
            error = next((r['error'] for r in exec_result if r.get('error')), None)
        
        if not error:
            raise HTTPException(status_code=400, detail="No errors found in the code")
        
        explanation = await llm.explain_errors(
            error=error,
            code=submission.code,
            problem=problem['description']
        )
        
        return ErrorExplanation(
            error_type=explanation.get('error_type', 'Runtime'),
            explanation=explanation.get('explanation', 'No explanation available'),
            suggested_fixes=explanation.get('suggested_fixes', []),
            original_error=error,
            relevant_line=explanation.get('line')
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explanation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to explain errors"
        )

@router.post("/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/optimize",
            response_model=OptimizationSuggestion,
            responses={
                400: {"description": "Invalid input or code errors"},
                404: {"description": "Problem not found"},
                500: {"description": "Failed to generate optimizations"}
            })
async def suggest_optimizations(
    course_id: str = Path(...),
    topic_id: str = Path(...),
    problem_id: str = Path(...),
    submission: CodeSubmission = Body(...),
    execution_service: CodeExecutionService = Depends(get_execution_service),
    llm: LLMService = Depends(get_llm_service),
    loader: CourseLoader = Depends(get_course_loader)
):
    """Suggest improvements for working or nearly-working code"""
    try:
        problem = await validate_problem_exists(course_id, topic_id, problem_id, loader)
        await validate_code_submission(submission.code, submission.language)

        # First try compilation
        compile_result = await execution_service.compile_code(
            code=submission.code,
            language=submission.language
        )
        
        if not compile_result['success']:
            raise HTTPException(
                status_code=400,
                detail=f"Code has compilation errors: {compile_result.get('error', 'Unknown error')}"
            )
        
        # Basic execution check
        test_case = problem.get('visible_test_cases', [{}])[0] if problem.get('visible_test_cases') else {}
        exec_result = await execution_service.execute_code(
            code=submission.code,
            language=submission.language,
            test_cases=[test_case]
        )
        
        if not exec_result or not exec_result[0].get('passed', False):
            error = exec_result[0].get('error', 'Test case failed') if exec_result else 'Execution failed'
            raise HTTPException(
                status_code=400,
                detail=f"Code fails basic test cases: {error}"
            )

        # Get optimization suggestions
        optimization = await llm.analyze_optimizations(
            code=submission.code,
            problem=problem['description'],
            language=submission.language
        )

        return OptimizationSuggestion(
            current_complexity=Complexity(**optimization.get('current_complexity', {"time": "Unknown", "space": "Unknown"})),
            suggested_complexity=Complexity(**optimization.get('suggested_complexity', {"time": "Unknown", "space": "Unknown"})),
            optimization_suggestions=optimization.get('optimization_suggestions', []),
            readability_suggestions=optimization.get('readability_suggestions', []),
            best_practice_suggestions=optimization.get('best_practice_suggestions', []),
            edge_cases=optimization.get('edge_cases', []),
            explanation=optimization.get('explanation', 'No optimization suggestions available'),
            code_snippet=optimization.get('code_snippet', '')
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization suggestion failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error while generating optimizations"
        )

@router.get("/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/steps",
           response_model=ConceptualSteps,
           responses={
               404: {"description": "Problem not found"},
               500: {"description": "Failed to generate steps"}
           })
async def get_conceptual_steps(
    course_id: str = Path(...),
    topic_id: str = Path(...),
    problem_id: str = Path(...),
    llm: LLMService = Depends(get_llm_service),
    loader: CourseLoader = Depends(get_course_loader)
):
    """Break down problem into smaller conceptual steps"""
    try:
        problem = await validate_problem_exists(course_id, topic_id, problem_id, loader)
        
        steps = await llm.generate_conceptual_steps(
            problem=problem['description']
        )
        
        return ConceptualSteps(
            steps=steps,
            current_step=None
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate steps: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate conceptual steps"
        )

@router.post("/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/debug",
            response_model=DebuggingSuggestion,
            responses={
                400: {"description": "Invalid input"},
                404: {"description": "Problem not found"},
                500: {"description": "Failed to generate debugging tips"}
            })
async def get_debugging_tips(
    course_id: str = Path(...),
    topic_id: str = Path(...),
    problem_id: str = Path(...),
    submission: CodeSubmission = Body(...),
    llm: LLMService = Depends(get_llm_service),
    loader: CourseLoader = Depends(get_course_loader)
):
    """Provide debugging guidance without revealing solution"""
    try:
        problem = await validate_problem_exists(course_id, topic_id, problem_id, loader)
        await validate_code_submission(submission.code, submission.language)
        
        debugging = await llm.suggest_debugging(
            code=submission.code,
            problem=problem['description']
        )
        
        return DebuggingSuggestion(
            strategy=debugging.get('strategy', 'Add print statements to track values'),
            variables_to_track=debugging.get('variables_to_track', [])
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Debugging failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate debugging tips"
        )

@router.post("/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/pseudocode",
            response_model=PseudocodeResponse,
            responses={
                400: {"description": "Invalid input"},
                404: {"description": "Problem not found"},
                500: {"description": "Failed to generate pseudocode"}
            })
async def generate_pseudocode(
    course_id: str = Path(...),
    topic_id: str = Path(...),
    problem_id: str = Path(...),
    submission: CodeSubmission = Body(...),
    llm: LLMService = Depends(get_llm_service),
    loader: CourseLoader = Depends(get_course_loader)
):
    """Help translate student logic into structured pseudocode"""
    try:
        problem = await validate_problem_exists(course_id, topic_id, problem_id, loader)
        await validate_code_submission(submission.code, submission.language)
        
        pseudocode = await llm.generate_pseudocode(
            code=submission.code,
            problem=problem['description']
        )
        
        return PseudocodeResponse(
            pseudocode=pseudocode.get('pseudocode', ''),
            explanation=pseudocode.get('explanation', '')
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pseudocode failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate pseudocode"
        )


async def run_flow_in_background(flow_func, *args, **kwargs):
    """Helper function to properly run a flow in the background"""
    try:
        await flow_func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Flow execution failed: {str(e)}", exc_info=True)

@router.post("/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/submit",
            response_model=SubmissionResponse)
async def submit_problem(
    background_tasks: BackgroundTasks,
    course_id: str = Path(...),
    topic_id: str = Path(...),
    problem_id: str = Path(...),
    submission: SubmissionRequest = Body(...),
    execution_service: CodeExecutionService = Depends(get_execution_service),
    llm_service: LLMService = Depends(get_llm_service),
    loader: CourseLoader = Depends(get_course_loader)
):
    """Submission endpoint that waits for completion"""
    submission_id = str(uuid4())
    timestamp = datetime.now(timezone.utc)

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
        "updated_at": timestamp.isoformat()  # Store as ISO string
    }
    await SubmissionStorage.create(submission_id, submission_data_init)

    try:
        problem = await loader.get_problem(course_id, topic_id, problem_id)
        if not problem:
            raise HTTPException(status_code=404, detail="Problem not found")

        #grading_service = GradingService(execution_service, llm_service)
        result = await grade_submission_workflow(
            submission_id=submission_id,
            code=submission.code,
            language=submission.language,
            problem=problem,
            execution_service=execution_service,
            llm_service=llm_service
        )

        final_status = "completed" if result.get("passed") else "failed"
        await SubmissionStorage.update(submission_id, {
            "status": final_status,
            "result": result,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat()
        })

        return SubmissionResponse(
            submission_id=submission_id,
            status=final_status,
            timestamp=timestamp,
            result=result
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Grading failed: {str(e)}")
        await SubmissionStorage.update(submission_id, {
            "status": "failed",
            "result": {"error": str(e)},
            "updated_at": datetime.now(timezone.utc).isoformat()
        })
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Grading process failed"
        )