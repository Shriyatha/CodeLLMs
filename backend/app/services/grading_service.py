"""Grading service for code submissions using Prefect workflows."""

from prefect import flow, task, get_run_logger, context
from prefect.tasks import task_input_hash
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
import asyncio
from app.services.llm_service import LLMService
from app.services.code_execution import CodeExecutionService
from app.services.mlflow_logger import MLFlowLogger
import mlflow

# Global storage for submission results with async-safe access
submissions_storage = {}
storage_lock = asyncio.Lock()
storage_condition = asyncio.Condition()

class SubmissionStorage:
    @staticmethod
    async def create(submission_id: str, data: Dict[str, Any]) -> None:
        async with storage_condition:
            async with storage_lock:
                submissions_storage[submission_id] = data
            storage_condition.notify_all()

    @staticmethod
    async def update(submission_id: str, updates: Dict[str, Any]) -> bool:
        async with storage_condition:
            async with storage_lock:
                if submission_id in submissions_storage:
                    submissions_storage[submission_id].update(updates)
                    storage_condition.notify_all()
                    return True
                return False

    @staticmethod
    async def get(submission_id: str) -> Optional[Dict[str, Any]]:
        async with storage_lock:
            return submissions_storage.get(submission_id)

    @staticmethod
    async def wait_for_result(submission_id: str, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Wait for a submission result with timeout"""
        async with storage_condition:
            # Check if already completed
            async with storage_lock:
                submission = submissions_storage.get(submission_id)
                if submission and submission.get("status") in ["completed", "failed"]:
                    return submission

            # Wait for completion
            try:
                await asyncio.wait_for(
                    storage_condition.wait_for(
                        lambda: submissions_storage.get(submission_id, {}).get("status") in ["completed", "failed"]
                    ),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                return None

            async with storage_lock:
                return submissions_storage.get(submission_id)

@task(name="validate_submission", retries=2, retry_delay_seconds=5)
async def validate_submission(code: str, language: str) -> bool:
    """Validate code submission"""
    logger = get_run_logger()
    mlflow_logger = None
    if context.get_run_context():
        mlflow_logger = MLFlowLogger()

    async def _safe_log(log_func, *args, **kwargs):
        if mlflow_logger and mlflow_logger.active_run():
            try:
                await log_func(*args, **kwargs)
            except Exception as e:
                logger.error(f"MLflow logging failed: {e}")

    try:
        if not code.strip():
            raise ValueError("Empty code submission")
        if language.lower() not in ["python", "javascript"]:
            raise ValueError(f"Unsupported language: {language}")
        if len(code) > 10000:
            raise ValueError("Code too long (max 10,000 characters)")

        await _safe_log(mlflow_logger.log_metric, "submission_length", float(len(code)))
        await _safe_log(mlflow_logger.log_param, "language", language)
        return True
    except Exception as e:
        await _safe_log(mlflow_logger.log_metric, "validation_failed", 1.0)
        await _safe_log(mlflow_logger.log_param, "validation_error", str(e))
        raise

@task(name="execute_tests", timeout_seconds=30)
async def execute_tests(
    code: str,
    language: str,
    test_cases: List[Dict[str, Any]],
    problem_id: str,
    execution_service: CodeExecutionService
) -> List[Dict[str, Any]]:
    """Execute code against test cases"""
    logger = get_run_logger()
    mlflow_logger = None
    if context.get_run_context():
        mlflow_logger = MLFlowLogger()

    async def _safe_log(log_func, *args, **kwargs):
        if mlflow_logger and mlflow_logger.active_run():
            try:
                await log_func(*args, **kwargs)
            except Exception as e:
                logger.error(f"MLflow logging failed: {e}")

    if not test_cases:
        raise ValueError("No test cases provided")

    try:
        await _safe_log(mlflow_logger.log_param, "problem_id", problem_id)
        await _safe_log(mlflow_logger.log_param, "test_case_count", len(test_cases))

        test_results = await execution_service.execute_code(
            code=code,
            language=language,
            test_cases=test_cases
        )

        passed_count = sum(1 for r in test_results if r.get('passed', False))
        await _safe_log(mlflow_logger.log_metric, "tests_passed", float(passed_count))
        await _safe_log(mlflow_logger.log_metric, "tests_failed", float(len(test_results) - passed_count))

        return test_results
    except Exception as e:
        await _safe_log(mlflow_logger.log_metric, "execution_failed", 1.0)
        raise

@task(name="calculate_score")
async def calculate_score(
    test_results: List[Dict[str, Any]],
    visible_count: int
) -> Dict[str, Any]:
    """Calculate score based on test results"""
    logger = get_run_logger()
    mlflow_logger = None
    if context.get_run_context():
        mlflow_logger = MLFlowLogger()

    async def _safe_log(log_func, *args, **kwargs):
        if mlflow_logger and mlflow_logger.active_run():
            try:
                await log_func(*args, **kwargs)
            except Exception as e:
                logger.error(f"MLflow logging failed: {e}")

    if not test_results:
        raise ValueError("No test results provided")

    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results if r.get('passed', False))

    # Calculate score with different weights for visible and hidden tests
    hidden_passed = sum(1 for r in test_results[visible_count:] if r.get('passed', False))
    hidden_total = total_tests - visible_count

    if visible_count > 0:
        visible_weight = 0.7
        visible_score = visible_weight * (sum(1 for r in test_results[:visible_count] if r.get('passed', False)) / visible_count)
    else:
        visible_score = 0.0
        visible_weight = 0.0

    if hidden_total > 0:
        hidden_weight = 0.3
        hidden_score = hidden_weight * (hidden_passed / hidden_total)
    else:
        hidden_score = 0.0
        hidden_weight = 0.0

    score = visible_score + hidden_score

    await _safe_log(mlflow_logger.log_metric, "final_score", score * 100)

    return {
        "score": round(score * 100, 2),
        "passed": passed_tests == total_tests,
        "passed_tests": passed_tests,
        "total_tests": total_tests
    }

@task(name="generate_feedback", cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
async def generate_feedback(
    code: str,
    test_results: List[Dict[str, Any]],
    problem_description: str,
    llm_service: LLMService
) -> str:
    """Generate accurate feedback using LLM"""
    logger = get_run_logger()
    mlflow_logger = None
    if context.get_run_context():
        mlflow_logger = MLFlowLogger()

    async def _safe_log(log_func, *args, **kwargs):
        if mlflow_logger and mlflow_logger.active_run():
            try:
                await log_func(*args, **kwargs)
            except Exception as e:
                logger.error(f"MLflow logging failed: {e}")

    if not test_results:
        return "No test results available"

    try:
        # Count passed/failed tests
        passed_count = sum(1 for r in test_results if r.get('passed', False))
        total_tests = len(test_results)

        if passed_count == total_tests:
            return "All test cases passed! Great job!"

        # Get errors from failed tests
        errors = []
        for r in test_results:
            if not r.get('passed', False):
                error_info = {
                    'input': r.get('input'),
                    'expected': r.get('expected'),
                    'output': r.get('output'),
                    'error': r.get('error')
                }
                errors.append(error_info)

        if errors:
            # Generate detailed feedback for failed tests
            error_str = "\n".join(
                f"Test case {i+1}:\n"
                f"Input: {e['input']}\n"
                f"Expected: {e['expected']}\n"
                f"Got: {e['output']}\n"
                f"{'Error: ' + e['error'] if e['error'] else ''}"
                for i, e in enumerate(errors))

            # Assuming LLMService has a method to log with MLflow
            feedback_result = await llm_service.explain_errors(
                error=error_str,
                code=code,
                problem=problem_description,
                mlflow_logger=mlflow_logger  # Pass the logger
            )
            return feedback_result.get('explanation',
                f"Failed {len(errors)}/{total_tests} test cases. Please review your code.")

        return f"Passed {passed_count}/{total_tests} test cases."

    except Exception as e:
        logger.error(f"Feedback generation failed: {str(e)}")
        return "Unable to generate detailed feedback due to an error."

@flow(name="grade_submission_workflow")
async def grade_submission_workflow(
    submission_id: str,
    code: str,
    language: str,
    problem: Dict[str, Any],
    execution_service: CodeExecutionService,
    llm_service: LLMService
) -> Dict[str, Any]:
    """Execute grading with accurate results"""
    logger = get_run_logger()
    mlflow_logger = None
    if context.get_run_context():
        mlflow_logger = MLFlowLogger()

    try:
        if mlflow_logger:
            mlflow_logger.start_run(
                run_name=f"submission_{submission_id}",
                nested=False
            )

        # Initialize storage
        await SubmissionStorage.create(submission_id, {
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat()
        })

        # Prepare test cases
        visible_test_cases = problem.get('visible_test_cases', [])
        hidden_test_cases = problem.get('hidden_test_cases', [])
        test_cases = visible_test_cases + hidden_test_cases
        visible_count = len(visible_test_cases)
        problem_id = problem.get("id", "unknown")
        course_id = problem.get("course_id", "unknown")
        topic_id = problem.get("topic_id", "unknown")
        complexity = problem.get("complexity", "unknown")

        if mlflow_logger and mlflow_logger.active_run():
            mlflow_logger.log_param("submission_id", submission_id)
            mlflow_logger.log_param("problem_id", problem_id)
            mlflow_logger.log_param("course_id", course_id)
            mlflow_logger.log_param("topic_id", topic_id)
            mlflow_logger.log_param("complexity", complexity)

        # Validate submission
        await validate_submission(code, language)

        # Execute tests
        test_results = await execute_tests(
            code=code,
            language=language,
            test_cases=test_cases,
            problem_id=problem_id,
            execution_service=execution_service
        )

        # Calculate score
        score_result = await calculate_score(
            test_results=test_results,
            visible_count=visible_count
        )

        # Generate accurate feedback
        feedback = await generate_feedback(
            code=code,
            test_results=test_results,
            problem_description=problem.get('description', ''),
            llm_service=llm_service
        )

        # Determine overall status based on test results
        passed_all = all(r.get('passed', False) for r in test_results)
        status = "completed" if passed_all else "partially_completed"

        # Prepare final result
        result = {
            "passed": passed_all,
            "score": score_result["score"],
            "feedback": feedback,
            "execution_time": datetime.now(timezone.utc).isoformat(),
            "test_results": test_results,
            "problem_id": problem_id,
            "course": course_id,
            "topic": topic_id
        }

        # Store results
        await SubmissionStorage.update(submission_id, {
            "status": status,
            "result": result,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat()
        })

        if mlflow_logger and mlflow_logger.active_run():
            mlflow_logger.log_metric("final_score", score_result["score"])
            mlflow_logger.log_param("final_status", status)
            mlflow_logger.log_artifact(f"temp_submission_code_{submission_id}.{'py' if language.lower() == 'python' else 'js'}", artifact_path="submission_code")
            mlflow_logger.log_dict(result, f"submission_result_{submission_id}.json")

        return result

    except Exception as e:
        logger.error(f"Grading failed for {submission_id}: {str(e)}", exc_info=True)

        error_result = {
            "error": str(e),
            "passed": False,
            "score": 0,
            "feedback": f"Grading failed: {str(e)}",
            "test_results": [],
            "problem_id": problem.get("id", "unknown")
        }

        await SubmissionStorage.update(submission_id, {
            "status": "failed",
            "result": error_result,
            "error": str(e)
        })

        return error_result
    finally:
        if mlflow_logger and mlflow_logger.active_run():
            mlflow_logger.end_run()