"""Grading service for code submissions using Prefect workflows."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from prefect import context, flow, get_run_logger, task
from prefect.tasks import task_input_hash

from app.services.code_execution import CodeExecutionService
from app.services.llm_service import LLMService
from app.services.mlflow_logger import MLFlowLogger

# Global storage for submission results with async-safe access
submissions_storage = {}
storage_lock = asyncio.Lock()
storage_condition = asyncio.Condition()

# Maximum length for code submissions
MAX_CODE_LENGTH = 10000


class SubmissionStorage:
    """Storage class for managing submission data with async capabilities."""

    @staticmethod
    async def create(submission_id: str, data: dict[str, Any]) -> None:
        """Create a new submission entry in storage.

        Args:
            submission_id: Unique identifier for the submission
            data: Initial data for the submission

        """
        async with storage_condition:
            async with storage_lock:
                submissions_storage[submission_id] = data
            storage_condition.notify_all()

    @staticmethod
    async def update(submission_id: str, updates: dict[str, Any]) -> bool:
        """Update an existing submission with new data.

        Args:
            submission_id: Unique identifier for the submission
            updates: Data updates to apply

        Returns:
            bool: True if update was successful, False if submission not found

        """
        async with storage_condition, storage_lock:
            if submission_id in submissions_storage:
                submissions_storage[submission_id].update(updates)
                storage_condition.notify_all()
                return True
            return False

    @staticmethod
    async def get(submission_id: str) -> dict[str, Any] | None:
        """Retrieve a submission by ID.

        Args:
            submission_id: Unique identifier for the submission

        Returns:
            Submission data or None if not found

        """
        async with storage_lock:
            return submissions_storage.get(submission_id)

    @staticmethod
    async def wait_for_result(submission_id: str, timeout: float = 30.0) -> dict[str, Any] | None:
        """Wait for a submission result with timeout.

        Args:
            submission_id: Unique identifier for the submission
            timeout: Maximum time to wait in seconds

        Returns:
            Submission data or None if timeout occurred

        """
        try:
            async with asyncio.timeout(timeout):
                async with storage_condition:
                    # Check if already completed
                    async with storage_lock:
                        submission = submissions_storage.get(submission_id)
                        if submission and submission.get("status") in ["completed", "failed"]:
                            return submission

                    # Wait for completion
                    completion_states = ["completed", "failed"]
                    await storage_condition.wait_for(
                        lambda: submissions_storage.get(submission_id, {}).get("status") in completion_states,
                    )
                    async with storage_lock:
                        return submissions_storage.get(submission_id)
        except TimeoutError:
            return None


@task(name="validate_submission", retries=2, retry_delay_seconds=5)
async def validate_submission(code: str, language: str) -> bool:
    """Validate code submission.

    Args:
        code: Source code to validate
        language: Programming language of the submission

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails

    """
    logger = get_run_logger()
    mlflow_logger = None
    if context.get_run_context():
        mlflow_logger = MLFlowLogger()

    async def _safe_log(log_func: Callable[..., None], *args: dict[str, Any], **kwargs: dict[str, Any]) -> None:
        if mlflow_logger and mlflow_logger.active_run():
            try:
                await log_func(*args, **kwargs)
            except Exception:
                logger.exception("MLflow logging failed")

    try:
        await _safe_log(mlflow_logger.log_metric, "submission_length", float(len(code)))
        await _safe_log(mlflow_logger.log_param, "language", language)
    except Exception as e:
        await _safe_log(mlflow_logger.log_metric, "validation_failed", 1.0)
        await _safe_log(mlflow_logger.log_param, "validation_error", str(e))
        raise
    else:
        return True


@task(name="execute_tests", timeout_seconds=30)
async def execute_tests(
    code: str,
    language: str,
    test_cases: list[dict[str, Any]],
    problem_id: str,
    execution_service: CodeExecutionService,
) -> list[dict[str, Any]]:
    """Execute code against test cases.

    Args:
        code: Source code to test
        language: Programming language of the code
        test_cases: List of test cases to execute
        problem_id: Identifier for the problem being tested
        execution_service: Service to execute code

    Returns:
        List of test results

    """
    logger = get_run_logger()
    mlflow_logger = None
    if context.get_run_context():
        mlflow_logger = MLFlowLogger()

    async def _safe_log(log_func: Callable[..., None], *args: dict[str, Any], **kwargs: dict[str, Any]) -> None:
        if mlflow_logger and mlflow_logger.active_run():
            try:
                await log_func(*args, **kwargs)
            except Exception:
                logger.exception("MLflow logging failed")

    if not test_cases:
        msg = "No test cases provided"
        raise ValueError(msg)

    try:
        await _safe_log(mlflow_logger.log_param, "problem_id", problem_id)
        await _safe_log(mlflow_logger.log_param, "test_case_count", len(test_cases))

        test_results = await execution_service.execute_code(
            code=code,
            language=language,
            test_cases=test_cases,
        )

        passed_count = sum(1 for r in test_results if r.get("passed", False))
        await _safe_log(mlflow_logger.log_metric, "tests_passed", float(passed_count))
        tests_failed = float(len(test_results) - passed_count)
        await _safe_log(mlflow_logger.log_metric, "tests_failed", tests_failed)
    except Exception:
        await _safe_log(mlflow_logger.log_metric, "execution_failed", 1.0)
        raise
    else:
        return test_results


@task(name="calculate_score")
async def calculate_score(
    test_results: list[dict[str, Any]],
    visible_count: int,
) -> dict[str, Any]:
    """Calculate score based on test results.

    Args:
        test_results: Results from test execution
        visible_count: Number of visible test cases

    Returns:
        Dictionary with score details

    """
    logger = get_run_logger()
    mlflow_logger = None
    if context.get_run_context():
        mlflow_logger = MLFlowLogger()

    async def _safe_log(log_func: Callable[..., None], *args: dict[str, Any], **kwargs: dict[str, Any]) -> None:
        if mlflow_logger and mlflow_logger.active_run():
            try:
                await log_func(*args, **kwargs)
            except Exception:
                logger.exception("MLflow logging failed")

    if not test_results:
        msg = "No test results provided"
        raise ValueError(msg)

    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results if r.get("passed", False))

    # Calculate score with different weights for visible and hidden tests
    hidden_passed = sum(1 for r in test_results[visible_count:] if r.get("passed", False))
    hidden_total = total_tests - visible_count

    if visible_count > 0:
        visible_weight = 0.7
        passed_visible = sum(1 for r in test_results[:visible_count] if r.get("passed", False))
        visible_score = visible_weight * (passed_visible / visible_count)
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
        "total_tests": total_tests,
    }


@task(name="generate_feedback", cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
async def generate_feedback(
    code: str,
    test_results: list[dict[str, Any]],
    problem_description: str,
    llm_service: LLMService,
) -> str:
    """Generate accurate feedback using LLM.

    Args:
        code: Source code to analyze
        test_results: Results from test execution
        problem_description: Description of the problem being solved
        llm_service: Service for leveraging LLM capabilities

    Returns:
        Feedback as a string

    """
    logger = get_run_logger()
    mlflow_logger = None
    if context.get_run_context():
        mlflow_logger = MLFlowLogger()

    async def _safe_log(log_func: Callable[..., None], *args: dict[str, Any], **kwargs: dict[str, Any]) -> None:
        if mlflow_logger and mlflow_logger.active_run():
            try:
                await log_func(*args, **kwargs)
            except (BaseException, Exception):
                logger.exception("MLflow logging failed")

    if not test_results:
        return "No test results available"

    try:
        # Count passed/failed tests
        passed_count = sum(1 for r in test_results if r.get("passed", False))
        total_tests = len(test_results)

        if passed_count == total_tests:
            return "All test cases passed! Great job!"

        # Get errors from failed tests
        errors = []
        for r in test_results:
            if not r.get("passed", False):
                error_info = {
                    "input": r.get("input"),
                    "expected": r.get("expected"),
                    "output": r.get("output"),
                    "error": r.get("error"),
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
            )
            return feedback_result.get("explanation",
                f"Failed {len(errors)}/{total_tests} test cases. Please review your code.")

    except (BaseException, Exception):
        return "Unable to generate detailed feedback due to an error."
    else:
        return f"Passed {passed_count}/{total_tests} test cases."


@flow(name="grade_submission_workflow")
async def grade_submission_workflow(
    submission_id: str,
    code: str,
    language: str,
    problem: dict[str, Any],
    execution_service: CodeExecutionService,
    llm_service: LLMService,
) -> dict[str, Any]:
    """Execute grading with accurate results.

    Coordinates the entire grading workflow from submission validation to feedback generation.

    Args:
        submission_id: Unique identifier for the submission
        code: Source code to grade
        language: Programming language of the code
        problem: Problem details including test cases
        execution_service: Service for executing code
        llm_service: Service for LLM-based feedback

    Returns:
        Grading results

    """
    mlflow_logger = None
    if context.get_run_context():
        mlflow_logger = MLFlowLogger()

    try:
        if mlflow_logger:
            mlflow_logger.start_run(
                run_name=f"submission_{submission_id}",
                nested=False,
            )

        # Initialize storage
        await SubmissionStorage.create(submission_id, {
            "status": "started",
            "started_at": datetime.now(UTC).isoformat(),
        })

        # Prepare test cases
        visible_test_cases = problem.get("visible_test_cases", [])
        hidden_test_cases = problem.get("hidden_test_cases", [])
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
            execution_service=execution_service,
        )

        # Calculate score
        score_result = await calculate_score(
            test_results=test_results,
            visible_count=visible_count,
        )

        # Generate accurate feedback
        feedback = await generate_feedback(
            code=code,
            test_results=test_results,
            problem_description=problem.get("description", ""),
            llm_service=llm_service,
        )

        # Determine overall status based on test results
        passed_all = all(r.get("passed", False) for r in test_results)
        status = "completed" if passed_all else "partially_completed"

        # Prepare final result
        result = {
            "passed": passed_all,
            "score": score_result["score"],
            "feedback": feedback,
            "execution_time": datetime.now(UTC).isoformat(),
            "test_results": test_results,
            "problem_id": problem_id,
            "course": course_id,
            "topic": topic_id,
        }

        # Store results
        await SubmissionStorage.update(submission_id, {
            "status": status,
            "result": result,
            "updated_at": datetime.now(UTC).isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
        })

        if mlflow_logger and mlflow_logger.active_run():
            mlflow_logger.log_metric("final_score", score_result["score"])
            mlflow_logger.log_param("final_status", status)
            extension = "py" if language.lower() == "python" else "js"
            filename = f"temp_submission_code_{submission_id}.{extension}"
            mlflow_logger.log_artifact(filename, artifact_path="submission_code")
            mlflow_logger.log_dict(result, f"submission_result_{submission_id}.json")

    except (Exception, BaseException, NameError) as e:
        error_result = {
            "error": str(e),
            "passed": False,
            "score": 0,
            "feedback": f"Grading failed: {e!s}",
            "test_results": [],
            "problem_id": problem.get("id", "unknown"),
        }

        await SubmissionStorage.update(submission_id, {
            "status": "failed",
            "result": error_result,
            "error": str(e),
        })

        return error_result
    else:
        return result
    finally:
        if mlflow_logger and mlflow_logger.active_run():
            mlflow_logger.end_run()
