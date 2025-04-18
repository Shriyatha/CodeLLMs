"""MLFlow logging utility for tracking experiments, metrics, and artifacts.

This module provides a wrapper around MLFlow's tracking API with additional
error handling and convenience methods for logging code submissions,
LLM interactions, and compilation results.
"""

import json
import logging
import types
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Run

__all__ = ["MLFlowLogger"]

@dataclass
class SubmissionData:
    """Container for all submission-related data."""

    course: str
    topic: str
    problem: str
    complexity: str
    language: str
    code: str
    execution_results: dict[str, Any]

class MLFlowLogger:
    """A utility class for logging experiments and metrics to MLflow."""

    _active_run_id: str | None = None
    _logger = logging.getLogger("MLFlowLogger")

    def __init__(self, tracking_uri: str = "http://localhost:5001") -> None:
        """Initialize MLFlow tracking with proper error handling.

        Args:
            tracking_uri: URI of the MLFlow tracking server

        """
        self._initialized = False
        try:
            mlflow.set_tracking_uri(tracking_uri)
            self.experiment_name = "CodeGymLLMS"
            try:
                mlflow.set_experiment(self.experiment_name)
                self._initialized = True
            except Exception:
                self._logger.exception("Failed to set experiment")
        except Exception:
            self._logger.exception("MLFlow initialization failed")

    def _create_run_name(self, course: str, topic: str, problem: str) -> str:
        """Generate a consistent run name.

        Args:
            course: Course identifier
            topic: Topic identifier
            problem: Problem identifier

        Returns:
            Generated run name string

        """
        return (
            f"{course}_{topic}_{problem}_"
            f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        )

    def start_run(
        self, run_name: str | None = None, *, nested: bool = False,
    ) -> Run | None:
        """Safely start a new MLflow run.

        Args:
            run_name: Name for the run
            nested: Whether to create a nested run

        Returns:
            The created Run object or None if failed

        """
        if not self._initialized:
            self._logger.warning("MLFlow not initialized - cannot start run")
            return None

        try:
            if MLFlowLogger._active_run_id:
                self._logger.warning(
                    "Active run exists %s - ending it", MLFlowLogger._active_run_id,
                )
                mlflow.end_run()

            run = mlflow.start_run(run_name=run_name, nested=nested)
            MLFlowLogger._active_run_id = run.info.run_uuid
            self._logger.info("Started run: %s", MLFlowLogger._active_run_id)
        except Exception:
            self._logger.exception("Failed to start run")
            return None
        else:
            return run

    def end_run(self) -> None:
        """Safely end the current MLflow run."""
        if not self._initialized:
            return

        try:
            if MLFlowLogger._active_run_id:
                mlflow.end_run()
                self._logger.info("Ended run: %s", MLFlowLogger._active_run_id)
                MLFlowLogger._active_run_id = None
        except Exception:
            self._logger.exception("Failed to end run")

    def log_param(self, key: str, value: str | float | bool) -> None:
        """Safely log a single parameter.

        Args:
            key: Parameter name
            value: Parameter value

        """
        if not self._initialized or not MLFlowLogger._active_run_id:
            self._logger.warning("Skipping param log - %s:%s", key, value)
            return

        try:
            mlflow.log_param(key, value)
        except Exception:
            self._logger.exception("Failed to log param %s", key)

    def log_metric(self, key: str, value: float) -> None:
        """Safely log a single metric.

        Args:
            key: Metric name
            value: Metric value

        """
        if not self._initialized or not MLFlowLogger._active_run_id:
            self._logger.warning("Skipping metric log - %s:%s", key, value)
            return

        try:
            mlflow.log_metric(key, value)
        except Exception:
            self._logger.exception("Failed to log metric %s", key)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Safely log multiple metrics.

        Args:
            metrics: Dictionary of metric names and values

        """
        for key, value in metrics.items():
            self.log_metric(key, value)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Safely log an artifact.

        Args:
            local_path: Path to the local file to log
            artifact_path: Optional path within the artifact URI

        """
        if not self._initialized or not MLFlowLogger._active_run_id:
            self._logger.warning("Skipping artifact log - %s", local_path)
            return

        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception:
            self._logger.exception("Failed to log artifact %s", local_path)

    def log_text(self, data: str, artifact_file: str) -> None:
        """Safely log text data as an artifact.

        Args:
            data: Text content to log
            artifact_file: Name for the artifact file

        """
        try:
            filepath = Path(artifact_file)
            with filepath.open("w") as f:
                f.write(data)
            self.log_artifact(artifact_file)
            filepath.unlink()
        except Exception:
            self._logger.exception("Failed to log text artifact")

    def log_dict(self, data: dict, artifact_file: str) -> None:
        """Safely log a dictionary as a JSON artifact.

        Args:
            data: Dictionary to log as JSON
            artifact_file: Name for the artifact file

        """
        try:
            filepath = Path(artifact_file)
            with filepath.open("w") as f:
                json.dump(data, f)
            self.log_artifact(artifact_file)
            filepath.unlink()
        except Exception:
            self._logger.exception("Failed to log dict artifact")

    def set_tag(self, key: str, value: str) -> None:
        """Safely set a tag for the current run.

        Args:
            key: Tag name
            value: Tag value

        """
        if not self._initialized or not MLFlowLogger._active_run_id:
            self._logger.warning("Skipping tag set - %s:%s", key, value)
            return

        try:
            mlflow.set_tag(key, value)
        except Exception:
            self._logger.exception("Failed to set tag %s", key)

    def active_run(self) -> Run | None:
        """Get the currently active MLflow run.

        Returns:
            The active Run object or None if no active run

        """
        if not self._initialized:
            return None
        return mlflow.active_run()

    def log_submission(self, submission: SubmissionData) -> None:
        """Log a complete code submission with relevant data.

        Args:
            submission: SubmissionData object containing all submission details

        """
        self._log_submission_internal(submission)

    def _log_submission_internal(
        self,
        submission: SubmissionData,
        llm_interactions: list[dict] | None = None,
    ) -> None:
        """Implement submission logging.

        Args:
            submission: SubmissionData object
            llm_interactions: Optional list of LLM interaction dictionaries

        """
        if not self._initialized:
            self._logger.warning("MLFlow not initialized - skipping submission log")
            return

        run_name = self._create_run_name(
            submission.course,
            submission.topic,
            submission.problem,
        )

        with self.start_run(run_name=run_name) as run:
            if not run:
                return

            try:
                # Log basic parameters
                self.log_param("course", submission.course)
                self.log_param("topic", submission.topic)
                self.log_param("problem", submission.problem)
                self.log_param("complexity", submission.complexity)
                self.log_param("language", submission.language)

                # Log test results
                details = submission.execution_results.get("details", {})
                if "test_results" in details:
                    test_results = details["test_results"]
                    passed_count = sum(
                        1 for res in test_results if res.get("passed", False)
                    )
                    total_tests = len(test_results)
                    self.log_metric("passed_tests", passed_count)
                    self.log_metric("total_tests", total_tests)

                # Log code artifact
                file_extension = (
                    "py" if submission.language.lower() == "python" else "js"
                )
                temp_code_file = f"temp_{run_name}_code.{file_extension}"
                self.log_text(submission.code, temp_code_file)

                # Log test results artifact
                test_results_file = f"temp_{run_name}_test_results.json"
                self.log_dict(details, test_results_file)

                # Log LLM interactions if available
                if llm_interactions:
                    llm_file = f"temp_{run_name}_llm.json"
                    self.log_dict(llm_interactions, llm_file)

                # Set status tag
                status = "failure"
                test_results_list = details.get("test_results", [])
                if test_results_list:
                    all_passed = all(
                        res.get("passed", False) for res in test_results_list
                    )
                    any_passed = any(
                        res.get("passed", False) for res in test_results_list
                    )
                    if all_passed:
                        status = "success"
                    elif any_passed:
                        status = "partial"
                self.set_tag("status", status)
                self.set_tag("type", "submission")

            except Exception:
                self._logger.exception("Error logging submission")

    def __enter__(self) -> "MLFlowLogger":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Context manager exit - ensure run is ended."""
        self.end_run()

    def log_llm_interaction(
        self, prompt: dict, response: str, metadata: dict | None = None,
    ) -> None:
        """Log interactions with the Language Model.

        Args:
            prompt: LLM prompt dictionary
            response: LLM response text
            metadata: Optional additional metadata

        """
        with self.start_run(nested=True):
            self.log_dict(prompt, "llm_prompt.json")
            self.log_text(response, "llm_response.txt")
            if metadata:
                self.log_dict(metadata, "llm_metadata.json")
            self.set_tag("type", "llm_interaction")
        self.end_run()

    def log_compilation(
        self, language: str, code: str, compilation_result: dict,
    ) -> None:
        """Log the result of code compilation.

        Args:
            language: Programming language
            code: Source code that was compiled
            compilation_result: Dictionary with compilation results

        """
        run_name = (
            f"Compilation_{language}_"
            f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        )
        with self.start_run(run_name=run_name):
            self.log_param("language", language)

            file_extension = (
                "py" if language.lower() == "python" else "js"
            )
            temp_code_file = f"temp_compile_{run_name}_code.{file_extension}"
            filepath = Path(temp_code_file)
            with filepath.open("w") as f:
                f.write(code)
            self.log_artifact(temp_code_file, "code")
            filepath.unlink()

            self.log_metric(
                "compilation_successful",
                1 if compilation_result.get("success", False) else 0,
            )
            if compilation_result.get("error"):
                self.log_text(
                    compilation_result["error"], "compilation_error.txt",
                )
            if compilation_result.get("raw_logs"):
                self.log_text(
                    compilation_result["raw_logs"], "compilation_logs.txt",
                )

            self.set_tag("type", "compilation")
            self.set_tag(
                "status",
                "success" if compilation_result.get("success", False) else "failure",
            )
        self.end_run()
