import mlflow
from datetime import datetime
import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

class MLFlowLogger:
    _active_run_id = None
    _logger = logging.getLogger("MLFlowLogger")

    def __init__(self, tracking_uri: str = "http://localhost:5001"):
        """Initialize MLFlow tracking with proper error handling"""
        self._initialized = False
        try:
            mlflow.set_tracking_uri(tracking_uri)
            self.experiment_name = "CodeGymLLMS"
            try:
                mlflow.set_experiment(self.experiment_name)
                self._initialized = True
            except Exception as e:
                self._logger.error(f"Failed to set experiment: {str(e)}")
        except Exception as e:
            self._logger.error(f"MLFlow initialization failed: {str(e)}")

    def _create_run_name(self, course: str, topic: str, problem: str) -> str:
        """Generate a consistent run name"""
        return f"{course}_{topic}_{problem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """Safely start a new MLflow run."""
        if not self._initialized:
            self._logger.warning("MLFlow not initialized - cannot start run")
            return None

        try:
            if MLFlowLogger._active_run_id:
                self._logger.warning(f"Active run exists {MLFlowLogger._active_run_id} - ending it")
                mlflow.end_run()

            run = mlflow.start_run(run_name=run_name, nested=nested)
            MLFlowLogger._active_run_id = run.info.run_uuid
            self._logger.info(f"Started run: {MLFlowLogger._active_run_id}")
            return run
        except Exception as e:
            self._logger.error(f"Failed to start run: {str(e)}")
            return None

    def end_run(self):
        """Safely end the current MLflow run."""
        if not self._initialized:
            return

        try:
            if MLFlowLogger._active_run_id:
                mlflow.end_run()
                self._logger.info(f"Ended run: {MLFlowLogger._active_run_id}")
                MLFlowLogger._active_run_id = None
        except Exception as e:
            self._logger.error(f"Failed to end run: {str(e)}")

    def log_param(self, key: str, value: Any):
        """Safely log a single parameter."""
        if not self._initialized or not MLFlowLogger._active_run_id:
            self._logger.warning(f"Skipping param log - {key}:{value}")
            return

        try:
            mlflow.log_param(key, value)
        except Exception as e:
            self._logger.error(f"Failed to log param {key}: {str(e)}")

    def log_metric(self, key: str, value: float):
        """Safely log a single metric."""
        if not self._initialized or not MLFlowLogger._active_run_id:
            self._logger.warning(f"Skipping metric log - {key}:{value}")
            return

        try:
            mlflow.log_metric(key, value)
        except Exception as e:
            self._logger.error(f"Failed to log metric {key}: {str(e)}")

    def log_metrics(self, metrics: Dict[str, float]):
        """Safely log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(key, value)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Safely log an artifact."""
        if not self._initialized or not MLFlowLogger._active_run_id:
            self._logger.warning(f"Skipping artifact log - {local_path}")
            return

        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            self._logger.error(f"Failed to log artifact {local_path}: {str(e)}")

    def log_text(self, data: str, artifact_file: str):
        """Safely log text data as an artifact."""
        try:
            with open(artifact_file, 'w') as f:
                f.write(data)
            self.log_artifact(artifact_file)
            os.remove(artifact_file)
        except Exception as e:
            self._logger.error(f"Failed to log text artifact: {str(e)}")

    def log_dict(self, data: Dict, artifact_file: str):
        """Safely log a dictionary as a JSON artifact."""
        try:
            with open(artifact_file, 'w') as f:
                json.dump(data, f)
            self.log_artifact(artifact_file)
            os.remove(artifact_file)
        except Exception as e:
            self._logger.error(f"Failed to log dict artifact: {str(e)}")

    def set_tag(self, key: str, value: str):
        """Safely set a tag for the current run."""
        if not self._initialized or not MLFlowLogger._active_run_id:
            self._logger.warning(f"Skipping tag set - {key}:{value}")
            return

        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            self._logger.error(f"Failed to set tag {key}: {str(e)}")

    def active_run(self) -> Optional[mlflow.entities.Run]:
        """Safely get the currently active MLflow run."""
        if not self._initialized:
            return None
        return mlflow.active_run()

    def log_submission(self, course: str, topic: str, problem: str, complexity: str,
                      language: str, code: str, execution_results: Dict,
                      llm_interactions: Optional[List] = None) -> None:
        """
        Safely log a complete code submission with all relevant data
        """
        if not self._initialized:
            self._logger.warning("MLFlow not initialized - skipping submission log")
            return

        run_name = self._create_run_name(course, topic, problem)
        with self.start_run(run_name=run_name) as run:
            if not run:
                return

            try:
                # Log basic parameters
                self.log_param('course', course)
                self.log_param('topic', topic)
                self.log_param('problem', problem)
                self.log_param('complexity', complexity)
                self.log_param('language', language)

                # Log test results
                if 'test_results' in execution_results.get('details', {}):
                    test_results = execution_results['details']['test_results']
                    passed_count = sum(1 for res in test_results if res.get('passed', False))
                    total_tests = len(test_results)
                    self.log_metric('passed_tests', passed_count)
                    self.log_metric('total_tests', total_tests)

                # Log code artifact
                temp_code_file = f"temp_{run_name}_code.{'py' if language.lower() == 'python' else 'js'}"
                self.log_text(code, temp_code_file)

                # Log test results artifact
                test_results_file = f"temp_{run_name}_test_results.json"
                self.log_dict(execution_results.get('details', {}), test_results_file)

                # Log LLM interactions if available
                if llm_interactions:
                    llm_file = f"temp_{run_name}_llm.json"
                    self.log_dict(llm_interactions, llm_file)

                # Set status tag
                status = 'failure'
                if 'test_results' in execution_results.get('details', {}):
                    if all(res.get('passed', False) for res in execution_results['details']['test_results']):
                        status = 'success'
                    elif any(res.get('passed', False) for res in execution_results['details']['test_results']):
                        status = 'partial'
                self.set_tag('status', status)
                self.set_tag('type', 'submission')

            except Exception as e:
                self._logger.error(f"Error logging submission: {str(e)}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure run is ended"""
        self.end_run()

    def log_llm_interaction(self, prompt: Dict, response: str, metadata: Optional[Dict] = None) -> None:
        """Log interactions with the Language Model."""
        with self.start_run(nested=True):
            self.log_dict(prompt, "llm_prompt.json")
            self.log_text(response, "llm_response.txt")
            if metadata:
                self.log_dict(metadata, "llm_metadata.json")
            self.set_tag("type", "llm_interaction")
        self.end_run()

    def log_compilation(self, language: str, code: str, compilation_result: Dict) -> None:
        """Log the result of code compilation."""
        run_name = f"Compilation_{language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with self.start_run(run_name=run_name):
            self.log_param("language", language)

            temp_code_file = f"temp_compile_{run_name}_code.{'py' if language.lower() == 'python' else 'js'}"
            with open(temp_code_file, 'w') as f:
                f.write(code)
            self.log_artifact(temp_code_file, "code")
            os.remove(temp_code_file)

            self.log_metric("compilation_successful", 1 if compilation_result.get("success", False) else 0)
            if compilation_result.get("error"):
                self.log_text(compilation_result["error"], "compilation_error.txt")
            if compilation_result.get("raw_logs"):
                self.log_text(compilation_result["raw_logs"], "compilation_logs.txt")

            self.set_tag('type', 'compilation')
            self.set_tag('status', 'success' if compilation_result.get("success", False) else 'failure')
        self.end_run()