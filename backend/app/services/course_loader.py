"""Module providing course loading functionality from TOML files."""
import json
import logging
import time
from typing import Any

import mlflow
import toml
from configs.config import Config

from app.services.mlflow_logger import MLFlowLogger

logger = logging.getLogger(__name__)


class CourseLoader:
    """Responsible for loading and managing course data from TOML files."""

    def __init__(self) -> None:
        """Initialize CourseLoader with empty cache and MLFlow logger."""
        self._courses_cache = None
        self._last_loaded = None
        self.mlflow_logger = MLFlowLogger()  # Instantiate MLFlowLogger

    async def load_all_courses(self) -> dict[str, dict[str, Any]]:
        """Load all courses from TOML files with caching."""
        courses = {}
        errored_files = []

        run = None
        try:
            # Check if cache is still valid
            if self._courses_cache and self._cache_is_valid():
                return self._courses_cache

            run = self.mlflow_logger.start_run(run_name="LoadAllCourses")
            self.mlflow_logger.log_param("courses_directory", str(Config.COURSES_DIR))
            loaded_count = 0

            for course_file in Config.COURSES_DIR.glob("*.toml"):
                try:
                    file_mtime = course_file.stat().st_mtime
                    with course_file.open(encoding="utf-8") as f:  # Using Path.open instead of open
                        data = toml.load(f)
                        if "course" not in data:
                            logger.warning("Invalid course file format in %s", course_file)
                            errored_files.append(str(course_file))
                            continue

                        course = self._validate_course(data["course"])
                        topics = self._process_topics(data.get("topics", []))

                        courses[course["id"]] = {
                            **course,
                            "topics": topics,
                            "last_modified": file_mtime,
                        }
                        loaded_count += 1
                except (BaseException, Exception) as e:  # pylint: disable=broad-except
                    errored_files.append(str(course_file))
                    self.mlflow_logger.log_text(str(e), f"error_loading_{course_file.name}")

            self.mlflow_logger.log_metric("loaded_courses_count", loaded_count)
            self.mlflow_logger.log_metric("errored_files_count", len(errored_files))
            load_duration = time.time() - self._last_loaded if self._last_loaded else 0
            self.mlflow_logger.log_metric("load_duration_seconds", load_duration)
            if errored_files:
               self.mlflow_logger.log_text(json.dumps(errored_files), "errored_course_files.json")

            self._courses_cache = courses
            self._last_loaded = time.time()

        except (BaseException, Exception) as e:  # pylint: disable=broad-except
            if self.mlflow_logger.active_run():
                self.mlflow_logger.log_text(str(e), "fatal_error_loading_courses")
            courses = {}  # Ensure an empty dict is returned in case of a fatal error
        finally:
            if run:
                mlflow.end_run()

        return courses

    def _cache_is_valid(self) -> bool:
        """Check if cached courses are still valid by checking file timestamps."""
        if not self._last_loaded:
            return False

        for _, course_data in self._courses_cache.values():  # Using values() instead of items()
            if course_data.get("last_modified", 0) > self._last_loaded:
                return False
        return True

    def _validate_course(self, course_data: dict) -> dict:
        """Validate course structure and set defaults."""
        run = None
        try:
            run = self.mlflow_logger.start_run(run_name="ValidateCourse", nested=True)
            self.mlflow_logger.log_param("course_data", course_data)
            required_fields = ["id", "title", "description"]
            if not all(field in course_data for field in required_fields):
                error_msg = "Course missing required fields."
                self.mlflow_logger.log_text(error_msg, "validation_error")
                raise ValueError(error_msg)

            validated_course = {
                "id": course_data["id"],
                "title": course_data["title"],
                "description": course_data.get("description", ""),
                "language": course_data.get("language", "python"),
            }
            self.mlflow_logger.log_dict(validated_course, "validated_course.json")
            return validated_course
        finally:
            if run:
                mlflow.end_run()

    def _process_topics(self, topics: list[dict]) -> list[dict]:
        """Process and validate topics structure."""
        validated_topics = []
        run_topics = None
        try:
            run_topics = self.mlflow_logger.start_run(run_name="ProcessTopics", nested=True)
            self.mlflow_logger.log_param("num_topics", len(topics))
            for topic in topics:
                run_topic = None
                try:
                    topic_id = topic.get("id", "unknown")
                    run_topic = self.mlflow_logger.start_run(
                        run_name=f"ProcessTopic_{topic_id}",
                        nested=True,
                    )
                    self.mlflow_logger.log_param("raw_topic_data", topic)
                    if "id" not in topic or "title" not in topic:
                        warning_msg = "Topic missing required fields, skipping."
                        logger.warning(warning_msg)
                        self.mlflow_logger.log_text(warning_msg, "validation_warning")
                        continue

                    validated_topic = {
                        "id": topic["id"],
                        "title": topic["title"],
                        "description": topic.get("description", ""),
                        "problems": self._process_problems(topic.get("problems", [])),
                    }
                    self.mlflow_logger.log_dict(validated_topic, "validated_topic.json")
                    validated_topics.append(validated_topic)
                except Exception as e:  # pylint: disable=broad-except
                    error_msg = f"Error processing topic: {e}"
                    logger.exception(error_msg)
                    self.mlflow_logger.log_text(str(e), f"error_processing_topic_{topic_id}")
                    continue
                finally:
                    if run_topic:
                        mlflow.end_run()
            self.mlflow_logger.log_metric("validated_topics_count", len(validated_topics))
        finally:
            if run_topics:
                mlflow.end_run()
        return validated_topics

    def _process_problems(self, problems: list[dict]) -> list[dict]:
        """Process and validate problems structure."""
        validated_problems = []
        run_problems = None
        try:
            run_problems = self.mlflow_logger.start_run(run_name="ProcessProblems", nested=True)
            self.mlflow_logger.log_param("num_problems", len(problems))
            for problem in problems:
                run_problem = None
                try:
                    problem_id = problem.get("id", "unknown")
                    run_problem = self.mlflow_logger.start_run(
                        run_name=f"ProcessProblem_{problem_id}",
                        nested=True,
                    )
                    self.mlflow_logger.log_param("raw_problem_data", problem)
                    if "id" not in problem or "title" not in problem:
                        warning_msg = "Problem missing required fields, skipping."
                        logger.warning(warning_msg)
                        self.mlflow_logger.log_text(warning_msg, "validation_warning")
                        continue

                    validated_problem = {
                        "id": problem["id"],
                        "title": problem["title"],
                        "description": problem.get("description", ""),
                        "complexity": problem.get("complexity", "medium"),
                        "starter_code": problem.get("starter_code", ""),
                        "visible_test_cases": problem.get("visible_test_cases", []),
                        "hidden_test_cases": problem.get("hidden_test_cases", []),
                    }

                    # Validate test cases
                    self._validate_test_cases(validated_problem["visible_test_cases"])
                    self._validate_test_cases(validated_problem["hidden_test_cases"])

                    self.mlflow_logger.log_dict(validated_problem, "validated_problem.json")
                    validated_problems.append(validated_problem)
                except Exception as e:  # pylint: disable=broad-except
                    error_msg = f"Error processing problem: {e}"
                    logger.exception(error_msg)
                    self.mlflow_logger.log_text(str(e), f"error_processing_problem_{problem_id}")
                    continue
                finally:
                    if run_problem:
                        mlflow.end_run()
            self.mlflow_logger.log_metric("validated_problems_count", len(validated_problems))
        finally:
            if run_problems:
                mlflow.end_run()
        return validated_problems

    def _validate_test_cases(self, test_cases: list[dict]) -> None:
        """Flexible test case validation that handles both 'output' and 'expected_output'."""
        run_test_cases = None
        try:
            run_test_cases = self.mlflow_logger.start_run(run_name="ValidateTestCases", nested=True)
            self.mlflow_logger.log_param("num_test_cases", len(test_cases))
            if not isinstance(test_cases, list):
                error_msg = "Test cases must be a list."
                self.mlflow_logger.log_text(error_msg, "validation_error")
                raise TypeError(error_msg)  # Changed from ValueError to TypeError

            for i, case in enumerate(test_cases):
                run_test_case = None
                try:
                    run_test_case = self.mlflow_logger.start_run(run_name=f"ValidateTestCase_{i}", nested=True)
                    self.mlflow_logger.log_param("raw_test_case", case)
                    if not isinstance(case, dict):
                        error_msg = "Each test case must be a dictionary."
                        self.mlflow_logger.log_text(error_msg, "validation_error")
                        raise TypeError(error_msg)  # Changed from ValueError to TypeError

                    # Handle both 'output' and 'expected_output'
                    if "expected_output" in case:
                        case["output"] = case["expected_output"]
                    elif "output" not in case:
                        error_msg = "Test case missing required output field."
                        self.mlflow_logger.log_text(error_msg, "validation_error")
                        raise ValueError(error_msg)

                    # Set default empty string if input is missing
                    if "input" not in case:
                        case["input"] = ""
                        self.mlflow_logger.log_param("input_missing", input_missing=True)

                    # Ensure output is string
                    if not isinstance(case["output"], str):
                        case["output"] = str(case["output"])
                        self.mlflow_logger.log_param("output_converted_to_string", output_converted=True)
                    self.mlflow_logger.log_dict(case, "validated_test_case.json")
                finally:
                    if run_test_case:
                        mlflow.end_run()
        finally:
            if run_test_cases:
                mlflow.end_run()

    async def get_course(self, course_id: str) -> dict[str, Any] | None:
        """Get a single course by ID."""
        run = None
        try:
            run = self.mlflow_logger.start_run(run_name=f"GetCourse_{course_id}", nested=True)
            self.mlflow_logger.log_param("course_id", course_id)
            courses = await self.load_all_courses()
            course = courses.get(course_id)
            if course:
                self.mlflow_logger.log_dict(course, f"found_course_{course_id}.json")
            else:
                self.mlflow_logger.log_text("Course not found.", f"get_course_{course_id}_status")
            return course
        finally:
            if run:
                mlflow.end_run()

    async def get_topic(self, course_id: str, topic_id: str) -> dict[str, Any] | None:
        """Get a topic from a course."""
        run = None
        try:
            run = self.mlflow_logger.start_run(run_name=f"GetTopic_{course_id}_{topic_id}", nested=True)
            self.mlflow_logger.log_param("course_id", course_id)
            self.mlflow_logger.log_param("topic_id", topic_id)
            course = await self.get_course(course_id)
            if not course:
                self.mlflow_logger.log_text("Course not found.", "get_topic_status")
                return None

            topic = next(
                (t for t in course.get("topics", []) if t["id"] == topic_id),
                None,
            )
            if topic:
                self.mlflow_logger.log_dict(topic, f"found_topic_{topic_id}.json")
            else:
                self.mlflow_logger.log_text("Topic not found.", "get_topic_status")
            return topic
        finally:
            if run:
                mlflow.end_run()

    async def get_problem(self, course_id: str, topic_id: str, problem_id: str) -> dict[str, Any] | None:
        """Get a problem from a topic."""
        run = None
        try:
            run = self.mlflow_logger.start_run(run_name=f"GetProblem_{course_id}_{topic_id}_{problem_id}", nested=True)
            self.mlflow_logger.log_param("course_id", course_id)
            self.mlflow_logger.log_param("topic_id", topic_id)
            self.mlflow_logger.log_param("problem_id", problem_id)
            topic = await self.get_topic(course_id, topic_id)
            if not topic:
                self.mlflow_logger.log_text("Topic not found.", "get_problem_status")
                return None

            problem = next(
                (p for p in topic.get("problems", []) if p["id"] == problem_id),
                None,
            )
            if problem:
                self.mlflow_logger.log_dict(problem, f"found_problem_{problem_id}.json")
            else:
                self.mlflow_logger.log_text("Problem not found.", "get_problem_status")
            return problem
        finally:
            if run:
                mlflow.end_run()
