"""Flask proxy frontend for the learning platform."""
import logging

import requests
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKEND_URL = "http://localhost:8000/api"  # FastAPI backend
REQUEST_TIMEOUT = 60  # seconds
NOT_FOUND = 404
INTERNAL_SERVER_ERROR = 500
OK = 200
INVALID_JSON_ERROR = "Invalid JSON response from backend server"
UNEXPECTED_RESPONSE_FORMAT = "Unexpected response format from backend"


@app.route("/")
def index() -> str:
    """Render the index page."""
    return render_template("index.html")


@app.route("/health")
def health_check() -> Response:
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "flask-proxy"})


@app.route("/courses")
def courses() -> str | tuple[str, int]:
    """Fetch and display a list of available courses."""
    def _handle_unexpected_response(message: str) -> None:
        """Raise a TypeError with the given message."""
        raise TypeError(message)

    try:
        response = requests.get(
            f"{BACKEND_URL}/courses",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        try:
            data = response.json()
        except ValueError as e:
            log_message = f"Invalid JSON response: {response.text}"
            logger.exception(log_message)
            invalid_json_message = "Invalid response from backend server"
            raise ValueError(invalid_json_message) from e

        if isinstance(data, dict):
            courses = list(data["data"].values()) if "data" in data else list(data.values())
        elif isinstance(data, list):
            courses = data
        else:
            _handle_unexpected_response(UNEXPECTED_RESPONSE_FORMAT)

        return render_template("courses.html", courses=courses)

    except requests.exceptions.RequestException:
        logger.exception("Request failed")
        return (
            render_template(
                "error.html",
                message="Could not connect to the server. Please try again later.",
            ),
            INTERNAL_SERVER_ERROR,
        )
    except Exception as e:
        logger.exception("Error processing courses")
        return render_template("error.html", message=str(e)), INTERNAL_SERVER_ERROR


@app.route("/course/<course_id>")
def course_detail(course_id: str) -> str | tuple[str, int]:
    """Fetch and display details for a specific course."""
    try:
        response = requests.get(
            f"{BACKEND_URL}/courses/{course_id}",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        course = response.json()
        return render_template("course_detail.html", course=course)

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == NOT_FOUND:
            return (
                render_template(
                    "error.html", message=f"Course {course_id} not found",
                ),
                NOT_FOUND,
            )
        logger.exception("HTTP error")
        return (
            render_template("error.html", message="Error loading course details"),
            INTERNAL_SERVER_ERROR,
        )
    except Exception:
        logger.exception("Error fetching course")
        return (
            render_template("error.html", message="Internal server error"),
            INTERNAL_SERVER_ERROR,
        )


@app.route("/course/<course_id>/topic/<topic_id>")
def topic_problems(course_id: str, topic_id: str) -> str | tuple[str, int]:
    """Fetch and display problems for a specific topic within a course."""
    try:
        response = requests.get(
            f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        return render_template(
            "topic_problems.html",
            course_id=course_id,
            topic_id=topic_id,
            problems=data.get("problems", []),
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == NOT_FOUND:
            return (
                render_template(
                    "error.html",
                    message=f"Topic {topic_id} not found in course {course_id}",
                ),
                NOT_FOUND,
            )
        logger.exception("HTTP error")
        return (
            render_template("error.html", message="Error loading problems"),
            INTERNAL_SERVER_ERROR,
        )
    except Exception:
        logger.exception("Error fetching problems")
        return (
            render_template("error.html", message="Internal server error"),
            INTERNAL_SERVER_ERROR,
        )


@app.route("/course/<course_id>/topic/<topic_id>/problem/<problem_id>")
def problem_detail(
    course_id: str, topic_id: str, problem_id: str,
) -> str | tuple[str, int]:
    """Fetch and display details for a specific problem."""
    try:
        # Get problem info
        problem_response = requests.get(
            f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}/problems/{problem_id}",
            timeout=REQUEST_TIMEOUT,
        )
        problem_response.raise_for_status()
        problem = problem_response.json()

        # Get topic info (optional)
        topic = None
        try:
            topic_response = requests.get(
                f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}",
                timeout=REQUEST_TIMEOUT,
            )
            topic_response.raise_for_status()
            if topic_response.status_code == OK:
                topic = topic_response.json()
        except requests.exceptions.RequestException:
            logger.warning("Couldn't fetch topic info")

        return render_template(
            "problem.html",
            course_id=course_id,
            topic_id=topic_id,
            problem=problem,
            topic=topic,
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == NOT_FOUND:
            return (
                render_template(
                    "error.html", message=f"Problem {problem_id} not found",
                ),
                NOT_FOUND,
            )
        logger.exception("HTTP error")
        return (
            render_template("error.html", message="Error loading problem details"),
            INTERNAL_SERVER_ERROR,
        )
    except Exception:
        logger.exception("Error fetching problem")
        return (
            render_template("error.html", message="Internal server error"),
            INTERNAL_SERVER_ERROR,
        )


@app.route(
    "/course/<course_id>/topic/<topic_id>/problem/<problem_id>/execute",
    methods=["POST"],
)
def api_execute(
    course_id: str, topic_id: str, problem_id: str,
) -> Response | tuple[Response, int]:
    """Proxy to FastAPI execution endpoint."""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        response = requests.post(
            f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/execute",
            json=request.get_json(),
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return jsonify(response.json())

    except requests.exceptions.HTTPError as e:
        logger.exception("Execution HTTP error")
        try:
            error_data = e.response.json()
            return jsonify(error_data), e.response.status_code
        except ValueError:
            return jsonify({"error": str(e)}), e.response.status_code
    except Exception:
        logger.exception("Execution error")
        return jsonify({"error": "Internal server error"}), INTERNAL_SERVER_ERROR


@app.route(
    "/course/<course_id>/topic/<topic_id>/problem/<problem_id>/hints",
    methods=["POST"],
)
def api_hints(
    course_id: str, topic_id: str, problem_id: str,
) -> Response | tuple[Response, int]:
    """Proxy to FastAPI hints endpoint."""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        response = requests.post(
            f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/hints",
            json=request.get_json(),
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return jsonify(response.json())

    except requests.exceptions.HTTPError as e:
        logger.exception("Hints HTTP error")
        try:
            error_data = e.response.json()
            return jsonify(error_data), e.response.status_code
        except ValueError:
            return jsonify({"error": str(e)}), e.response.status_code
    except Exception:
        logger.exception("Hints error")
        return jsonify({"error": "Internal server error"}), INTERNAL_SERVER_ERROR


@app.route(
    "/course/<course_id>/topic/<topic_id>/problem/<problem_id>/explain",
    methods=["POST"],
)
def api_explain(
    course_id: str, topic_id: str, problem_id: str,
) -> Response | tuple[Response, int]:
    """Generate structured error explanation with actionable fixes."""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        request_data = {"code": request.get_json().get("code"), "language": request.get_json().get("language")}

        response = requests.post(
            f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/explain",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return jsonify(response.json())

    except requests.exceptions.HTTPError as e:
        logging.exception("Explain HTTP error")
        try:
            error_data = e.response.json()
            return jsonify(error_data), e.response.status_code
        except ValueError:
            return jsonify({"error": str(e)}), e.response.status_code
    except Exception:
        logging.exception("Explain error")
        return jsonify({"error": "Internal server error"}), INTERNAL_SERVER_ERROR


@app.route(
    "/course/<course_id>/topic/<topic_id>/problem/<problem_id>/optimize",
    methods=["POST"],
)
def api_optimize(
    course_id: str, topic_id: str, problem_id: str,
) -> Response | tuple[Response, int]:
    """Proxy to FastAPI optimise endpoint."""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        response = requests.post(
            f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/optimize",
            json=request.get_json(),
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return jsonify(response.json())

    except requests.exceptions.HTTPError as e:
        logger.exception("Optimize HTTP error")
        try:
            error_data = e.response.json()
            return jsonify(error_data), e.response.status_code
        except ValueError:
            return jsonify({"error": str(e)}), e.response.status_code
    except Exception:
        logger.exception("Optimize error")
        return jsonify({"error": "Internal server error"}), INTERNAL_SERVER_ERROR


@app.route(
    "/course/<course_id>/topic/<topic_id>/problem/<problem_id>/pseudocode",
    methods=["POST"],
)
def api_pseudocode(
    course_id: str, topic_id: str, problem_id: str,
) -> Response | tuple[Response, int]:
    """Proxy to FastAPI pseudocode endpoint."""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        response = requests.post(
            f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/pseudocode",
            json=request.get_json(),
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return jsonify(response.json())

    except requests.exceptions.HTTPError as e:
        logger.exception("Pseudocode HTTP error")
        try:
            error_data = e.response.json()
            return jsonify(error_data), e.response.status_code
        except ValueError:
            return jsonify({"error": str(e)}), e.response.status_code
    except Exception:
        logger.exception("Pseudocode error")
        return jsonify({"error": "Internal server error"}), INTERNAL_SERVER_ERROR


@app.route(
    "/course/<course_id>/topic/<topic_id>/problem/<problem_id>/steps",
    methods=["GET"],
)
def api_conceptual_steps(
    course_id: str, topic_id: str, problem_id: str,
) -> Response | tuple[Response, int]:
    """Proxy to FastAPI conceptual steps endpoint."""
    try:
        response = requests.get(
            f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/steps",
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return jsonify(response.json())

    except requests.exceptions.HTTPError as e:
        logger.exception("Conceptual steps HTTP error")
        try:
            error_data = e.response.json()
            return jsonify(error_data), e.response.status_code
        except ValueError:
            return jsonify({"error": str(e)}), e.response.status_code
    except Exception:
        logger.exception("Conceptual steps error")
        return jsonify({"error": "Internal server error"}), INTERNAL_SERVER_ERROR


@app.route(
    "/course/<course_id>/topic/<topic_id>/problem/<problem_id>/submit",
    methods=["POST"],
)
def api_submit(
    course_id: str, topic_id: str, problem_id: str,
) -> Response | tuple[Response, int]:
    """Proxy to FastAPI submission endpoint."""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        response = requests.post(
            f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/submit",
            json=request.get_json(),
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return jsonify(response.json())

    except requests.exceptions.HTTPError as e:
        logger.exception("Submit HTTP error")
        try:
            error_data = e.response.json()
            return jsonify(error_data), e.response.status_code
        except ValueError:
            return jsonify({"error": str(e)}), e.response.status_code
    except Exception:
        logger.exception("Submit error")
        return jsonify({"error": "Internal server error"}), INTERNAL_SERVER_ERROR


if __name__ == "__main__":
    app.run(port=3000, debug=False)
