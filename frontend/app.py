from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS
import requests
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKEND_URL = "http://localhost:8000/api"  # FastAPI backend
REQUEST_TIMEOUT = 30  # seconds

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "flask-proxy"})

@app.route('/courses')
def courses():
    try:
        response = requests.get(
            f"{BACKEND_URL}/courses",
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        
        try:
            data = response.json()
        except ValueError as e:
            logger.error(f"Invalid JSON response: {response.text}")
            raise ValueError("Invalid response from backend server")
        
        # Handle response format
        if isinstance(data, dict):
            if 'data' in data:
                courses = list(data['data'].values())
            else:
                courses = list(data.values())
        elif isinstance(data, list):
            courses = data
        else:
            raise ValueError("Unexpected response format from backend")
            
        return render_template('courses.html', courses=courses)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return render_template('error.html', 
                            message="Could not connect to the server. Please try again later."), 500
    except Exception as e:
        logger.error(f"Error processing courses: {str(e)}")
        return render_template('error.html', message=str(e)), 500

@app.route('/course/<course_id>')
def course_detail(course_id):
    try:
        response = requests.get(
            f"{BACKEND_URL}/courses/{course_id}",
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        course = response.json()
        return render_template('course_detail.html', course=course)
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return render_template('error.html', 
                                message=f"Course {course_id} not found"), 404
        logger.error(f"HTTP error: {str(e)}")
        return render_template('error.html', 
                            message="Error loading course details"), 500
    except Exception as e:
        logger.error(f"Error fetching course: {str(e)}")
        return render_template('error.html', 
                            message="Internal server error"), 500

@app.route('/course/<course_id>/topic/<topic_id>')
def topic_problems(course_id, topic_id):
    try:
        response = requests.get(
            f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}",
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        
        return render_template(
            'topic_problems.html',
            course_id=course_id,
            topic_id=topic_id,
            problems=data.get('problems', [])
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return render_template('error.html',
                                message=f"Topic {topic_id} not found in course {course_id}"), 404
        logger.error(f"HTTP error: {str(e)}")
        return render_template('error.html',
                            message="Error loading problems"), 500
    except Exception as e:
        logger.error(f"Error fetching problems: {str(e)}")
        return render_template('error.html',
                            message="Internal server error"), 500

@app.route('/course/<course_id>/topic/<topic_id>/problem/<problem_id>')
def problem_detail(course_id, topic_id, problem_id):
    try:
        # Get problem info
        problem_response = requests.get(
            f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}/problems/{problem_id}",
            timeout=REQUEST_TIMEOUT
        )
        problem_response.raise_for_status()
        problem = problem_response.json()
        
        # Get topic info (optional)
        topic = None
        try:
            topic_response = requests.get(
                f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}",
                timeout=REQUEST_TIMEOUT
            )
            if topic_response.status_code == 200:
                topic = topic_response.json()
        except Exception as e:
            logger.warning(f"Couldn't fetch topic info: {str(e)}")
        
        return render_template(
            'problem.html',
            course_id=course_id,
            topic_id=topic_id,
            problem=problem,
            topic=topic
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return render_template('error.html',
                                message=f"Problem {problem_id} not found"), 404
        logger.error(f"HTTP error: {str(e)}")
        return render_template('error.html',
                            message="Error loading problem details"), 500
    except Exception as e:
        logger.error(f"Error fetching problem: {str(e)}")
        return render_template('error.html',
                            message="Internal server error"), 500
    
@app.route('/course/<course_id>/topic/<topic_id>/problem/<problem_id>/execute', methods=['POST'])
def api_execute(course_id, topic_id, problem_id):
    """Proxy to FastAPI execution endpoint"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        response = requests.post(
            f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/execute",
            json=request.get_json(),
            headers={'Content-Type': 'application/json'},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return jsonify(response.json())
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"Execution HTTP error: {str(e)}")
        try:
            error_data = e.response.json()
            return jsonify(error_data), e.response.status_code
        except ValueError:
            return jsonify({"error": str(e)}), e.response.status_code
    except Exception as e:
        logger.error(f"Execution error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/course/<course_id>/topic/<topic_id>/problem/<problem_id>/hints', methods=['POST'])
def api_hints(course_id, topic_id, problem_id):
    """Proxy to FastAPI hints endpoint"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        response = requests.post(
            f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/hints",
            json=request.get_json(),
            headers={'Content-Type': 'application/json'},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return jsonify(response.json())
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"Hints HTTP error: {str(e)}")
        try:
            error_data = e.response.json()
            return jsonify(error_data), e.response.status_code
        except ValueError:
            return jsonify({"error": str(e)}), e.response.status_code
    except Exception as e:
        logger.error(f"Hints error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
    
@app.route('/course/<course_id>/topic/<topic_id>/problem/<problem_id>/explain', methods=['POST'])
def api_explain(course_id, topic_id, problem_id):
    """Proxy to FastAPI explain endpoint"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        response = requests.post(
            f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/explain",
            json=request.get_json(),
            headers={'Content-Type': 'application/json'},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return jsonify(response.json())

    except requests.exceptions.HTTPError as e:
        logger.error(f"Explain HTTP error: {str(e)}")
        try:
            error_data = e.response.json()
            return jsonify(error_data), e.response.status_code
        except ValueError:
            return jsonify({"error": str(e)}), e.response.status_code
    except Exception as e:
        logger.error(f"Explain error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
    
@app.route('/course/<course_id>/topic/<topic_id>/problem/<problem_id>/optimize', methods=['POST'])
def api_optimize(course_id, topic_id, problem_id):
    """Proxy to FastAPI optimise endpoint"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        response = requests.post(
            f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/optimize",
            json=request.get_json(),
            headers={'Content-Type': 'application/json'},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return jsonify(response.json())

    except requests.exceptions.HTTPError as e:
        logger.error(f"Optimize HTTP error: {str(e)}")
        try:
            error_data = e.response.json()
            return jsonify(error_data), e.response.status_code
        except ValueError:
            return jsonify({"error": str(e)}), e.response.status_code
    except Exception as e:
        logger.error(f"Optimize error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
    
@app.route('/course/<course_id>/topic/<topic_id>/problem/<problem_id>/pseudocode', methods=['POST'])
def api_pseudocode(course_id, topic_id, problem_id):
    """Proxy to FastAPI pseudocode endpoint"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        response = requests.post(
            f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/pseudocode",
            json=request.get_json(),
            headers={'Content-Type': 'application/json'},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return jsonify(response.json())

    except requests.exceptions.HTTPError as e:
        logger.error(f"Pseudocode HTTP error: {str(e)}")
        try:
            error_data = e.response.json()
            return jsonify(error_data), e.response.status_code
        except ValueError:
            return jsonify({"error": str(e)}), e.response.status_code
    except Exception as e:
        logger.error(f"Pseudocode error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
    
@app.route('/course/<course_id>/topic/<topic_id>/problem/<problem_id>/steps', methods=['GET'])
def api_conceptual_steps(course_id, topic_id, problem_id):
    """Proxy to FastAPI conceptual steps endpoint"""
    try:
        response = requests.get(
            f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/steps",
            headers={'Content-Type': 'application/json'},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return jsonify(response.json())

    except requests.exceptions.HTTPError as e:
        logger.error(f"Conceptual steps HTTP error: {str(e)}")
        try:
            error_data = e.response.json()
            return jsonify(error_data), e.response.status_code
        except ValueError:
            return jsonify({"error": str(e)}), e.response.status_code
    except Exception as e:
        logger.error(f"Conceptual steps error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
    
@app.route('/course/<course_id>/topic/<topic_id>/problem/<problem_id>/submit', methods=['POST'])
def api_submit(course_id, topic_id, problem_id):
    """Proxy to FastAPI submission endpoint"""
    print('x')
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        response = requests.post(
            f"{BACKEND_URL}/problems/courses/{course_id}/topics/{topic_id}/problems/{problem_id}/submit",
            json=request.get_json(),
            headers={'Content-Type': 'application/json'},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return jsonify(response.json())

    except requests.exceptions.HTTPError as e:
        logger.error(f"Submit HTTP error: {str(e)}")
        try:
            error_data = e.response.json()
            return jsonify(error_data), e.response.status_code
        except ValueError:
            return jsonify({"error": str(e)}), e.response.status_code
    except Exception as e:
        logger.error(f"Submit error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(port=3000, debug=True)