# Set shell mode to work with environment variables
set shell := ["bash", "-c"]

# Load environment variables from .env file
set dotenv-load

setup:
    # Create virtual environment and install dependencies
    uv venv .venv_test
    . .venv_test/bin/activate && uv pip install -r requirements.txt
    uv pip install mlflow
    
    # Pull required Docker images
    docker pull python:3.12-slim
    docker pull node:18-slim
    ollama pull codegemma:7b
    ollama pull phi

run:
    # Start all services in the foreground
    . .venv_test/bin/activate && \
    ollama pull codegemma:7b & \
    ollama pull phi & \
    cd backend && python3 start_mlflow.py & \
    cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --timeout-keep-alive 300 & \
    cd frontend && python3 app.py & \
    wait

docs:
    . .venv_test/bin/activate && \
    mkdocs serve

stop:
    # Stop all services
    pkill -f "python.*start_mlflow.py" || true
    pkill -f "uvicorn.*app.main:app" || true
    pkill -f "python.*app.py" || true
    echo "All services stopped"

clean:
    # Clean up
    rm -rf .venv_test
    docker rmi python:3.12-slim node:18-slim || true