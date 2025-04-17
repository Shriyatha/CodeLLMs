import os
from pathlib import Path
import mlflow
import yaml

def start_mlflow_server():
    backend_dir = Path(__file__).parent
    config_path = backend_dir / "configs" / "mlflow_config.yaml"
    
    # Read config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Ensure artifact directory exists
    artifact_path = Path(config['artifact']['location'])
    artifact_path.mkdir(exist_ok=True)
    
    # Start server (in production, you'd run this separately)
    os.system(f"""
        mlflow server \
            --backend-store-uri sqlite:///{artifact_path}/mlflow.db \
            --default-artifact-root {artifact_path.absolute()} \
            --host 0.0.0.0 \
            --port 5001
    """)

if __name__ == "__main__":
    start_mlflow_server()