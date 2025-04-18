"""Script to start the MLflow server with database validation."""
import logging
import sqlite3
import subprocess
from pathlib import Path

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_sqlite_db(db_path: Path) -> bool:
    """Check if SQLite database is valid."""
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA integrity_check")
    except sqlite3.DatabaseError:
        return False
    else:
        conn.close()
        return True
    finally:
        if "conn" in locals():
            conn.close()


def repair_sqlite_db(db_path: Path) -> bool:
    """Attempt to repair a corrupted SQLite database."""
    backup_path = db_path.with_name(f"{db_path.stem}_backup{db_path.suffix}")
    try:
        corrupt_conn = sqlite3.connect(str(db_path))
        new_conn = sqlite3.connect(str(backup_path))

        with new_conn:
            for line in corrupt_conn.iterdump():
                if line.strip():
                    try:
                        new_conn.execute(line)
                    except sqlite3.Error:
                        continue

        db_path.unlink(missing_ok=True)
        backup_path.rename(db_path)
    except (sqlite3.Error, OSError) as e:
        logger.warning("Could not repair database: %s", str(e))
        db_path.unlink(missing_ok=True)
        return False
    else:
        logger.info("Successfully repaired corrupted database")
        return True
    finally:
        if "corrupt_conn" in locals():
            corrupt_conn.close()
        if "new_conn" in locals():
            new_conn.close()


def start_mlflow_server() -> None:
    """Start the MLflow server with configurations from mlflow_config.yaml."""
    backend_dir = Path(__file__).parent
    config_path = backend_dir / "configs" / "mlflow_config.yaml"

    # Read config
    with config_path.open() as f:
        config = yaml.safe_load(f)

    # Ensure artifact directory exists
    artifact_path = Path(config["artifact"]["location"]).resolve()
    artifact_path.mkdir(exist_ok=True, parents=True)

    # Check SQLite database
    db_path = artifact_path / "mlflow.db"
    if db_path.exists() and not validate_sqlite_db(db_path):
        logger.warning("Database corruption detected. Attempting repair...")
        if not repair_sqlite_db(db_path):
            logger.warning("Starting with fresh database")

    # Start server
    command = [
        "mlflow",
        "server",
        "--backend-store-uri",
        f"sqlite:///{artifact_path}/mlflow.db",
        "--default-artifact-root",
        str(artifact_path),
        "--host",
        "127.0.0.1",
        "--port",
        "5001",
    ]
    # Command structure is fully controlled - no user input affects command structure
    subprocess.run(command, check=False)  # noqa: S603


if __name__ == "__main__":
    start_mlflow_server()
