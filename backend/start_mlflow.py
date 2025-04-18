"""Script to start the MLflow server with robust error handling."""
import logging
import sqlite3
import subprocess
import sys
from pathlib import Path

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_sqlite_db(db_path: Path) -> bool:
    """Check if SQLite database is valid."""
    conn = None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA integrity_check")
    except sqlite3.DatabaseError:
        return False
    finally:
        if conn:
            conn.close()
    return True


def repair_sqlite_db(db_path: Path) -> bool:
    """Attempt to repair a corrupted SQLite database."""
    backup_path = db_path.with_name(f"{db_path.stem}_backup{db_path.suffix}")
    corrupt_db = None
    new_db = None
    try:
        # Try to create a backup/recovery
        corrupt_db = sqlite3.connect(str(db_path))
        new_db = sqlite3.connect(str(backup_path))

        with new_db:
            for line in corrupt_db.iterdump():
                if line.strip():
                    try:
                        new_db.execute(line)
                    except sqlite3.Error:
                        logger.warning("Skipping error during database repair: %s", line)
                        continue

        if db_path.exists():
            db_path.unlink()  # Remove corrupted db
        backup_path.rename(db_path)  # Replace with repaired version
        logger.info("Successfully repaired corrupted database: %s", db_path)
        return True
    except sqlite3.Error:
        logger.exception("Error during database repair")
        # If repair fails, just delete the corrupted db
        if db_path.exists():
            db_path.unlink()
            logger.warning("Deleted corrupted database: %s", db_path)
        return False
    except Exception:
        logger.exception("Unexpected error during database repair")
        if db_path.exists():
            db_path.unlink()
            logger.warning("Deleted corrupted database due to unexpected error: %s", db_path)
        return False
    finally:
        if corrupt_db:
            corrupt_db.close()
        if new_db:
            new_db.close()


def start_mlflow_server() -> int | None:
    """Start the MLflow server with configurations from mlflow_config.yaml."""
    backend_dir = Path(__file__).parent
    config_path = backend_dir / "configs" / "mlflow_config.yaml"

    try:
        # Read config
        with config_path.open() as f:
            config = yaml.safe_load(f)

        # Ensure artifact directory exists
        artifact_path = Path(config["artifact"]["location"])
        artifact_path.mkdir(exist_ok=True, parents=True)

        # Check SQLite database
        db_path = artifact_path / "mlflow.db"
        if db_path.exists() and not validate_sqlite_db(db_path):
            logger.warning("Database corruption detected. Attempting repair...")
            if not repair_sqlite_db(db_path):
                logger.warning("Could not repair database. Starting with a fresh database.")

        # Start server
        command = [
            "mlflow",
            "server",
            "--backend-store-uri",
            f"sqlite:///{db_path}",
            "--default-artifact-root",
            str(artifact_path.absolute()),
            "--host",
            "127.0.0.1",
            "--port",
            "5001",
        ]

        logger.info("🚀 Starting MLflow server...")
        logger.info("🔗 Tracking URI: http://127.0.0.1:5001")
        logger.info("💾 Artifact location: %s", artifact_path)

        # Run server in foreground
        result = subprocess.run(command, check=False)
        return result.returncode

    except FileNotFoundError as e:
        logger.exception("Configuration file not found: %s", e)
        return 1
    except yaml.YAMLError as e:
        logger.exception("Error parsing YAML config: %s", e)
        return 1
    except Exception:
        logger.exception("Unexpected error")
        return 1


if __name__ == "__main__":
    sys.exit(start_mlflow_server() or 0)
