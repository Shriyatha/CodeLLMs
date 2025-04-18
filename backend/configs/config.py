"""Application configuration settings."""
import os
from pathlib import Path
from typing import ClassVar

# Constants for validation
MAX_EXECUTION_TIME_LIMIT = 30
MAX_OUTPUT_LENGTH_LIMIT = 100000
MIN_PORT = 1
MAX_PORT = 65535
HTTP_PREFIXES = ("http://", "https://")

class Config:
    """Application configuration settings."""

    # Application settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Virtual environment detection
    IN_VIRTUAL_ENV: bool = os.getenv("VIRTUAL_ENV") is not None

    # Ollama LLM settings
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    DEFAULT_LLM: str = os.getenv("DEFAULT_LLM", "phi")
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))

    # Course configuration - Directly set path
    COURSES_DIR: Path = (Path(__file__).parent / "courses").resolve()

    # Execution settings
    MAX_CODE_EXECUTION_TIME: int = int(os.getenv("MAX_CODE_EXECUTION_TIME", "5"))
    MAX_OUTPUT_LENGTH: int = int(os.getenv("MAX_OUTPUT_LENGTH", "10000"))  # characters

    # Security settings
    ALLOWED_ORIGINS: ClassVar[list[str]] = [
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ]

    @classmethod
    def _raise_value_error(cls, message: str) -> None:
        """Raise a ValueError with the given message."""
        raise ValueError(message)

    @classmethod
    def validate_config(cls) -> None:
        """Validate configuration values and setup required directories."""
        try:
            # Create and validate courses directory
            cls.COURSES_DIR.mkdir(parents=True, exist_ok=True)

            # Validate directory is writable
            test_file = cls.COURSES_DIR / ".config_test"
            try:
                test_file.touch()
                test_file.unlink()
            except OSError:
                error_message = f"Courses directory not writable: {cls.COURSES_DIR}"
                cls._raise_value_error(error_message)

            # Validate numerical values
            if cls.MAX_CODE_EXECUTION_TIME > MAX_EXECUTION_TIME_LIMIT:
                error_message = (
                    f"MAX_CODE_EXECUTION_TIME cannot exceed "
                    f"{MAX_EXECUTION_TIME_LIMIT} seconds"
                )
                cls._raise_value_error(error_message)

            if cls.MAX_OUTPUT_LENGTH > MAX_OUTPUT_LENGTH_LIMIT:
                error_message = (
                    f"MAX_OUTPUT_LENGTH cannot exceed "
                    f"{MAX_OUTPUT_LENGTH_LIMIT} characters"
                )
                cls._raise_value_error(error_message)

            # Validate network settings
            if not cls.HOST:
                cls._raise_value_error("HOST cannot be empty")

            if not (MIN_PORT <= cls.PORT <= MAX_PORT):
                error_message = (
                    f"PORT must be between {MIN_PORT} and {MAX_PORT}"
                )
                cls._raise_value_error(error_message)

            # Validate LLM settings
            if not any(cls.OLLAMA_HOST.startswith(prefix) for prefix in HTTP_PREFIXES):
                error_message = "OLLAMA_HOST must start with http:// or https://"
                cls._raise_value_error(error_message)

        except ValueError:
            if cls.DEBUG:
                import traceback
                traceback.print_exc()
            raise

# Validate configuration on import
Config.validate_config()
