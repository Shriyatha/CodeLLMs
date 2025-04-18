import os
from pathlib import Path
from typing import List

class Config:
    # Application settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Virtual environment detection
    IN_VIRTUAL_ENV: bool = os.getenv("VIRTUAL_ENV") is not None
    
    # Ollama LLM settings
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    DEFAULT_LLM: str = os.getenv("DEFAULT_LLM", "phi")
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))  # seconds
    
    # Course configuration - Directly set path
    COURSES_DIR: Path = (Path(__file__).parent / "courses").resolve()
    
    # Execution settings
    MAX_CODE_EXECUTION_TIME: int = int(os.getenv("MAX_CODE_EXECUTION_TIME", "5"))  # seconds
    MAX_OUTPUT_LENGTH: int = int(os.getenv("MAX_OUTPUT_LENGTH", "10000"))  # characters
    
    # Security settings
    ALLOWED_ORIGINS: List[str] = [
        origin.strip() 
        for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ]
    
    @classmethod
    def validate_config(cls) -> None:
        """Validate configuration values and setup required directories"""
        try:
            # Create and validate courses directory
            cls.COURSES_DIR.mkdir(parents=True, exist_ok=True)
            
            # Validate directory is writable
            test_file = cls.COURSES_DIR / ".config_test"
            try:
                test_file.touch()
                test_file.unlink()
            except OSError as e:
                raise ValueError(f"Courses directory not writable: {cls.COURSES_DIR}") from e
            
            # Validate numerical values
            if cls.MAX_CODE_EXECUTION_TIME > 30:
                raise ValueError("MAX_CODE_EXECUTION_TIME cannot exceed 30 seconds")
                
            if cls.MAX_OUTPUT_LENGTH > 100000:
                raise ValueError("MAX_OUTPUT_LENGTH cannot exceed 100,000 characters")
                
            # Validate network settings
            if not cls.HOST:
                raise ValueError("HOST cannot be empty")
                
            if not (0 < cls.PORT <= 65535):
                raise ValueError("PORT must be between 1 and 65535")
                
            # Validate LLM settings
            if not cls.OLLAMA_HOST.startswith(("http://", "https://")):
                raise ValueError("OLLAMA_HOST must start with http:// or https://")
                
        except Exception as e:
            print(f"Configuration error: {str(e)}")
            if cls.DEBUG:
                import traceback
                traceback.print_exc()
            raise

    @classmethod
    def print_config(cls) -> None:
        """Print current configuration (useful for debugging)"""
        print("\nCurrent Configuration:")
        print("----------------------")
        for key, value in vars(cls).items():
            if not key.startswith("_") and not callable(value):
                print(f"{key}: {value}")
        print("----------------------\n")

# Validate configuration on import
Config.validate_config()

# # Print config in debug mode
# if Config.DEBUG:
#     Config.print_config()