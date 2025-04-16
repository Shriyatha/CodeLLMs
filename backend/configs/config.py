import os
from pathlib import Path
from typing import Optional

class Config:
    # Application settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Ollama LLM settings
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    DEFAULT_LLM: str = os.getenv("DEFAULT_LLM", "phi")  # Phi-2 (2.7B parameters)
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))  # seconds
    
    # Course configuration - FIXED PATH
    COURSES_DIR: Path = (
        Path(os.getenv("COURSES_DIR")) 
        if os.getenv("COURSES_DIR")
        else Path(__file__).parent / "courses"  # Now points to configs/courses
    )
    
    # Execution settings
    MAX_CODE_EXECUTION_TIME: int = int(os.getenv("MAX_CODE_EXECUTION_TIME", "5"))  # seconds
    MAX_OUTPUT_LENGTH: int = int(os.getenv("MAX_OUTPUT_LENGTH", "10000"))  # characters
    
    # Security settings
    ALLOWED_ORIGINS: list[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    
    @classmethod
    def validate_config(cls):
        """Validate configuration values"""
        try:
            if not cls.COURSES_DIR.exists():
                # Create directory if it doesn't exist
                cls.COURSES_DIR.mkdir(parents=True, exist_ok=True)
                print(f"Created courses directory at: {cls.COURSES_DIR}")
                
            if cls.MAX_CODE_EXECUTION_TIME > 30:
                raise ValueError("MAX_CODE_EXECUTION_TIME cannot exceed 30 seconds")
                
            if cls.MAX_OUTPUT_LENGTH > 100000:
                raise ValueError("MAX_OUTPUT_LENGTH cannot exceed 100,000 characters")
        except Exception as e:
            print(f"Configuration error: {str(e)}")
            raise

# Validate configuration on import
Config.validate_config()