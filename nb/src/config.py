"""
Centralized configuration management for all services.

This module provides a unified configuration system for the diet classification
pipeline, managing service URLs, API keys, database paths, and operational
parameters. It uses environment variables with sensible defaults and provides
validation capabilities.

The configuration supports both development and production environments
through environment variable overrides. All sensitive data (API keys) are
loaded from environment variables for security.

Example:
    >>> from config import app_config
    >>> print(app_config.OPENSEARCH_URL)
    'http://localhost:9200'
    >>> print(app_config.KETO_CARBS_THRESHOLD)
    10.0
"""
import os
from pathlib import Path
from typing import Optional

try:
    from decouple import config
except ImportError:
    # Fallback if decouple not available
    def config(key, default=None, cast=str):
        """Fallback configuration function when python-decouple is not available.
        
        Args:
            key: Environment variable name to retrieve
            default: Default value if environment variable is not set
            cast: Type conversion function (str, int, float, bool)
            
        Returns:
            The environment variable value converted to the specified type
        """
        return cast(os.environ.get(key, default))

class Config:
    """
    Centralized configuration class for all services.
    
    This class manages all configuration parameters for the diet classification
    system, including service endpoints, API credentials, database paths,
    and operational thresholds. It implements a singleton-like pattern where
    all configuration is accessed through the global `app_config` instance.
    
    Attributes:
        OPENSEARCH_URL: URL for OpenSearch service endpoint
        OLLAMA_URL: URL for Ollama LLM service endpoint  
        MLFLOW_TRACKING_URI: URI for MLflow experiment tracking
        HUGGING_FACE_HUB_TOKEN: API token for Hugging Face Hub access
        GOOGLE_API_KEY: API key for Google services
        PROJECT_ROOT: Root directory of the project (nb/src/)
        DB_PATH: Path to the SQLite knowledge graph database
        RAW_DATA_DIR: Directory containing raw CSV data files
        KETO_CARBS_THRESHOLD: Maximum carbohydrate threshold for keto classification
        RPM_LIMIT: Rate limiting for requests per minute
        TPM_LIMIT: Token per minute limit for LLM requests
        BATCH_SIZE: Default batch size for data processing operations
        REDIS_HOST: Hostname for Redis service
        REDIS_PORT: Port for Redis service
        REDIS_DB: Database number for Redis
        QWEN_MAX_CONTEXT_TOKENS: Maximum tokens for Qwen context window
    """
    
    # Service URLs
    OPENSEARCH_URL: str = config('OPENSEARCH_URL', default='http://localhost:9200')
    OLLAMA_URL: str = config('OLLAMA_URL', default='http://ollama:11434')
    MLFLOW_TRACKING_URI: str = config('MLFLOW_TRACKING_URI', default='http://localhost:5000')
    
    # Redis Configuration
    REDIS_HOST: str = config('REDIS_HOST', default='redis')
    REDIS_PORT: int = config('REDIS_PORT', default=6379, cast=int)
    REDIS_DB: int = config('REDIS_DB', default=0, cast=int)

    # API Keys
    HUGGING_FACE_HUB_TOKEN: Optional[str] = config('HUGGING_FACE_HUB_TOKEN', default=None)
    GOOGLE_API_KEY: Optional[str] = config('GOOGLE_API_KEY', default=None)
    
    # Project paths - Updated for nb/src/ relative paths
    PROJECT_ROOT = Path(__file__).resolve().parent  # nb/src/
    DB_PATH = PROJECT_ROOT / "data" / "knowledge_graph.db"
    RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
    
    # Classification thresholds
    KETO_CARBS_THRESHOLD: float = config('KETO_CARBS_THRESHOLD', default=10.0, cast=float)
    
    # LLM Context Limits
    QWEN_MAX_CONTEXT_TOKENS: int = config('QWEN_MAX_CONTEXT_TOKENS', default=1500, cast=int)

    # Rate limiting
    RPM_LIMIT: int = config('RPM_LIMIT', default=10, cast=int)
    TPM_LIMIT: int = config('TPM_LIMIT', default=200000, cast=int)
    
    # Batch processing
    BATCH_SIZE: int = config('BATCH_SIZE', default=25, cast=int)
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate that required configuration is present and accessible.
        
        This method checks that all required file paths exist and are accessible.
        It raises descriptive exceptions if critical paths are missing.
        
        Returns:
            bool: True if all validation checks pass
            
        Raises:
            FileNotFoundError: If any required directory or file path does not exist
            
        Example:
            >>> Config.validate()
            True
        """
        required_paths = [cls.RAW_DATA_DIR]
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Required path does not exist: {path}")
        return True

# Global config instance
app_config = Config() 