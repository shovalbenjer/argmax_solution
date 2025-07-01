"""
Centralized configuration management for all services.
"""
import os
from pathlib import Path
from typing import Optional
from decouple import config

class Config:
    """Centralized configuration class for all services."""
    
    # Service URLs
    OPENSEARCH_URL: str = config('OPENSEARCH_URL', default='http://localhost:9200')
    OLLAMA_URL: str = config('OLLAMA_URL', default='http://localhost:11434') 
    MLFLOW_TRACKING_URI: str = config('MLFLOW_TRACKING_URI', default='http://localhost:5000')
    
    # API Keys
    HUGGING_FACE_HUB_TOKEN: Optional[str] = config('HUGGING_FACE_HUB_TOKEN', default=None)
    GOOGLE_API_KEY: Optional[str] = config('GOOGLE_API_KEY', default=None)
    
    # Database paths
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DB_PATH = PROJECT_ROOT / "nb" / "src" / "data" / "knowledge_graph.db"
    RAW_DATA_DIR = PROJECT_ROOT / "nb" / "src" / "raw_data"
    
    # Classification thresholds
    KETO_CARBS_THRESHOLD: float = config('KETO_CARBS_THRESHOLD', default=20.0, cast=float)
    
    # Rate limiting
    RPM_LIMIT: int = config('RPM_LIMIT', default=10, cast=int)
    TPM_LIMIT: int = config('TPM_LIMIT', default=200000, cast=int)
    
    # Batch processing
    BATCH_SIZE: int = config('BATCH_SIZE', default=25, cast=int)
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        required_paths = [cls.RAW_DATA_DIR]
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Required path does not exist: {path}")
        return True

# Global config instance
app_config = Config() 