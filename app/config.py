# Configuration for LLM Email Autowriter
import os
from typing import Optional

class Config:
    # Model Configuration
    MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
    MODEL_DTYPE: str = "auto"
    
    # Generation Parameters
    DEFAULT_TEMPERATURE: float = 0.65
    DEFAULT_TOP_P: float = 0.95
    DEFAULT_MAX_TOKENS: int = 1024
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # vLLM OpenAI-compatible server settings
    VLLM_HOST: str = os.getenv("VLLM_HOST", "localhost")
    VLLM_PORT: int = int(os.getenv("VLLM_PORT", "8000"))
    VLLM_API_BASE: str = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
    
    # Gradio Configuration
    GRADIO_HOST: str = "0.0.0.0"
    GRADIO_PORT: int = 7860
    GRADIO_SHARE: bool = True
    
    # Email Generation Settings
    TONE_OPTIONS = ["Professional", "Friendly", "Formal", "Casual", "Polite"]
    LENGTH_OPTIONS = ["Short", "Medium", "Long"]
    
    # Token limits by length
    TOKEN_LIMITS = {
        "Short": 150,
        "Medium": 300,
        "Long": 500
    }
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # HuggingFace Token (if needed)
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
