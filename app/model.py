"""
Model handler for LLM inference using vLLM OpenAI-compatible API.
Adapted from the prototype notebook's ModelCall class.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from openai import OpenAI
from .config import Config
from .prompt_template import EmailPromptTemplate

logger = logging.getLogger(__name__)

class EmailGeneratorModel:
    """
    Handles LLM model interactions for email generation.
    Uses vLLM's OpenAI-compatible API for inference.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.prompt_template = EmailPromptTemplate()
        self._client = None
        self._initialized = False
    
    def _get_client(self) -> OpenAI:
        """Get or create OpenAI client for vLLM server"""
        if self._client is None:
            try:
                self._client = OpenAI(
                    base_url=self.config.VLLM_API_BASE,
                    api_key="EMPTY"  # vLLM doesn't require API key
                )
                logger.info(f"Initialized OpenAI client for vLLM server at {self.config.VLLM_API_BASE}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                raise
        return self._client
    
    def generate_email(
        self,
        intent: str,
        tone: str = "Professional",
        length: str = "Medium",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate an email based on user intent and preferences.
        
        Args:
            intent: User's email intent/purpose
            tone: Email tone (Professional, Friendly, Formal, Casual, Polite)
            length: Email length (Short, Medium, Long)
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated email text
        """
        try:
            # Set default parameters
            temperature = temperature or self.config.DEFAULT_TEMPERATURE
            top_p = top_p or self.config.DEFAULT_TOP_P
            max_tokens = max_tokens or self.config.TOKEN_LIMITS.get(length, self.config.DEFAULT_MAX_TOKENS)
            
            # Generate prompt
            messages = self.prompt_template.generate_email_prompt(intent, tone, length)
            
            # Get client and make request
            client = self._get_client()
            
            logger.info(f"Generating email with intent: '{intent[:50]}...', tone: {tone}, length: {length}")
            
            response = client.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=False
            )
            
            if response.choices and len(response.choices) > 0:
                generated_text = response.choices[0].message.content.strip()
                logger.info(f"Successfully generated email ({len(generated_text)} characters)")
                return generated_text
            else:
                logger.warning("No response generated from model")
                return "Error: No response generated from the model."
                
        except Exception as e:
            logger.error(f"Error generating email: {e}")
            return f"Error generating email: {str(e)}"
    
    async def generate_email_async(
        self,
        intent: str,
        tone: str = "Professional",
        length: str = "Medium",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Async version of email generation.
        """
        # Run the synchronous method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate_email,
            intent, tone, length, temperature, top_p, max_tokens
        )
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the model service is healthy and responsive.
        
        Returns:
            Dictionary with health status information
        """
        try:
            client = self._get_client()
            
            # Try a simple generation request
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello' in one word."}
            ]
            
            response = client.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=test_messages,
                max_tokens=10,
                temperature=0.1
            )
            
            if response.choices:
                return {
                    "status": "healthy",
                    "model": self.config.MODEL_NAME,
                    "api_base": self.config.VLLM_API_BASE,
                    "test_response": response.choices[0].message.content.strip()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "No response from model",
                    "model": self.config.MODEL_NAME,
                    "api_base": self.config.VLLM_API_BASE
                }
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.config.MODEL_NAME,
                "api_base": self.config.VLLM_API_BASE
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.config.MODEL_NAME,
            "api_base": self.config.VLLM_API_BASE,
            "default_temperature": self.config.DEFAULT_TEMPERATURE,
            "default_top_p": self.config.DEFAULT_TOP_P,
            "default_max_tokens": self.config.DEFAULT_MAX_TOKENS,
            "available_tones": self.config.TONE_OPTIONS,
            "available_lengths": self.config.LENGTH_OPTIONS,
            "token_limits": self.config.TOKEN_LIMITS
        }


# Global model instance (initialized when needed)
_model_instance: Optional[EmailGeneratorModel] = None

def get_model() -> EmailGeneratorModel:
    """
    Get the global model instance (singleton pattern).
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = EmailGeneratorModel()
    return _model_instance
