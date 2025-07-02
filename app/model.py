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
                
                # Test the connection
                test_response = self._client.chat.completions.create(
                    model=self.config.MODEL_NAME,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                logger.info("vLLM server connection successful")
                
            except Exception as e:
                logger.error(f"Failed to connect to vLLM server: {e}")
                self._client = None  # Reset to None so we can try alternatives
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
            logger.error(f"Error generating email with vLLM: {e}")
            logger.info("Falling back to template-based generation")
            return self._generate_fallback_email(intent, tone, length)
    
    def _generate_fallback_email(self, intent: str, tone: str, length: str) -> str:
        """Generate a template-based email when AI model is unavailable"""
        try:
            # Template-based email generation
            tone_styles = {
                "Professional": {
                    "greeting": "Dear [Recipient],",
                    "closing": "Best regards,",
                    "style": "formal and business-appropriate"
                },
                "Friendly": {
                    "greeting": "Hi [Recipient],",
                    "closing": "Best wishes,", 
                    "style": "warm and approachable"
                },
                "Formal": {
                    "greeting": "Dear [Recipient],",
                    "closing": "Sincerely,",
                    "style": "very formal and official"
                },
                "Casual": {
                    "greeting": "Hey [Recipient],",
                    "closing": "Thanks,",
                    "style": "relaxed and informal"
                },
                "Polite": {
                    "greeting": "Dear [Recipient],",
                    "closing": "Kind regards,",
                    "style": "courteous and respectful"
                }
            }
            
            style = tone_styles.get(tone, tone_styles["Professional"])
            
            # Generate subject based on intent
            subject = f"Re: {intent.strip()}"
            if len(subject) > 50:
                subject = subject[:47] + "..."
            
            # Generate body based on length
            if length == "Short":
                body = f"""I hope this message finds you well.

{intent.strip()}.

Please let me know if you need any additional information."""
            elif length == "Long":
                body = f"""I hope this message finds you well.

I am writing to you regarding the following matter: {intent.strip()}.

I would greatly appreciate your assistance with this request. If you require any additional information or documentation to process this request, please do not hesitate to let me know.

I understand that you may need time to review this matter, and I am happy to work with your schedule. Please feel free to contact me if you have any questions or if there is anything else I can provide to facilitate this process.

Thank you very much for your time and consideration. I look forward to hearing from you soon."""
            else:  # Medium
                body = f"""I hope this message finds you well.

I am writing to request your assistance with the following: {intent.strip()}.

I would be grateful if you could help me with this matter. Please let me know if you need any additional information from my end.

Thank you for your time and consideration. I look forward to your response."""
            
            # Construct the final email
            email = f"""Subject: {subject}

{style['greeting']}

{body}

{style['closing']}
[Your Name]

---
Generated by LLM Email Autowriter (Template Mode)
Tone: {tone} | Length: {length}"""
            
            logger.info(f"Generated fallback email for intent: '{intent[:30]}...'") 
            return email
            
        except Exception as e:
            logger.error(f"Error in fallback generation: {e}")
            return f"""Subject: {intent}

Dear [Recipient],

I hope this message finds you well.

Regarding: {intent}

[Please customize this template email based on your specific needs]

Best regards,
[Your Name]

---
Generated by LLM Email Autowriter (Basic Template)"""
    
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
