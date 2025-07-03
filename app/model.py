"""
Model handler for LLM inference using vLLM OpenAI-compatible API.
Adapted from the prototype notebook's ModelCall class.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import Config
from .prompt_template import EmailPromptTemplate

logger = logging.getLogger(__name__)

class EmailGeneratorModel:
    """
    Handles LLM model interactions for email generation.
    Downloads and runs Qwen 7B model directly.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.prompt_template = EmailPromptTemplate()
        self.model = None
        self.tokenizer = None
        self._initialized = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
    
    def _load_model(self):
        """Download and load Qwen 7B model"""
        if self._initialized:
            return
            
        try:
            logger.info("Loading Qwen 7B model...")
            
            # Use a smaller quantized version for better performance
            model_name = "Qwen/Qwen2.5-7B-Instruct"
            
            logger.info(f"Downloading tokenizer from {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            logger.info(f"Downloading model from {model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            self.model.eval()
            self._initialized = True
            logger.info(f"✅ Qwen 7B model loaded successfully on {self.device}!")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen 7B model: {e}")
            raise
    
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
        Generate an email using the actual Qwen 7B model.
        
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
            # Load model if not already loaded
            self._load_model()
            
            # Set default parameters
            temperature = temperature or self.config.DEFAULT_TEMPERATURE
            top_p = top_p or self.config.DEFAULT_TOP_P
            max_tokens = max_tokens or self.config.TOKEN_LIMITS.get(length, self.config.DEFAULT_MAX_TOKENS)
            
            logger.info(f"🚀 Generating email with Qwen 7B - Intent: '{intent[:50]}...', Tone: {tone}, Length: {length}")
            
            # Generate prompt using our template
            messages = self.prompt_template.generate_email_prompt(intent, tone, length)
            
            # Convert messages to chat format for Qwen
            chat_text = ""
            for msg in messages:
                if msg["role"] == "system":
                    chat_text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
                elif msg["role"] == "user":
                    chat_text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            chat_text += "<|im_start|>assistant\n"
            
            # Tokenize
            inputs = self.tokenizer(
                chat_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            ).strip()
            
            # Clean up the response
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0].strip()
            
            logger.info(f"✅ Successfully generated email with Qwen 7B ({len(response)} characters)")
            
            # Add model info footer
            response += "\n\n---\nGenerated by Qwen 7B via LLM Email Autowriter"
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating email with Qwen 7B: {e}")
            raise Exception(f"Failed to generate email with Qwen 7B: {str(e)}")
    
    
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
        Check if the Qwen 7B model is loaded and responsive.
        
        Returns:
            Dictionary with health status information
        """
        try:
            # Try to load the model
            self._load_model()
            
            # Try a simple generation
            test_response = self.generate_email(
                intent="test email",
                tone="Professional", 
                length="Short",
                max_tokens=50
            )
            
            return {
                "status": "healthy",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "device": self.device,
                "initialized": self._initialized,
                "test_response_length": len(test_response)
            }
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "device": self.device,
                "initialized": self._initialized
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "device": self.device,
            "model_type": "Direct Transformers (No API)",
            "initialized": self._initialized,
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
