"""
Model handler for LLM inference using vLLM.
Adapted from the prototype notebook's ModelCall class.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
import torch
from .config import Config
from .prompt_template import EmailPromptTemplate

logger = logging.getLogger(__name__)

class EmailGeneratorModel:
    """
    Handles LLM model interactions for email generation.
    Uses vLLM for efficient inference as shown in the prototype.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.prompt_template = EmailPromptTemplate()
        self.llm = None
        self._initialized = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
    
    def _load_model(self):
        """Load vLLM model"""
        if self._initialized:
            return
            
        try:
            from vllm import LLM, SamplingParams
            logger.info("Loading model with vLLM...")
            
            # Use the model specified in config
            model_name = self.config.MODEL_NAME
            logger.info(f"Loading model: {model_name}")
            
            # Initialize vLLM
            self.llm = LLM(
                model=model_name,
                dtype=self.config.MODEL_DTYPE,
                trust_remote_code=True
            )
            
            self._initialized = True
            logger.info(f"✅ vLLM model {model_name} loaded successfully!")
            
        except ImportError:
            logger.error("vLLM not available, falling back to transformers")
            self._load_model_transformers()
        except Exception as e:
            logger.error(f"Failed to load vLLM model: {e}")
            logger.info("Attempting fallback to transformers...")
            self._load_model_transformers()
    
    def _load_model_transformers(self):
        """Fallback to transformers if vLLM fails"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            logger.info("Loading model with transformers (fallback)...")
            
            model_name = self.config.MODEL_NAME
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Load model with proper device handling
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                # For CPU, load without device_map
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(self.device)
                
            self.model.eval()  #puts model in evaluation mode, no gradient needed
            self._initialized = True
            logger.info(f"✅ Transformers model {model_name} loaded successfully on {self.device}!")
            
        except Exception as e:
            logger.error(f"Failed to load model with transformers: {e}")
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
        Generate an email using vLLM or transformers fallback.
        
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
            
            logger.info(f"🚀 Generating email - Intent: '{intent[:50]}...', Tone: {tone}, Length: {length}")
            
            # Generate prompt using our template
            messages = self.prompt_template.generate_email_prompt(intent, tone, length)
            
            # Use vLLM if available
            if self.llm is not None:
                return self._generate_with_vllm(messages, temperature, top_p, max_tokens)
            else:
                return self._generate_with_transformers(messages, temperature, top_p, max_tokens)
            
        except Exception as e:
            logger.error(f"❌ Error generating email: {e}")
            raise Exception(f"Failed to generate email: {str(e)}")
    
    def _generate_with_vllm(self, messages, temperature, top_p, max_tokens):
        """Generate using vLLM (like the prototype)"""
        try:
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
            
            # Use vLLM's chat interface
            outputs = self.llm.chat(messages, sampling_params)
            response = outputs[0].outputs[0].text
            
            logger.info(f"✅ Successfully generated email with vLLM ({len(response)} characters)")
            return response
            
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            raise
    
    def _generate_with_transformers(self, messages, temperature, top_p, max_tokens):
        """Fallback generation using transformers"""
        try:
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
            
            # Clean up the response, cleans trailing assistant message end tokens.
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0].strip()
            
            logger.info(f"✅ Successfully generated email with transformers ({len(response)} characters)")
            return response
            
        except Exception as e:
            logger.error(f"Transformers generation failed: {e}")
            raise
    
    
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
                "model": self.config.MODEL_NAME,
                "device": self.device,
                "initialized": self._initialized,
                "test_response_length": len(test_response)
            }
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.config.MODEL_NAME,
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
            "model_name": self.config.MODEL_NAME,
            "device": self.device,
            "model_type": "vLLM with Transformers Fallback",
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
