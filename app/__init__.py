"""
LLM Email Autowriter Application Package

A modular, production-ready application for generating professional emails
using local LLMs through vLLM with transformers fallback.

Components:
- config: Configuration management
- model: LLM model handling and inference
- prompt_template: Email prompt generation and templating
- main: FastAPI backend application
- gradio_ui: Gradio frontend interface
"""

from .config import Config
from .model import EmailGeneratorModel, get_model
from .prompt_template import EmailPromptTemplate

__version__ = "1.0.0"
__author__ = "LLM Email Autowriter Team"
__description__ = "AI-powered email generation using vLLM and Gradio"

__all__ = [
    "Config",
    "EmailGeneratorModel", 
    "get_model",
    "EmailPromptTemplate"
]
