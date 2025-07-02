"""
Lightning AI application for LLM Email Autowriter
Configures the app for deployment on Lightning AI platform
"""
import lightning as L
from lightning.app.api import Post
from lightning.app.components import ServeGradio, BuildConfig
import subprocess
import sys
import os
from pathlib import Path

class LLMEmailAutowriterWork(L.LightningWork):
    """
    Lightning Work for running the email autowriter application
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ready = False
    
    def run(self):
        """Run the application setup and start the Gradio interface"""
        
        # Install dependencies
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        
        # Set up environment
        os.environ["ENVIRONMENT"] = "production"
        os.environ["DEBUG"] = "false"
        os.environ["GRADIO_SHARE"] = "false"  # Lightning handles sharing
        
        # Import and launch the UI
        from app.gradio_ui import launch_ui
        from app.config import Config
        
        config = Config()
        config.GRADIO_SHARE = False  # Lightning handles this
        
        # Mark as ready
        self.ready = True
        
        # Launch the Gradio interface
        launch_ui(
            config=config,
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )


class EmailAutowriterGradioServe(ServeGradio):
    """
    Lightning Gradio component for the email autowriter
    """
    
    def __init__(self):
        super().__init__(
            LLMEmailAutowriterWork,
            cloud_build_config=BuildConfig(
                requirements=["requirements.txt"],
                dockerfile="Dockerfile"
            )
        )
    
    def configure_layout(self):
        """Configure the layout for Lightning AI dashboard"""
        return {
            "name": "LLM Email Autowriter",
            "content": self.work.url + "/gradio"
        }


class LLMEmailAutowriterAPI(L.LightningWork):
    """
    Lightning Work for the FastAPI backend
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ready = False
    
    def run(self):
        """Run the FastAPI server"""
        
        # Install dependencies
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        
        # Set up environment
        os.environ["ENVIRONMENT"] = "production"
        os.environ["DEBUG"] = "false"
        
        # Mark as ready
        self.ready = True
        
        # Start FastAPI server
        subprocess.run([
            "uvicorn",
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--workers", "1"
        ])
    
    @Post("/generate")
    def generate_email(self, intent: str, tone: str = "Professional", length: str = "Medium"):
        """API endpoint for email generation"""
        if not self.ready:
            return {"error": "Service not ready"}
        
        try:
            from app.model import get_model
            model = get_model()
            email_content = model.generate_email(
                intent=intent,
                tone=tone,
                length=length
            )
            return {"email": email_content}
        except Exception as e:
            return {"error": str(e)}


class LLMEmailAutowriterApp(L.LightningApp):
    """
    Main Lightning App that combines the API and UI
    """
    
    def __init__(self):
        super().__init__()
        
        # Create the main components
        self.gradio_serve = EmailAutowriterGradioServe()
        self.api = LLMEmailAutowriterAPI(
            cloud_compute=L.CloudCompute("gpu-fast")  # Use GPU for model inference
        )
    
    def configure_layout(self):
        """Configure the Lightning dashboard layout"""
        return [
            {
                "name": "LLM Email Autowriter - UI",
                "content": self.gradio_serve
            },
            {
                "name": "API Documentation", 
                "content": self.api.url + "/docs"
            }
        ]


# Entry point for Lightning AI
app = LLMEmailAutowriterApp()


if __name__ == "__main__":
    # For local testing
    import lightning as L
    L.LightningApp(LLMEmailAutowriterApp()).run()
