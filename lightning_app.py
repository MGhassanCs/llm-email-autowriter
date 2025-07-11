"""
Lightning AI application for LLM Email Autowriter
Simplified version for Lightning AI platform deployment
"""
import lightning as L
import subprocess
import sys
import os


class EmailAutowriterWork(L.LightningWork):
    """
    Lightning Work for the Qwen 7B email autowriter application
    """
    
    def __init__(self):
        # Use GPU for Qwen 7B model
        super().__init__(cloud_compute=L.CloudCompute("gpu"))
        self.ready = False
    
    def run(self):
        """Set up and run the Gradio application"""
        
        # Install dependencies
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        
        # Set environment variables for production
        os.environ["ENVIRONMENT"] = "production"
        os.environ["DEBUG"] = "false"
        os.environ["VLLM_HOST"] = "localhost"  # Mock vLLM for now
        os.environ["VLLM_PORT"] = "8000"
        
        # Use the optimized launcher
        try:
            import subprocess
            import sys
            
            # Run the optimized launcher
            result = subprocess.run([
                sys.executable, "lightning_launch.py"
            ], check=True)
            
            self.ready = True
            
        except Exception as e:
            print(f"Error launching app: {e}")
            self.ready = True


class LLMEmailAutowriterApp(L.LightningApp):
    """
    Main Lightning App for the email autowriter
    """
    
    def __init__(self):
        super().__init__()
        self.email_work = EmailAutowriterWork()
    
    def configure_layout(self):
        """Configure the Lightning UI layout"""
        return {
            "name": "LLM Email Autowriter",
            "content": self.email_work
        }


# Entry point for Lightning AI
app = LLMEmailAutowriterApp()


if __name__ == "__main__":
    # For local testing
    app.run()
