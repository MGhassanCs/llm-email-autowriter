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
    Simplified Lightning Work for the email autowriter application
    """
    
    def __init__(self):
        super().__init__(cloud_compute=L.CloudCompute("cpu-medium"))
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
        
        # Import after installing dependencies
        try:
            from app.gradio_ui import create_ui
            from app.config import Config
            
            # Create configuration for Lightning
            config = Config()
            config.GRADIO_SHARE = False
            config.GRADIO_HOST = "0.0.0.0"
            config.GRADIO_PORT = 7860
            
            # Create and launch the UI
            ui = create_ui(config)
            self.ready = True
            
            # Launch with Lightning-friendly settings
            ui.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                show_error=True,
                quiet=False
            )
            
        except Exception as e:
            print(f"Error launching app: {e}")
            # Fallback: create a simple demo
            import gradio as gr
            
            def simple_demo(intent, tone, length):
                return f"Demo Email:\n\nSubject: {intent}\n\nDear Recipient,\n\nThis is a demo email generated with {tone} tone and {length} length based on your intent: {intent}\n\nBest regards,\n[Your Name]"
            
            demo = gr.Interface(
                fn=simple_demo,
                inputs=[
                    gr.Textbox(label="Email Intent", placeholder="e.g., Ask for meeting"),
                    gr.Dropdown(["Professional", "Friendly", "Formal"], value="Professional", label="Tone"),
                    gr.Dropdown(["Short", "Medium", "Long"], value="Medium", label="Length")
                ],
                outputs=gr.Textbox(label="Generated Email", lines=10),
                title="LLM Email Autowriter (Demo Mode)",
                description="Generate professional emails from simple intents."
            )
            
            self.ready = True
            demo.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False
            )


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
