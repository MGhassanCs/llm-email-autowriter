"""
Gradio UI for the LLM Email Autowriter.
Provides an interactive interface with tone and length selection.
"""
import gradio as gr
import asyncio
import logging
from typing import Tuple, Optional
from .model import get_model
from .config import Config
from .prompt_template import EmailPromptTemplate

logger = logging.getLogger(__name__)

class EmailGeneratorUI:
    """Gradio UI wrapper for the email generator"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.model = get_model()
        self.prompt_template = EmailPromptTemplate()
    
    def generate_email_ui(self, intent: str, tone: str, length: str) -> Tuple[str, str]:
        """
        Generate email for the Gradio UI.
        
        Args:
            intent: User's email intent
            tone: Selected tone
            length: Selected length
            
        Returns:
            Tuple of (generated_email, status_message)
        """
        if not intent.strip():
            return "", "❌ Please enter an email intent."
        
        try:
            logger.info(f"UI: Generating email with intent='{intent[:50]}...', tone={tone}, length={length}")
            
            # Generate email
            email_content = self.model.generate_email(
                intent=intent.strip(),
                tone=tone,
                length=length
            )
            
            if email_content.startswith("Error"):
                return "", f"❌ {email_content}"
            
            status_msg = f"✅ Email generated successfully! (Tone: {tone}, Length: {length})"
            return email_content, status_msg
            
        except Exception as e:
            logger.error(f"UI error generating email: {e}")
            return "", f"❌ Error: {str(e)}"
    
    def get_example_intent(self, example_text: str) -> str:
        """Return the selected example intent"""
        return example_text
    
    def clear_all(self) -> Tuple[str, str, str, str]:
        """Clear all fields"""
        return "", "Professional", "Medium", ""
    
    def create_interface(self) -> gr.Interface:
        """Create and configure the Gradio interface"""
        
        # Custom CSS for better styling
        custom_css = """
        .gradio-container {
            max-width: 900px !important;
            margin: auto !important;
        }
        .main-header {
            text-align: center;
            margin-bottom: 20px;
        }
        .example-box {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .status-success {
            color: #28a745;
        }
        .status-error {
            color: #dc3545;
        }
        """
        
        with gr.Blocks(css=custom_css, title="LLM Email Autowriter") as interface:
            
            # Header
            gr.Markdown(
                """
                # 📧 LLM Email Autowriter
                
                Generate professional emails from simple intents using AI. Just describe what you want to say, 
                choose your preferred tone and length, and get a well-formatted email ready to send!
                """,
                elem_classes=["main-header"]
            )
            
            with gr.Row():
                # Left column - Input
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Email Intent")
                    
                    intent_input = gr.Textbox(
                        label="What do you want the email to say?",
                        placeholder="e.g., Ask professor for deadline extension due to personal issues",
                        lines=3,
                        max_lines=5
                    )
                    
                    with gr.Row():
                        tone_dropdown = gr.Dropdown(
                            label="📨 Tone",
                            choices=self.config.TONE_OPTIONS,
                            value="Professional",
                            info="Choose the tone of your email"
                        )
                        
                        length_dropdown = gr.Dropdown(
                            label="📏 Length", 
                            choices=self.config.LENGTH_OPTIONS,
                            value="Medium",
                            info="Choose email length"
                        )
                    
                    with gr.Row():
                        generate_btn = gr.Button("✨ Generate Email", variant="primary", scale=2)
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                    
                    # Status message
                    status_output = gr.Markdown("", elem_classes=["status"])
                
                # Right column - Examples
                with gr.Column(scale=1):
                    gr.Markdown("### 💡 Example Intents")
                    
                    example_intents = self.prompt_template.get_example_intents()
                    
                    # Create example buttons
                    example_buttons = []
                    for i, example in enumerate(example_intents[:8]):  # Limit to 8 examples
                        btn = gr.Button(
                            example,
                            size="sm",
                            variant="secondary"
                        )
                        example_buttons.append(btn)
                        
                        # Connect button to update intent input
                        btn.click(
                            fn=lambda x=example: x,
                            outputs=intent_input
                        )
            
            # Output section
            gr.Markdown("### 📄 Generated Email")
            email_output = gr.Textbox(
                label="Your Email",
                lines=15,
                max_lines=25,
                show_copy_button=True,
                placeholder="Your generated email will appear here..."
            )
            
            # Model info section (collapsible)
            with gr.Accordion("ℹ️ Model Information", open=False):
                model_info = self.model.get_model_info()
                gr.Markdown(f"""
                **Model:** {model_info['model_name']}  
                **Model Type:** {model_info['model_type']}  
                **Device:** {model_info['device']}  
                **Available Tones:** {', '.join(model_info['available_tones'])}  
                **Available Lengths:** {', '.join(model_info['available_lengths'])}  
                **Token Limits:** {model_info['token_limits']}
                """)
            
            # Event handlers
            generate_btn.click(
                fn=self.generate_email_ui,
                inputs=[intent_input, tone_dropdown, length_dropdown],
                outputs=[email_output, status_output]
            )
            
            clear_btn.click(
                fn=self.clear_all,
                outputs=[intent_input, tone_dropdown, length_dropdown, email_output]
            )
            
            # Footer
            gr.Markdown(
                """
                ---
                <div style="text-align: center; color: #666; font-size: 0.9em;">
                    Powered by vLLM and Gradio | LLM Email Autowriter v1.0
                </div>
                """,
                elem_classes=["footer"]
            )
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        interface = self.create_interface()
        
        # Default launch parameters
        launch_params = {
            "server_name": self.config.GRADIO_HOST,
            "server_port": self.config.GRADIO_PORT,
            "share": self.config.GRADIO_SHARE,
            "show_error": True,
            "quiet": False
        }
        
        # Override with any provided kwargs
        launch_params.update(kwargs)
        
        logger.info(f"Launching Gradio UI on {launch_params['server_name']}:{launch_params['server_port']}")
        
        try:
            interface.launch(**launch_params)
        except Exception as e:
            logger.error(f"Failed to launch Gradio UI: {e}")
            raise


def create_ui(config: Optional[Config] = None) -> EmailGeneratorUI:
    """Factory function to create UI instance"""
    return EmailGeneratorUI(config)


def launch_ui(config: Optional[Config] = None, **kwargs):
    """Quick launch function for the UI"""
    ui = create_ui(config)
    ui.launch(**kwargs)
