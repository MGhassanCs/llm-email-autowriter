#!/usr/bin/env python3
"""
Local Development Script for LLM Email Autowriter

Simple launcher for local development and testing.
"""

def main():
    """Launch for local development"""
    try:
        print("🚀 Starting LLM Email Autowriter (Local Development)...")
        
        # Import and run
        from app.gradio_ui import create_ui
        from app.config import Config
        
        config = Config()
        config.GRADIO_SHARE = True  # Enable public sharing
        config.GRADIO_HOST = "0.0.0.0"  # Allow external access
        config.GRADIO_PORT = 7860
        
        ui = create_ui(config)
        print("✅ Launching Gradio interface with public link...")
        
        ui.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,  # Create public link
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you've installed dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
