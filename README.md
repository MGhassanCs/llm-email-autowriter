# LLM Email Autowriter Readme

Welcome to the LLM Email Autowriter project - an application that helps you generate professional, well-structured emails using AI. Just provide a brief description of your email's intent, and the system will craft a complete email based on the tone and length you choose.

## Features

- **FastAPI Backend**: A RESTful API with a `/generate` endpoint to produce emails based on provided intents.
- **Gradio Frontend**: An intuitive interface for users to interact and experience real-time email generation.
- **vLLM Compatible**: Uses local language models like Qwen or Mistral through vLLM.
- **Modular Architecture**: Separate components for API, model handling, and UI ensure easy maintenance and scalability.
- **Dockerized**: Ready for containerization for consistent environments.

## Getting Started

### Prerequisites

- Python 3.10+
- GPU (recommended for Qwen 7B model)
- Lightning AI account (for cloud deployment)
- Docker (for local containerized deployment)

### Installation

1. **Clone the repository**
    ```bash
    git clone <repository-url>
    cd llm_email_autowriter
    ```

2. **Create a virtual environment**
    ```bash
    python -m venv env
    source env/bin/activate  # or `env\Scripts\activate` on Windows
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application**
    - **Lightning AI (Recommended for Cloud)**
      ```bash
      lightning run app lightning_app.py --cloud
      ```
    - **Local Development**
      ```bash
      python local_dev.py
      ```
    - **Docker (Optional)**
      ```bash
      docker build -t llm-email-autowriter .
      docker run -p 7860:7860 llm-email-autowriter
      ```

## Configuration

Environment variables are managed via the `.env` file:

- **ENVIRONMENT**: Set to `development` or `production`.
- **DEBUG**: Set to `true` or `false`.
- **MODEL_NAME**: Define the model used for generation.
- **VLLM_HOST** and **VLLM_PORT**: Define the host and port where vLLM is running.

## Usage

- **API**: Use the `/generate` endpoint, providing intent, tone, and length as parameters.
- **UI**: Access the Gradio interface to interactively generate emails.

## Contributions

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## License

MIT License. See `LICENSE` for more information.
