# Use lightweight Python 3.10 base image
FROM python:3.10-slim

# Set environment variables to make Python behave well in containers
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install required system packages
# - curl: for testing endpoints or health checks
# - build-essential: needed for compiling Python packages like tokenizers
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first to take advantage of Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and scripts
COPY app/ ./app/

# Optional: copy .env file if your app loads it at runtime
COPY .env .

# Optional: include Lightning scripts if used to launch the app
COPY lightning_app.py .
COPY lightning_launch.py .

# Expose FastAPI and Gradio ports
EXPOSE 8000
EXPOSE 7860

# Set the default command to run the app
# Update this if your app is started from a different script
CMD ["python", "-m", "app.main"]
