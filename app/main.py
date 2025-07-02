from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import get_model
from .config import Config

app = FastAPI()
config = Config()
model = get_model()

# Define data model for API
class EmailRequest(BaseModel):
    intent: str
    tone: str = "Professional"
    length: str = "Medium"

@app.get("/")
async def root():
    return {"message": "Welcome to LLM Email Autowriter API!"}

@app.get("/health")
async def health_check():
    health_status = model.health_check()
    if health_status["status"] == "healthy":
        return health_status
    else:
        raise HTTPException(status_code=503, detail=health_status)

@app.get("/model-info")
async def model_info():
    return model.get_model_info()

@app.post("/generate")
async def generate_email(request: EmailRequest):
    try:
        email_content = await model.generate_email_async(
            intent=request.intent,
            tone=request.tone,
            length=request.length
        )
        return {"email": email_content}
    except Exception as e:
        detail = f"Error generating email: {str(e)}"
        raise HTTPException(status_code=500, detail=detail)
