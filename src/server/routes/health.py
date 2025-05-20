# routes/health.py
from fastapi import APIRouter, HTTPException, Depends
from starlette.responses import RedirectResponse
from logging_config import logger
from core.dependencies import get_settings  # Updated import

router = APIRouter(prefix="/v1", tags=["health"])

@router.get("/health")
async def health_check(settings=Depends(get_settings)):
    return {"status": "healthy", "model": settings.llm_model_name}

@router.get("/", include_in_schema=False)
async def home():
    return RedirectResponse(url="/docs")

