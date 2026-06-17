from fastapi import APIRouter
from app.services import florence
from app.core.config import settings

router = APIRouter()


@router.get("/health", summary="서버 상태 확인")
async def health_check():
    return {
        "status": "ok",
        "florence_loaded": florence.is_loaded(),
        "rag_configured": bool(settings.supabase_url and settings.supabase_key),
        "gemini_configured": bool(settings.gemini_api_key),
    }
