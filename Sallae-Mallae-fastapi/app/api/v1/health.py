from fastapi import APIRouter
from app.services import florence

router = APIRouter()


@router.get("/health", summary="서버 상태 확인")
async def health_check():
    return {
        "status": "ok",
        "florence_loaded": florence.is_loaded(),
    }
