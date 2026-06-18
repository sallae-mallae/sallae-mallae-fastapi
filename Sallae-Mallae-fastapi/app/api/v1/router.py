from fastapi import APIRouter
from app.api.v1 import auth, health, analyze, history, labeling, chat

router = APIRouter()

router.include_router(auth.router)
router.include_router(health.router, tags=["Health"])
router.include_router(analyze.router, tags=["Analyze"])
router.include_router(history.router, tags=["History"])
router.include_router(labeling.router, tags=["Labeling"])
router.include_router(chat.router)
