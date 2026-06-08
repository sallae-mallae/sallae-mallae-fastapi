from fastapi import APIRouter
from app.api.v1 import health, analyze, history

router = APIRouter()

router.include_router(health.router, tags=["Health"])
router.include_router(analyze.router, tags=["Analyze"])
router.include_router(history.router, tags=["History"])
