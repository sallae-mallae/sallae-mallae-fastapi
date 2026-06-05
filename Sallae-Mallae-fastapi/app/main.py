from fastapi import FastAPI
from app.core.config import settings
from app.api.v1.router import router as v1_router

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
)

app.include_router(v1_router, prefix=settings.api_v1_prefix)


@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.app_name}"}
