from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "살래말래 API"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # 이미지 저장
    save_images: bool = False

    # YOLO 모델 (라벨링/파인튜닝용)
    # yolo11n.pt (가벼움) / yolo11s.pt / yolo11m.pt (정확도 높음)
    yolo_model: str = "yolo11n.pt"

    # DB
    database_url: str = "sqlite+aiosqlite:///./sallae_mallae.db"

    class Config:
        env_file = ".env"


settings = Settings()
