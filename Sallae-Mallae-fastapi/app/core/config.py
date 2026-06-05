from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "살래말래 API"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Florence-2
    florence_model_id: str = "microsoft/Florence-2-base"

    # 이미지 저장
    save_images: bool = False

    # DB
    database_url: str = "sqlite+aiosqlite:///./sallae_mallae.db"

    class Config:
        env_file = ".env"


settings = Settings()
