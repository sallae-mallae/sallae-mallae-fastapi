from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "살래말래 API"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Florence-2
    florence_model_id: str = "microsoft/Florence-2-base"

    # Supabase RAG
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_table: str = "products"      # RAG 검색 테이블
    supabase_match_count: int = 5         # 검색 결과 개수

    # 이미지 저장
    save_images: bool = False

    # YOLO 모델 (라벨링/파인튜닝용)
    yolo_model: str = "yolo11n.pt"

    # DB
    database_url: str = "sqlite+aiosqlite:///./sallae_mallae.db"

    # JWT
    jwt_secret_key: str = "sallae-mallae-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 7일

    class Config:
        env_file = ".env"


settings = Settings()
