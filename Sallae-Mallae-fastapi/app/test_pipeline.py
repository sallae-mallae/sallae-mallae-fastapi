import asyncio

from app.services import florence
from app.services.supabase_rag import query_rag_context
from app.core.config import settings


async def main():
    # Florence 로드
    florence.load_model(settings.florence_model_id)

    # 이미지 분석
    caption = florence.test_local_image("test.png")

    print("\n===== FLORENCE CAPTION =====")
    print(caption)

    # Supabase RAG 조회
    rag = await query_rag_context(
        caption=caption,
        category="Shoes"
    )

    print("\n===== SUPABASE RAG =====")
    print(rag)

    print("\n===== TEST END =====")


if __name__ == "__main__":
    asyncio.run(main())