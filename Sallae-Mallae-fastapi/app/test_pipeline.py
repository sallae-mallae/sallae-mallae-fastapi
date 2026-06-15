import asyncio
from app.services import florence
from app.services.supabase_rag import query_rag_context
from app.services.llm import judge_purchase
from app.schemas.analyze import AnalyzeRequest, ContextInput, Verdict # ContextInput으로 수정!
from app.core.config import settings

async def main():
    # 1. 모델 로드
    florence.load_model(settings.florence_model_id)

    # 2. 이미지 캡션 생성
    image_path = "app/test.png"
    caption = florence.test_local_image(image_path)
    print(f"\n===== FLORENCE CAPTION =====\n{caption}")

    # 3. RAG 데이터 조회
    rag_context = await query_rag_context(
        caption=caption,
        category="Shoes"
    )
    print(f"\n===== SUPABASE RAG =====\n{rag_context}")

    # 4. 최종 판단 (Gemini)
    # 스키마에 맞게 ContextInput 객체 사용
    test_request = AnalyzeRequest(
        image_base64="dummy", 
        question="이 신발 살 만한가요?",
        context=ContextInput(
            category="Shoes",
            price="280,000원",
            purpose="데일리 운동화",
            criteria=["가격", "착화감"]
        ),
        save_image=False
    )
    
    print("\n===== GEMINI JUDGEMENT (판단 중...) =====")
    final_result = await judge_purchase(caption, rag_context, test_request)
    
    print(f"판단: {final_result.verdict}")
    print(f"잔소리: {final_result.reason}")
    print("\n===== TEST END =====")

if __name__ == "__main__":
    asyncio.run(main())