import logging
import httpx
import re
from supabase import create_client, Client
from app.core.config import settings

logger = logging.getLogger(__name__)

_supabase: Client = None

def get_supabase() -> Client | None:
    global _supabase
    if _supabase is None and settings.supabase_url and settings.supabase_key:
        try:
            _supabase = create_client(
                settings.supabase_url,
                settings.supabase_key
            )
        except Exception as e:
            logger.error(f"Supabase 클라이언트 생성 실패: {e}")
    return _supabase

async def get_embedding(text: str) -> list[float]:
    """Gemini API를 사용해 텍스트를 768차원 벡터로 변환"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={settings.gemini_api_key}"
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": 768  # 3072가 아닌 768로 압축 요청
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
    return resp.json()["embedding"]["values"]

async def query_rag_context(
    caption: str,
    category: str | None
) -> str:

    client = get_supabase()
    if not client:
        return ""

    target_category = category or "Shoes"

    try:
        # 카테고리 조언 및 소비 후회 긁어오기
        advice_resp = client.table("category_advice").select("advice").eq("category", target_category).execute()
        regret_resp = client.table("purchase_regrets").select("regret").eq("category", target_category).execute()

        # -----------------------
        # 벡터 기반 시맨틱 상품 매칭 (pgvector)
        # -----------------------
        product_context = ""
        query_vector = await get_embedding(caption)
        
        # match_products RPC 호출
        rpc_response = client.rpc(
            "match_products", 
            {
                "query_embedding": query_vector,
                "match_threshold": 0.4, 
                "match_count": 1
            }
        ).execute()

        if rpc_response.data:
            p = rpc_response.data[0]
            product_context = f"""
- 매칭된 상품: {p['name']} ({p['brand']}) (유사도: {p['similarity']:.2f})
- 카테고리: {p['category']}
- 가격대: {p['price_min']:,}원 ~ {p['price_max']:,}원
- 장점: {p['pros']}
- 단점: {p['cons']}
- 추천 대상: {p['recommendation']}
"""

        # -----------------------
        # 최종 Context 생성 및 요약
        # -----------------------
        context_lines = [
            f"[Supabase RAG 검색 데이터]",
            f"- 카테고리: {target_category}",
            f"- 구매 조언: {advice_resp.data[0]['advice'] if advice_resp.data else '데이터 없음'}",
            f"- 소비 후회 사례: {regret_resp.data[0]['regret'] if regret_resp.data else '데이터 없음'}"
        ]

        if product_context:
            context_lines.append(product_context)

        final_context = "\n".join(context_lines)
        
        # [최적화] 프롬프트 터짐 방지: 1000자 이상이면 잘라내기
        if len(final_context) > 1000:
            final_context = final_context[:1000] + "\n... (데이터 요약됨)"
            
        return final_context

    except Exception as e:
        logger.error(f"Supabase RAG 데이터 조회 오류: {e}")
        return ""