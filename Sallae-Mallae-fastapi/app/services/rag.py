"""
Supabase RAG 서비스 (동환님 커스텀 DB 맞춤형)
- Gemini Embedding 모델 → 텍스트를 벡터로 변환
- Supabase pgvector (match_products) → 유사 상품 및 후회 사례 검색
"""
import logging
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)

async def get_embedding(text: str) -> list[float]:
    """Gemini Embedding API를 사용해 텍스트(캡션)를 벡터로 변환"""
    if not settings.gemini_api_key:
        return []
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={settings.gemini_api_key}"
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]}
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]["values"]


async def search(caption: str) -> str:
    """캡션을 벡터로 변환 후 Supabase에서 관련 데이터 검색"""
    if not settings.supabase_url or not settings.supabase_key:
        logger.warning("Supabase 미설정 — RAG 컨텍스트 없이 진행")
        return ""

    try:
        # 1. 캡션을 벡터 임베딩으로 변환 (동환님 핵심 로직)
        query_embedding = await get_embedding(caption)
        if not query_embedding:
            return ""

        # 2. Supabase RPC 호출 (동환님 DB의 match_products 함수 호출)
        url = f"{settings.supabase_url}/rest/v1/rpc/match_products"
        headers = {
            "apikey": settings.supabase_key,
            "Authorization": f"Bearer {settings.supabase_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query_embedding": query_embedding,
            "filter_category": "", # 만약 함수에서 카테고리 필터를 받는다면 이렇게 빈 문자열이라도 넘겨줘야 합니다.
            "match_count": settings.supabase_match_count,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            docs = resp.json()

        if not docs:
            return ""

        # 3. 검색 결과를 Gemini 프롬프트용 텍스트로 조합
        lines = ["[Supabase RAG 검색 데이터]"]
        for i, doc in enumerate(docs, 1):
            # 동환님 DB에서 반환되는 content(또는 조합된 텍스트) 사용
            content = doc.get("content") or doc.get("name") or str(doc)
            lines.append(f"{i}. {content}")

        result = "\n".join(lines)
        logger.info(f"RAG 검색 완료: {len(docs)}건")
        return result

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
             logger.error("Supabase 에러: 'match_products' 함수를 찾을 수 없습니다. DB SQL 쿼리를 다시 확인하세요.")
        else:
             logger.error(f"Supabase RAG 검색 HTTP 오류: {e}")
        return ""
    except Exception as e:
        logger.error(f"Supabase RAG 검색 오류: {e}")
        return ""