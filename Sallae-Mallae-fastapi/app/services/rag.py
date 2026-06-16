"""
Supabase RAG 서비스
- Florence-2 캡션 → Supabase pgvector 검색
- 상품 정보 + 조언 + 후회 사례 검색 → Gemini 컨텍스트로 전달
"""
import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


async def search(caption: str) -> str:
    """
    캡션 텍스트로 Supabase에서 관련 상품 정보/조언/후회 사례 검색
    Supabase 미설정 시 빈 문자열 반환 (RAG 없이도 동작)
    """
    if not settings.supabase_url or not settings.supabase_key:
        logger.warning("Supabase 미설정 — RAG 컨텍스트 없이 진행")
        return ""

    try:
        # Supabase match_documents RPC 호출 (pgvector 유사도 검색)
        url = f"{settings.supabase_url}/rest/v1/rpc/match_documents"
        headers = {
            "apikey": settings.supabase_key,
            "Authorization": f"Bearer {settings.supabase_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query_text": caption,
            "match_count": settings.supabase_match_count,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()

        docs = resp.json()
        if not docs:
            return ""

        # 검색 결과를 Gemini 프롬프트용 텍스트로 조합
        lines = ["[참고 정보 — 비슷한 상품 구매 사례 및 조언]"]
        for i, doc in enumerate(docs, 1):
            content = doc.get("content", "").strip()
            if content:
                lines.append(f"{i}. {content}")

        result = "\n".join(lines)
        logger.info(f"RAG 검색 완료: {len(docs)}건")
        return result

    except Exception as e:
        logger.error(f"Supabase RAG 검색 오류: {e}")
        return ""
