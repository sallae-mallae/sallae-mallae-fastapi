"""
Supabase RAG 서비스
실제 DB 구조:
  - products           : 상품 정보 + embedding (pgvector)
  - category_advice    : 카테고리별 구매 조언
  - purchase_regrets   : 카테고리별 후회 사례

처리 흐름:
  캡션 → Gemini Embedding API → 벡터
  벡터 → Supabase match_products RPC (pgvector 유사도 검색)
  category + Supabase REST → category_advice, purchase_regrets
"""
import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

GEMINI_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/gemini-embedding-001:embedContent?key={api_key}"
)


async def _get_embedding(text: str) -> list[float]:
    """캡션 텍스트 → Gemini 임베딩 벡터"""
    url = GEMINI_EMBED_URL.format(api_key=settings.gemini_api_key)
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]},
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
    return resp.json()["embedding"]["values"]


async def _match_products(embedding: list[float], match_count: int) -> list[dict]:
    """pgvector 유사도 검색 — 비슷한 상품 반환"""
    url = f"{settings.supabase_url}/rest/v1/rpc/match_products"
    headers = {
        "apikey": settings.supabase_key,
        "Authorization": f"Bearer {settings.supabase_key}",
        "Content-Type": "application/json",
    }
    payload = {"query_embedding": embedding, "match_count": match_count}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
    return resp.json() or []


async def _get_category_advice(category: str) -> str:
    """카테고리별 구매 조언"""
    url = f"{settings.supabase_url}/rest/v1/category_advice"
    headers = {
        "apikey": settings.supabase_key,
        "Authorization": f"Bearer {settings.supabase_key}",
    }
    params = {"select": "advice", "category": f"eq.{category}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
    rows = resp.json()
    return rows[0]["advice"] if rows else ""


async def _get_purchase_regrets(category: str) -> str:
    """카테고리별 후회 사례"""
    url = f"{settings.supabase_url}/rest/v1/purchase_regrets"
    headers = {
        "apikey": settings.supabase_key,
        "Authorization": f"Bearer {settings.supabase_key}",
    }
    params = {"select": "regret", "category": f"eq.{category}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
    rows = resp.json()
    return rows[0]["regret"] if rows else ""


async def search(caption: str, category: str | None = None) -> str:
    """
    캡션 + 카테고리로 Supabase에서 RAG 컨텍스트 조합
    Supabase 미설정 시 빈 문자열 반환
    """
    if not settings.supabase_url or not settings.supabase_key:
        logger.warning("Supabase 미설정 — RAG 없이 진행")
        return ""

    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY 없음 — 임베딩 생성 불가")
        return ""

    parts = []

    try:
        # 1. 캡션 → 임베딩 → 유사 상품 검색
        embedding = await _get_embedding(caption)
        products = await _match_products(embedding, settings.supabase_match_count)

        if products:
            parts.append("[매칭된 유사 상품]")
            for p in products:
                # match_products RPC는 name/brand/pros 등을 content 문자열 하나로 합쳐서 반환
                content = p.get("content", "").strip()
                sim     = p.get("similarity", "")
                sim_str = f" (유사도: {sim:.2f})" if isinstance(sim, float) else ""
                if content:
                    parts.append(f"- {content}{sim_str}")

        logger.info(f"RAG 유사 상품 {len(products)}건 검색 완료")

    except Exception as e:
        logger.error(f"match_products 오류: {e}")

    # 2. 카테고리별 조언 + 후회 사례 (category 있을 때만)
    if category:
        try:
            advice = await _get_category_advice(category)
            if advice:
                parts.append(f"\n[{category} 카테고리 구매 조언]")
                parts.append(f"- {advice}")
        except Exception as e:
            logger.warning(f"category_advice 조회 실패: {e}")

        try:
            regret = await _get_purchase_regrets(category)
            if regret:
                parts.append(f"\n[{category} 소비 후회 사례]")
                parts.append(f"- {regret}")
        except Exception as e:
            logger.warning(f"purchase_regrets 조회 실패: {e}")

    result = "\n".join(parts)
    if result:
        logger.info(f"RAG 컨텍스트 구성 완료 ({len(result)}자)")
        logger.info(f"=== RAG 컨텍스트 내용 ===\n{result}\n========================")
    return result
