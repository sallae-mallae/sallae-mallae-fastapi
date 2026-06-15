import logging
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


async def query_rag_context(
    caption: str,
    category: str | None
) -> str:

    client = get_supabase()

    if not client:
        return ""

    target_category = category or "Shoes"

    try:
        # 카테고리 조언
        advice_resp = (
            client.table("category_advice")
            .select("advice")
            .eq("category", target_category)
            .execute()
        )

        # 소비 후회
        regret_resp = (
            client.table("purchase_regrets")
            .select("regret")
            .eq("category", target_category)
            .execute()
        )

        # -----------------------
        # 상품 자동 매칭
        # -----------------------
        product_context = ""

        products_resp = (
            client.table("products")
            .select("*")
            .execute()
        )

        if products_resp.data:

            caption_lower = caption.lower()

            for p in products_resp.data:

                product_name = p["name"].lower()

                if product_name in caption_lower:

                    product_context = f"""
- 매칭된 상품: {p['name']} ({p['brand']})
- 카테고리: {p['category']}
- 가격대: {p['price_min']:,}원 ~ {p['price_max']:,}원
- 장점: {p['pros']}
- 단점: {p['cons']}
- 추천 대상: {p['recommendation']}
"""
                    break

        # -----------------------
        # 최종 Context 생성
        # -----------------------
        context_lines = [
            f"[Supabase RAG 검색 데이터]",
            f"- 카테고리: {target_category}",
            f"- 구매 조언: {advice_resp.data[0]['advice'] if advice_resp.data else '데이터 없음'}",
            f"- 소비 후회 사례: {regret_resp.data[0]['regret'] if regret_resp.data else '데이터 없음'}"
        ]

        if product_context:
            context_lines.append(product_context)

        return "\n".join(context_lines)

    except Exception as e:
        logger.error(
            f"Supabase RAG 데이터 조회 오류: {e}"
        )
        return ""