import logging
from supabase import create_client, Client
from app.core.config import settings

logger = logging.getLogger(__name__)

_supabase: Client = None

def get_supabase() -> Client | None:
    """Supabase 클라이언트 싱글톤 로드"""
    global _supabase
    if _supabase is None and settings.supabase_url and settings.supabase_key:
        try:
            _supabase = create_client(settings.supabase_url, settings.supabase_key)
        except Exception as e:
            logger.error(f"Supabase 클라이언트 생성 실패: {e}")
    return _supabase

async def query_rag_context(caption: str, category: str | None) -> str:
    """
    Florence-2 캡션과 사용자가 입력한 카테고리를 바탕으로
    Supabase에서 족보 데이터를 조회하여 하나의 문맥(Context) 문장으로 만듭니다.
    """
    client = get_supabase()
    if not client:
        return ""

    # 검색 기준 키워드 (사용자 입력 대분류가 없으면 기본값 Shoes)
    target_category = category or "Shoes"
    
    try:
        # 1. 카테고리 조언 테이블 조회
        advice_resp = client.table("category_advice").select("advice").eq("category", target_category).execute()
        # 2. 후회 사례 테이블 조회
        regret_resp = client.table("purchase_regrets").select("regret").eq("category", target_category).execute()
        
        # 3. 특정 상품명 매칭 (Florence-2 캡션 텍스트에 상품명이 포함되어 있는지 검사)
        product_context = ""
        # 꿀팁: DB에 있는 상품명 리스트를 순회하며 캡션에 들어있는지 확인합니다.
        known_products = ["New Balance 993", "Nike Air Force 1", "Apple Watch", "Galaxy Watch", "MX Master 3S"]
        
        for p_name in known_products:
            if p_name.lower() in caption.lower():
                p_resp = client.table("products").select("*").eq("name", p_name).execute()
                if p_resp.data:
                    p = p_resp.data[0]
                    product_context = f"""
- 매칭된 구체적 상품: {p['name']} ({p['brand']})
- 해당 상품 가격대: {p['price_min']:,}원 ~ {p['price_max']:,}원
- 상품 장점: {p['pros']}
- 상품 단점: {p['cons']}
- 추천 타겟: {p['recommendation']}"""
                    break

        # 텍스트로 예쁘게 조립
        context_lines = [
            f"\n[Supabase RAG 검색 데이터 (카테고리: {target_category})]",
            f"- 전문가의 구매 조언: {advice_resp.data[0]['advice'] if advice_resp.data else '데이터 없음'}",
            f"- 이 카테고리의 흔한 소비 후회: {regret_resp.data[0]['regret'] if regret_resp.data else '데이터 없음'}"
        ]
        if product_context:
            context_lines.append(product_context)

        return "\n".join(context_lines)

    except Exception as e:
        logger.error(f"Supabase RAG 데이터 조회 중 오류 발생: {e}")
        return ""