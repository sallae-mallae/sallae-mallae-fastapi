import asyncio
import httpx
from app.core.config import settings
from app.services.supabase_rag import get_supabase

async def get_embedding(text: str) -> list[float]:
    """Gemini API를 사용해 텍스트를 768차원 벡터로 변환"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={settings.gemini_api_key}"
    
    # 👇 payload에 outputDimensionality 옵션 1줄 추가!
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": 768  # 핵심: 3072차원이 아닌 768차원으로 압축해서 받기
    }
    
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        
    return resp.json()["embedding"]["values"]

async def main():
    client = get_supabase()
    
    # 1. DB에 있는 모든 상품 가져오기
    print("상품 데이터를 불러옵니다...")
    resp = client.table("products").select("*").execute()
    products = resp.data
    
    if not products:
        print("DB에 상품이 없습니다.")
        return

    # 2. 각 상품마다 특징을 묶어서 벡터로 변환 후 DB 업데이트
    for p in products:
        # AI가 이해하기 쉽게 상품의 모든 특징을 하나의 문장으로 합침
        text_to_embed = f"{p['name']} {p['brand']} {p['category']} 장점: {p['pros']} 단점: {p['cons']}"
        
        print(f"[{p['name']}] 벡터 변환 및 저장 중...")
        vector = await get_embedding(text_to_embed)
        
        # Supabase DB의 embedding 컬럼에 생성된 숫자 배열 저장
        client.table("products").update({"embedding": vector}).eq("id", p["id"]).execute()
        
    print("✅ 모든 상품의 벡터 임베딩 저장이 완벽하게 끝났습니다!")

if __name__ == "__main__":
    asyncio.run(main())