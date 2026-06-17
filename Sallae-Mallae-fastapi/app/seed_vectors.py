import asyncio
import httpx
from supabase import create_client, Client
from app.core.config import settings

# 💡 삭제된 파일 대신, 여기서 직접 Supabase 클라이언트를 만듭니다!
def get_supabase() -> Client:
    return create_client(settings.supabase_url, settings.supabase_key)

async def get_embedding(text: str) -> list[float]:
    """Gemini API를 사용해 텍스트를 768차원 벡터로 변환"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={settings.gemini_api_key}"
    
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": 768  # 핵심: 3072차원이 아닌 768차원으로 압축
    }
    
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        
    return resp.json()["embedding"]["values"]

async def main():
    client = get_supabase()
    
    print("상품 데이터를 불러옵니다...")
    resp = client.table("products").select("*").execute()
    products = resp.data
    
    if not products:
        print("DB에 상품이 없습니다. (Supabase 대시보드에서 데이터를 먼저 넣어주세요!)")
        return

    for p in products:
        # None 값이 있을 경우 에러 방지를 위해 안전하게 처리
        name = p.get('name') or ""
        brand = p.get('brand') or ""
        pros = p.get('pros') or ""
        cons = p.get('cons') or ""
        
        text_to_embed = f"{name} {brand} 장점: {pros} 단점: {cons}"
        
        print(f"[{name}] 벡터 변환 및 저장 중...")
        vector = await get_embedding(text_to_embed)
        
        client.table("products").update({"embedding": vector}).eq("id", p["id"]).execute()
        
    print("✅ 모든 상품의 벡터 임베딩 저장이 완벽하게 끝났습니다!")

if __name__ == "__main__":
    asyncio.run(main())