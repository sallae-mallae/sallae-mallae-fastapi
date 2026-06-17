"""
/api/v1/auth/register  — 회원가입
/api/v1/auth/login     — 로그인 → JWT 발급
/api/v1/auth/me        — 내 정보 조회 (토큰 필요)
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserInfo
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["Auth"])
bearer_scheme = HTTPBearer()


@router.post("/register", response_model=TokenResponse, status_code=201, summary="회원가입")
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    if await auth_service.get_user_by_email(db, body.email):
        raise HTTPException(status_code=409, detail="이미 사용 중인 이메일입니다.")

    if len(body.password) < 6:
        raise HTTPException(status_code=422, detail="비밀번호는 6자 이상이어야 합니다.")

    user = await auth_service.create_user(db, body.email, body.password, body.nickname)
    token = auth_service.create_access_token(user.id, user.email)
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        email=user.email,
        nickname=user.nickname,
    )


@router.post("/login", response_model=TokenResponse, summary="로그인")
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await auth_service.get_user_by_email(db, body.email)
    if not user or not auth_service.verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 올바르지 않습니다.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="비활성화된 계정입니다.")

    token = auth_service.create_access_token(user.id, user.email)
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        email=user.email,
        nickname=user.nickname,
    )


@router.get("/me", response_model=UserInfo, summary="내 정보 조회")
async def me(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
):
    try:
        payload = auth_service.decode_token(credentials.credentials)
        email = payload["email"]
    except Exception:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")

    user = await auth_service.get_user_by_email(db, email)
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    return user
