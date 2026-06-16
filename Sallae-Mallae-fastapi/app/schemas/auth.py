from pydantic import BaseModel, EmailStr


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    nickname: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    email: str
    nickname: str | None = None


class UserInfo(BaseModel):
    id: int
    email: str
    nickname: str | None = None
    is_active: bool

    class Config:
        from_attributes = True
