import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


JWT_ALGORITHM = "HS256"
bearer_scheme = HTTPBearer(auto_error=False)
# B10：token 同时以 httpOnly Cookie 下发（前端不再用 localStorage）。
# SameSite=Strict 抵御 CSRF；生产环境默认启用 Secure，本地 http
# 开发环境可显式设置 COOKIE_SECURE=false。
COOKIE_NAME = "agent_token"


def set_auth_cookie(response, token: str) -> None:
    max_age = int(os.getenv("JWT_EXPIRE_MINUTES", "60")) * 60
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=max_age,
        httponly=True,
        samesite="strict",
        secure=os.getenv(
            "COOKIE_SECURE",
            "true" if os.getenv("APP_ENV", "").lower() == "production" else "false",
        ).lower()
        == "true",
        path="/",
    )


def clear_auth_cookie(response) -> None:
    response.delete_cookie(key=COOKIE_NAME, path="/")

# 历史版本的默认密钥：即使被写入环境也视为未配置（B1）。
_DEPRECATED_DEFAULT_SECRET = "change-this-development-secret"


def require_jwt_secret() -> str:
    """读取 JWT 密钥；未配置（或等于历史默认值）时直接抛错。

    绝不回退到硬编码默认密钥——该密钥公开存在于仓库历史中，
    拿到它就能签发任意用户 token，完全绕过登录（B1）。
    """
    secret = os.getenv("JWT_SECRET_KEY", "").strip()
    if not secret or secret == _DEPRECATED_DEFAULT_SECRET:
        raise RuntimeError(
            "JWT_SECRET_KEY 未配置（或仍是历史默认值）。"
            "为防止伪造 token，系统拒绝签发/校验 JWT。"
            "请在 .env 中设置至少 32 位的随机密钥后重启。"
        )
    return secret


def assert_jwt_configured() -> None:
    """启动期自检：密钥缺失即抛错，由 lifespan 终止应用启动（B1）。"""
    require_jwt_secret()


def create_access_token(user_id: str, username: str) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
    )
    payload = {
        "sub": user_id,
        "username": username,
        "jti": secrets.token_urlsafe(24),
        "exp": expires_at,
    }
    return jwt.encode(payload, require_jwt_secret(), algorithm=JWT_ALGORITHM)


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict:
    from auth.database import get_user_by_id, is_token_revoked
    # B10：优先 httpOnly Cookie，回退 Bearer 头（评测脚本/CLI 兼容）。
    if credentials is not None and credentials.scheme.lower() == "bearer":
        token = credentials.credentials
    else:
        token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="请先登录",
        )

    try:
        payload = jwt.decode(
            token,
            require_jwt_secret(),
            algorithms=[JWT_ALGORITHM],
        )
        user_id = payload.get("sub")
        username = payload.get("username")
        jti = payload.get("jti")
        if not user_id or not username or not jti or is_token_revoked(jti):
            raise ValueError("missing claims")
    except (jwt.InvalidTokenError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录已过期，请重新登录",
        ) from exc

    # 阶段二：角色实时读库而不是信任 JWT 载荷——
    # 授权/撤销立即生效，不等到 token 过期。
    role = "user"
    user_row = get_user_by_id(user_id)
    if user_row is not None:
        role = user_row["role"] or "user"
    else:
        # token 有效但用户已从库中删除：视为认证失效。
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录已过期，请重新登录",
        )

    return {"id": user_id, "username": username, "role": role}


def require_role(*roles: str):
    """阶段二：客服/管理员接口的角色门控依赖工厂。"""

    def dependency(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user.get("role") not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="权限不足，无法执行该操作",
            )
        return current_user

    return dependency
