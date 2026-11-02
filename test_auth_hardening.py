"""认证加固的回归测试：JWT 吊销、旧令牌拒绝、Secure Cookie 生产默认、MySQL 最小权限默认。

对应 PR：Harden authentication and deployment defaults。
"""

import importlib
import json
import sqlite3

import jwt
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from auth import database as auth_db


@pytest.fixture
def isolated_auth(tmp_path, monkeypatch):
    """隔离的账户库 + 固定 JWT 密钥；api.main 重新加载以拾取新数据库路径。"""
    monkeypatch.setenv("AUTH_DATABASE_PATH", str(tmp_path / "app.db"))
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-for-auth-hardening")
    importlib.reload(auth_db)
    auth_db.init_database()
    importlib.reload(api_main)
    yield api_main
    importlib.reload(auth_db)
    importlib.reload(api_main)


def _register(client, username="hardening_user", password="Secret#123"):
    response = client.post(
        "/api/auth/register", json={"username": username, "password": password}
    )
    assert response.status_code == 201, response.text
    return response.json()["access_token"]


def test_logout_revokes_token_and_reuse_is_rejected(isolated_auth):
    """登出后，同一令牌（Cookie 或 Bearer）必须立即失效。"""
    client = TestClient(api_main.app)
    token = _register(client)

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200, "登出前令牌应可用"

    client.post("/api/auth/logout")

    reused = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert reused.status_code == 401, "登出后旧令牌必须被拒绝"


def test_legacy_token_without_jti_is_rejected(isolated_auth):
    """部署前签发的旧令牌没有 jti 声明：必须拒绝，强制重新登录。"""
    client = TestClient(api_main.app)
    _register(client)
    from auth.security import create_access_token  # noqa: PLC0415

    # 手工构造旧版 payload（无 jti）。
    legacy_token = jwt.encode(
        {"sub": "someone", "username": "someone", "exp": 4102444800},
        __import__("os").getenv("JWT_SECRET_KEY"),
        algorithm="HS256",
    )
    reused = client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {legacy_token}"}
    )
    assert reused.status_code == 401
    assert create_access_token is not None  # 防止误删导入


def test_expired_revocations_are_pruned(isolated_auth, monkeypatch):
    """过期的吊销记录在检查时清理，不长期占用撤销表。"""
    from datetime import datetime, timedelta, timezone

    auth_db.revoke_token("expired-jti", (datetime.now(timezone.utc) - timedelta(days=1)).isoformat())
    auth_db.revoke_token("live-jti", (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat())

    assert auth_db.is_token_revoked("expired-jti") is False, "过期吊销应视为失效并清理"
    assert auth_db.is_token_revoked("live-jti") is True

    with sqlite3.connect(auth_db.DATABASE_PATH) as db:
        remaining = db.execute("SELECT jti FROM revoked_tokens").fetchall()
    assert remaining == [("live-jti",)]


def test_secure_cookie_enabled_in_production(isolated_auth, monkeypatch):
    """APP_ENV=production 时 Cookie 必须带 Secure；未设置时保持本地 http 可用。"""
    from auth.security import set_auth_cookie

    captured = {}

    class FakeResponse:
        def set_cookie(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("APP_ENV", "production")
    set_auth_cookie(FakeResponse(), "token")
    assert captured["secure"] is True, "生产环境必须下发 Secure Cookie"

    monkeypatch.delenv("APP_ENV")
    set_auth_cookie(FakeResponse(), "token")
    assert captured["secure"] is False, "未声明生产环境时保持本地 http 可用"


def test_mysql_default_user_is_least_privilege_account(monkeypatch):
    """未显式配置时，应用默认使用最小权限账号而非 root。"""
    from storage.mysql_store import MySQLStore

    monkeypatch.delenv("MYSQL_USER", raising=False)
    assert MySQLStore._config()["user"] == "smart_cs_app"
