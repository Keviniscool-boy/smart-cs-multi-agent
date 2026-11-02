import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = Path(
    os.getenv("AUTH_DATABASE_PATH", str(PROJECT_ROOT / "data" / "app.db"))
)


def _connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database() -> None:
    with _connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS revoked_tokens (
                jti TEXT PRIMARY KEY,
                expires_at TEXT NOT NULL
            );
            """
        )
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(users)")}
        if "role" not in columns:
            try:
                connection.execute(
                    "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'"
                )
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        salt_hex, digest_hex = encoded.split("$", 1)
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            bytes.fromhex(salt_hex),
            120_000,
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def create_user(username: str, password: str) -> dict:
    user_id = secrets.token_hex(16)
    created_at = datetime.now(timezone.utc).isoformat()
    with _connection() as connection:
        role = "user"
        connection.execute(
            "INSERT INTO users (id, username, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, username, hash_password(password), role, created_at),
        )
    return {"id": user_id, "username": username, "role": role, "created_at": created_at}


def get_user_by_username(username: str) -> sqlite3.Row | None:
    with _connection() as connection:
        return connection.execute(
            "SELECT id, username, password_hash, role, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()


def get_user_by_id(user_id: str) -> sqlite3.Row | None:
    with _connection() as connection:
        return connection.execute(
            "SELECT id, username, role, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()


def set_user_role(username: str, role: str) -> sqlite3.Row | None:
    """admin 授权入口；角色白名单由调用方校验。"""
    with _connection() as connection:
        connection.execute(
            "UPDATE users SET role = ? WHERE username = ?",
            (role, username),
        )
        return connection.execute(
            "SELECT id, username, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()


def get_usernames_by_ids(user_ids: list[str]) -> dict[str, str]:
    """批量取用户名映射（工作台展示用；售后库与账户库分文件，不能 SQL JOIN）。"""
    ids = [uid for uid in set(user_ids) if uid]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    with _connection() as connection:
        rows = connection.execute(
            f"SELECT id, username FROM users WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    return {row["id"]: row["username"] for row in rows}


def save_session_owner(session_id: str, user_id: str) -> None:
    with _connection() as connection:
        connection.execute(
            "INSERT OR IGNORE INTO sessions (session_id, user_id, created_at) VALUES (?, ?, ?)",
            (session_id, user_id, datetime.now(timezone.utc).isoformat()),
        )


def user_owns_session(session_id: str, user_id: str) -> bool:
    with _connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
    return row is not None


def revoke_token(jti: str, expires_at: str) -> None:
    with _connection() as connection:
        connection.execute(
            "INSERT OR REPLACE INTO revoked_tokens (jti, expires_at) VALUES (?, ?)",
            (jti, expires_at),
        )


def is_token_revoked(jti: str) -> bool:
    with _connection() as connection:
        connection.execute(
            "DELETE FROM revoked_tokens WHERE expires_at < ?",
            (datetime.now(timezone.utc).isoformat(),),
        )
        row = connection.execute(
            "SELECT 1 FROM revoked_tokens WHERE jti = ?",
            (jti,),
        ).fetchone()
    return row is not None


def session_exists(session_id: str) -> bool:
    with _connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return row is not None


def list_user_sessions(user_id: str) -> list[dict]:
    """返回指定用户拥有的会话，按创建时间倒序排列。"""
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT session_id, created_at
            FROM sessions
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]
