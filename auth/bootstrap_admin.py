"""Promote an existing account using local administrative access."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from auth.database import init_database, set_user_role


def main():
    parser = argparse.ArgumentParser(description="Grant administrator role to an existing local account")
    parser.add_argument("username")
    args = parser.parse_args()
    init_database()
    user = set_user_role(args.username, "admin")
    if user is None:
        parser.error("账号不存在：请先用该用户名注册普通账号")
    print(f"Administrator role granted to {user['username']}")


if __name__ == "__main__":
    main()
