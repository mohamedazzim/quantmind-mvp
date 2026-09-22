"""Administrative CLI for QuantMind Application Layer (PRD v4.0 Productization)."""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

from quantmind.app.config import get_settings
from quantmind.app.auth.service import AuthService
from quantmind.app.auth.models import UserRole, UserCreate


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantMind Application Admin CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # bootstrap-admin
    boot_parser = subparsers.add_parser("bootstrap-admin", help="Bootstrap initial admin account")
    boot_parser.add_argument("--username", default="admin", help="Admin username")
    boot_parser.add_argument("--email", default="admin@quantmind.internal", help="Admin email")
    boot_parser.add_argument("--password", default=None, help="Password (or prompt if omitted)")
    boot_parser.add_argument("--force-change", action="store_true", help="Force password rotation on first login")

    # create-user
    user_parser = subparsers.add_parser("create-user", help="Create a new user")
    user_parser.add_argument("--username", required=True, help="Username")
    user_parser.add_argument("--email", required=True, help="Email")
    user_parser.add_argument("--role", choices=["ADMIN", "RESEARCHER", "VIEWER"], default="RESEARCHER", help="User role")
    user_parser.add_argument("--password", default=None, help="Password (or prompt if omitted)")

    args = parser.parse_args()
    settings = get_settings()
    auth_service = AuthService(
        db_path=settings.app_db_path,
        secret_key=settings.secret_key,
        algorithm=settings.jwt_algorithm,
        app_env=settings.app_env,
    )

    if args.command == "bootstrap-admin":
        password = args.password or os.getenv("QUANTMIND_ADMIN_INITIAL_PASSWORD")
        if not password:
            password = getpass.getpass("Enter password for initial admin (min 12 chars): ")
        try:
            user = auth_service.bootstrap_admin(
                username=args.username,
                email=args.email,
                password=password,
                force_password_change=args.force_change,
            )
            print(f"[SUCCESS] Admin account created: {user.username} ({user.email}) [ID: {user.user_id}]")
        except Exception as exc:
            print(f"[ERROR] Failed to bootstrap admin: {exc}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "create-user":
        password = args.password
        if not password:
            password = getpass.getpass(f"Enter password for user '{args.username}' (min 12 chars): ")
        try:
            uc = UserCreate(
                username=args.username,
                email=args.email,
                password=password,
                role=UserRole(args.role),
            )
            user = auth_service.create_user(uc)
            print(f"[SUCCESS] User created: {user.username} ({user.role.value}) [ID: {user.user_id}]")
        except Exception as exc:
            print(f"[ERROR] Failed to create user: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
