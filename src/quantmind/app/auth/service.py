"""Authentication and User Management Service (PRD v4.0 Productization)."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets
import sqlite3
from typing import Any
import jwt

from quantmind.app.auth.models import User, UserRole, UserCreate


class AuthService:
    """Manages application accounts, password verification, and JWT sessions."""

    def __init__(
        self,
        db_path: Path,
        secret_key: str,
        algorithm: str = "HS256",
        app_env: str = "development",
    ) -> None:
        self.db_path = Path(db_path)
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.app_env = app_env.upper()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        if self.app_env == "DEMO":
            self._seed_demo_users()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    force_password_change INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS app_audit_logs (
                    log_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations=100000,
        ).hex()

    def _seed_demo_users(self) -> None:
        """Seed demo users exclusively when APP_ENV=DEMO is enabled."""
        demo_accounts = [
            ("demo_admin", "admin@demo.quantmind.local", "QuantMindDemoAdmin2026!", UserRole.ADMIN),
            ("demo_researcher", "researcher@demo.quantmind.local", "QuantMindDemoResearch2026!", UserRole.RESEARCHER),
            ("demo_viewer", "viewer@demo.quantmind.local", "QuantMindDemoViewer2026!", UserRole.VIEWER),
        ]
        with self._get_connection() as conn:
            for username, email, password, role in demo_accounts:
                row = conn.execute("SELECT user_id FROM users WHERE username = ?", (username,)).fetchone()
                if row is None:
                    user_id = f"USR-{secrets.token_hex(6)}"
                    salt = secrets.token_hex(16)
                    h_pw = self._hash_password(password, salt)
                    now_ts = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        """
                        INSERT INTO users (user_id, username, email, hashed_password, salt, role, is_active, force_password_change, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, 1, 0, ?)
                        """,
                        (user_id, username, email, h_pw, salt, role.value, now_ts),
                    )

    def bootstrap_admin(
        self,
        username: str,
        email: str,
        password: str,
        force_password_change: bool = False,
    ) -> User:
        """Bootstrap an administrative user. Fails if user already exists."""
        if len(password) < 12:
            raise ValueError("Bootstrap admin password must be at least 12 characters long.")
        with self._get_connection() as conn:
            existing = conn.execute(
                "SELECT user_id FROM users WHERE username = ? OR email = ?",
                (username, email),
            ).fetchone()
            if existing:
                raise ValueError(f"User with username '{username}' or email '{email}' already exists.")

            user_id = f"USR-{secrets.token_hex(6)}"
            salt = secrets.token_hex(16)
            h_pw = self._hash_password(password, salt)
            now_ts = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO users (user_id, username, email, hashed_password, salt, role, is_active, force_password_change, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (user_id, username, email, h_pw, salt, UserRole.ADMIN.value, int(force_password_change), now_ts),
            )
            return User(
                user_id=user_id,
                username=username,
                email=email,
                role=UserRole.ADMIN,
                is_active=True,
                force_password_change=force_password_change,
                created_at=now_ts,
            )

    def create_user(
        self,
        user_create: UserCreate,
        creator: User | None = None,
    ) -> User:
        """Create a new application user account."""
        with self._get_connection() as conn:
            existing = conn.execute(
                "SELECT user_id FROM users WHERE username = ? OR email = ?",
                (user_create.username, user_create.email),
            ).fetchone()
            if existing:
                raise ValueError(f"User with username '{user_create.username}' or email '{user_create.email}' already exists.")

            user_id = f"USR-{secrets.token_hex(6)}"
            salt = secrets.token_hex(16)
            h_pw = self._hash_password(user_create.password, salt)
            now_ts = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO users (user_id, username, email, hashed_password, salt, role, is_active, force_password_change, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    user_id,
                    user_create.username,
                    user_create.email,
                    h_pw,
                    salt,
                    user_create.role.value,
                    int(user_create.force_password_change),
                    now_ts,
                ),
            )

            created_user = User(
                user_id=user_id,
                username=user_create.username,
                email=user_create.email,
                role=user_create.role,
                is_active=True,
                force_password_change=user_create.force_password_change,
                created_at=now_ts,
            )
            if creator:
                self.log_action(creator, "CREATE_USER", {"target_user_id": user_id, "role": user_create.role.value})
            return created_user

    def authenticate_user(self, username: str, password: str) -> User | None:
        """Authenticate user credentials against the application database."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ? AND is_active = 1",
                (username,),
            ).fetchone()
            if row is None:
                return None

            computed_hash = self._hash_password(password, row["salt"])
            if secrets.compare_digest(computed_hash, row["hashed_password"]):
                return User(
                    user_id=row["user_id"],
                    username=row["username"],
                    email=row["email"],
                    role=UserRole(row["role"]),
                    is_active=bool(row["is_active"]),
                    force_password_change=bool(row["force_password_change"]),
                    created_at=row["created_at"],
                )
            return None

    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """Change user password after verifying existing credentials."""
        if len(new_password) < 12:
            raise ValueError("New password must be at least 12 characters.")
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id = ? AND is_active = 1", (user_id,)).fetchone()
            if not row:
                return False
            computed_old = self._hash_password(old_password, row["salt"])
            if not secrets.compare_digest(computed_old, row["hashed_password"]):
                return False

            new_salt = secrets.token_hex(16)
            new_hash = self._hash_password(new_password, new_salt)
            conn.execute(
                "UPDATE users SET hashed_password = ?, salt = ?, force_password_change = 0 WHERE user_id = ?",
                (new_hash, new_salt, user_id),
            )
            return True

    def create_access_token(self, user: User, expires_delta: timedelta | None = None) -> tuple[str, int]:
        """Create signed JWT access token for user."""
        expire_seconds = int(expires_delta.total_seconds()) if expires_delta else 86400
        expire = datetime.now(timezone.utc) + timedelta(seconds=expire_seconds)
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "force_password_change": user.force_password_change,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
        }
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token, expire_seconds

    def get_user_from_token(self, token: str) -> User | None:
        """Decode and validate a JWT access token returning active User."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id: str = payload.get("sub")
            if not user_id:
                return None
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM users WHERE user_id = ? AND is_active = 1",
                    (user_id,),
                ).fetchone()
                if row is None:
                    return None
                return User(
                    user_id=row["user_id"],
                    username=row["username"],
                    email=row["email"],
                    role=UserRole(row["role"]),
                    is_active=bool(row["is_active"]),
                    force_password_change=bool(row["force_password_change"]),
                    created_at=row["created_at"],
                )
        except (jwt.PyJWTError, KeyError):
            return None

    def log_action(self, user: User, action: str, details: dict[str, Any] | None = None) -> None:
        """Record an application-level user action for audit logging."""
        log_id = f"LOG-{secrets.token_hex(8)}"
        now_ts = datetime.now(timezone.utc).isoformat()
        details_json = json.dumps(details or {}, sort_keys=True)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO app_audit_logs (log_id, timestamp, user_id, username, action, details_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (log_id, now_ts, user.user_id, user.username, action, details_json),
            )

    def get_audit_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve recent application audit logs."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM app_audit_logs ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {
                    "log_id": r["log_id"],
                    "timestamp": r["timestamp"],
                    "user_id": r["user_id"],
                    "username": r["username"],
                    "action": r["action"],
                    "details": json.loads(r["details_json"]),
                }
                for r in rows
            ]
