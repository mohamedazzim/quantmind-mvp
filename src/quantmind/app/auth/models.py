"""Authentication domain models and role definitions."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, EmailStr, Field


class UserRole(str, Enum):
    """Authoritative application roles."""

    ADMIN = "ADMIN"
    RESEARCHER = "RESEARCHER"
    VIEWER = "VIEWER"


class User(BaseModel):
    """User profile model."""

    user_id: str
    username: str
    email: str
    role: UserRole
    is_active: bool = True
    force_password_change: bool = False
    created_at: str


class UserCreate(BaseModel):
    """Schema for user creation."""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=12)
    role: UserRole = UserRole.RESEARCHER
    force_password_change: bool = False


class LoginRequest(BaseModel):
    """Login request payload."""

    username: str
    password: str


class PasswordChangeRequest(BaseModel):
    """Password change payload."""

    old_password: str
    new_password: str = Field(..., min_length=12)


class TokenResponse(BaseModel):
    """Authentication token response payload."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: User
