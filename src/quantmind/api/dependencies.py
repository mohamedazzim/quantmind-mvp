"""FastAPI Dependency Injection functions."""

from __future__ import annotations

from typing import Callable
from fastapi import Depends, Header, HTTPException, status

from quantmind.app.auth.models import User, UserRole
from quantmind.app.context import AppContext, get_app_context


def get_ctx() -> AppContext:
    """Provide the global AppContext."""
    return get_app_context()


def get_current_user(
    authorization: str | None = Header(None, alias="Authorization"),
    ctx: AppContext = Depends(get_ctx),
) -> User:
    """Extract and validate bearer token, returning authenticated User."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTHENTICATION_REQUIRED", "message": "Missing or invalid authorization header"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization.split(" ", 1)[1]
    user = ctx.auth_service.get_user_from_token(token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_TOKEN", "message": "Access token is invalid or expired"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_role(*allowed_roles: UserRole) -> Callable[[User], User]:
    """Dependency factory checking user permission role."""

    def role_checker(user: User = Depends(get_current_user)) -> User:
        # ADMIN can access all roles
        if user.role == UserRole.ADMIN or user.role in allowed_roles:
            return user
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "INSUFFICIENT_PERMISSIONS",
                "message": f"Action requires one of {[r.value for r in allowed_roles]}, user has {user.role.value}",
            },
        )

    return role_checker
