"""Authentication and User Management Routes."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status

from quantmind.api.dependencies import get_ctx, get_current_user, require_role
from quantmind.app.auth.models import LoginRequest, PasswordChangeRequest, TokenResponse, User, UserRole
from quantmind.app.context import AppContext

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, ctx: AppContext = Depends(get_ctx)) -> TokenResponse:
    """Authenticate user with username and password, returning signed JWT."""
    user = ctx.auth_service.authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_CREDENTIALS", "message": "Incorrect username or password"},
        )

    token, expire_secs = ctx.auth_service.create_access_token(user)
    ctx.auth_service.log_action(user, "USER_LOGIN", {"username": user.username})
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expire_secs,
        user=user,
    )


@router.get("/me", response_model=User)
async def get_current_user_profile(user: User = Depends(get_current_user)) -> User:
    """Return profile of the currently authenticated user."""
    return user


@router.post("/logout")
async def logout(
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Log out current user."""
    ctx.auth_service.log_action(user, "USER_LOGOUT", {"username": user.username})
    return {"message": "Logged out successfully"}


@router.post("/change-password")
async def change_password(
    req: PasswordChangeRequest,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Change current user's password."""
    success = ctx.auth_service.change_password(user.user_id, req.old_password, req.new_password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "PASSWORD_CHANGE_FAILED", "message": "Invalid current password or update failed"},
        )
    ctx.auth_service.log_action(user, "PASSWORD_CHANGED", {"username": user.username})
    return {"message": "Password changed successfully"}


@router.get("/audit-logs", dependencies=[Depends(require_role(UserRole.ADMIN))])
async def get_audit_logs(
    limit: int = 50,
    ctx: AppContext = Depends(get_ctx),
) -> list[dict[str, Any]]:
    """Retrieve application-level user audit events (admin only)."""
    return ctx.auth_service.get_audit_logs(limit=limit)

