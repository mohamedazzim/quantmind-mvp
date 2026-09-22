"""Structured Error Envelope and Exception Handlers."""

from __future__ import annotations

import uuid
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from quantmind.data.registry import DatasetRegistryError
from quantmind.paper.engine import PaperReplaySecurityError
from quantmind.paper.evaluation.feedback import ResearchFeedbackError
from quantmind.paper.evaluation.governance import GovernanceCausalError, GovernanceIntegrityError
from quantmind.paper.evaluation.ledger import EvaluationLedgerIntegrityError
from quantmind.paper.evaluation.service import PaperEvaluationServiceError
from quantmind.paper.feed import MarketFeedSecurityError
from quantmind.research_integrity.holdout import HoldoutSecurityError
from quantmind.research_integrity.qualification import QualificationLedgerError
from quantmind.research_integrity.trial_ledger import TrialBudgetExceeded
from quantmind.strategy.registry import StrategyRegistryError
from quantmind.strategy.spec import StrategySpecError


class ApiError(Exception):
    """Base API application exception."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def register_error_handlers(app: FastAPI) -> None:
    """Register uniform structured error handlers for all API exceptions."""

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
                "request_id": request_id,
            },
        )

    @app.exception_handler(GovernanceIntegrityError)
    @app.exception_handler(GovernanceCausalError)
    async def governance_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "code": "GOVERNANCE_INTEGRITY_VIOLATION",
                "message": str(exc),
                "details": {"type": type(exc).__name__},
                "request_id": request_id,
            },
        )

    @app.exception_handler(StrategyRegistryError)
    async def strategy_registry_handler(request: Request, exc: StrategyRegistryError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "code": "STRATEGY_REGISTRY_ERROR",
                "message": str(exc),
                "details": {},
                "request_id": request_id,
            },
        )

    @app.exception_handler(HoldoutSecurityError)
    @app.exception_handler(MarketFeedSecurityError)
    @app.exception_handler(PaperReplaySecurityError)
    async def security_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "code": "SECURITY_BOUNDARY_VIOLATION",
                "message": str(exc),
                "details": {"type": type(exc).__name__},
                "request_id": request_id,
            },
        )

    @app.exception_handler(EvaluationLedgerIntegrityError)
    @app.exception_handler(QualificationLedgerError)
    @app.exception_handler(DatasetRegistryError)
    async def ledger_integrity_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "code": "LEDGER_INTEGRITY_CONFLICT",
                "message": str(exc),
                "details": {"type": type(exc).__name__},
                "request_id": request_id,
            },
        )

    @app.exception_handler(TrialBudgetExceeded)
    async def budget_exceeded_handler(request: Request, exc: TrialBudgetExceeded) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "code": "TRIAL_BUDGET_EXCEEDED",
                "message": str(exc),
                "details": {},
                "request_id": request_id,
            },
        )

    @app.exception_handler(StrategySpecError)
    async def spec_error_handler(request: Request, exc: StrategySpecError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "code": "INVALID_STRATEGY_SPEC",
                "message": str(exc),
                "details": {},
                "request_id": request_id,
            },
        )

    @app.exception_handler(KeyError)
    async def key_error_handler(request: Request, exc: KeyError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
        clean_msg = str(exc).strip("\"'")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "code": "RESOURCE_NOT_FOUND",
                "message": f"Resource not found: {clean_msg}",
                "details": {},
                "request_id": request_id,
            },
        )

