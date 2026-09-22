"""QuantMind FastAPI Application Factory and Root Configuration."""

from __future__ import annotations

from contextlib import asynccontextmanager
import time
from typing import AsyncGenerator
import uuid
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from quantmind.api.errors import register_error_handlers
from quantmind.api.routes import (
    audit,
    auth,
    dashboard,
    datasets,
    feedback,
    governance,
    jobs,
    monitoring,
    paper,
    qualification,
    research,
    strategies,
    trials,
)
from quantmind.app.config import AppSettings, get_settings
from quantmind.app.context import AppContext, get_app_context


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize application context and verify database connectivity."""
    ctx = get_app_context()
    # Confirm core and app DB connections
    with ctx.get_core_connection() as conn:
        conn.execute("SELECT 1")
    yield


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Build and configure the main FastAPI application."""
    if settings is not None:
        ctx = AppContext(settings)
        app_settings = settings
    else:
        ctx = get_app_context()
        app_settings = ctx.settings

    app = FastAPI(
        title="QuantMind Quantitative Platform API",
        description="REST API for QuantMind v4.0 research, execution, evaluation, and governance platform.",
        version="4.0.0",
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.ctx = ctx

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID and latency tracking middleware
    @app.middleware("http")
    async def add_request_metadata(request: Request, call_next) -> Response:
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request.state.request_id = req_id
        start_time = time.time()

        response = await call_next(request)

        duration_ms = round((time.time() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        return response

    # Register structured domain and security error handlers
    register_error_handlers(app)

    # Health and Readiness Endpoints
    @app.get("/health", tags=["Health"])
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "service": "quantmind-api", "version": "4.0.0"}

    @app.get("/ready", tags=["Health"])
    async def readiness() -> JSONResponse:
        """Readiness check validating database connectivity."""
        ctx = get_app_context()
        try:
            with ctx.get_core_connection() as conn:
                conn.execute("SELECT 1")
            return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ready"})
        except Exception as e:
            return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "not ready", "error": str(e)})

    # Mount API v1 routers
    api_v1_prefix = "/api/v1"
    app.include_router(auth.router, prefix=api_v1_prefix)
    app.include_router(dashboard.router, prefix=api_v1_prefix)
    app.include_router(datasets.router, prefix=api_v1_prefix)
    app.include_router(strategies.router, prefix=api_v1_prefix)
    app.include_router(research.router, prefix=api_v1_prefix)
    app.include_router(trials.router, prefix=api_v1_prefix)
    app.include_router(qualification.router, prefix=api_v1_prefix)
    app.include_router(paper.router, prefix=api_v1_prefix)
    app.include_router(monitoring.router, prefix=api_v1_prefix)
    app.include_router(governance.router, prefix=api_v1_prefix)
    app.include_router(feedback.router, prefix=api_v1_prefix)
    app.include_router(audit.router, prefix=api_v1_prefix)
    app.include_router(jobs.router, prefix=api_v1_prefix)

    return app


app = create_app()
