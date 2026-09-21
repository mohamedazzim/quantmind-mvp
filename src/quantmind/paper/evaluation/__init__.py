"""QuantMind Paper Evaluation & Monitoring Subsystem (PRD v4.0).

This package provides window-bounded evaluation, rolling performance and execution
quality metrics, degradation detection, and immutable evaluation evidence.
"""

from .models import (
    DegradationEvent,
    MonitoringConfig,
    MonitoringConfigError,
    MonitoringProvenanceError,
    MonitoringSnapshot,
    PaperEvaluationTransition,
    ResearchFeedbackRecord,
)

__all__ = [
    "DegradationEvent",
    "MonitoringConfig",
    "MonitoringConfigError",
    "MonitoringProvenanceError",
    "MonitoringSnapshot",
    "PaperEvaluationTransition",
    "ResearchFeedbackRecord",
]
