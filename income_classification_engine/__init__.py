"""Income classification engine package."""

from serviflow.models import PipelineResult

from .domain.summary import build_summary
from .engine import IncomeEngine
from .pipeline import run_pipeline
from .presentation.reporting import write_report

__all__ = [
    "IncomeEngine",
    "PipelineResult",
    "build_summary",
    "run_pipeline",
    "write_report",
]
