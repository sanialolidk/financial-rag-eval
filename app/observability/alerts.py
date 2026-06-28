"""Alert when eval scores fall below thresholds."""

from __future__ import annotations

from app.config import settings
from app.models import EvalScores


def check_alerts(scores: EvalScores) -> list[str]:
    alerts: list[str] = []
    if scores.faithfulness < settings.faithfulness_alert_threshold:
        alerts.append(
            f"LOW_FAITHFULNESS: {scores.faithfulness:.2f} < {settings.faithfulness_alert_threshold}"
        )
    if scores.chunk_recall < settings.chunk_recall_alert_threshold:
        alerts.append(
            f"LOW_CHUNK_RECALL: {scores.chunk_recall:.2f} < {settings.chunk_recall_alert_threshold}"
        )
    if scores.hallucination_detected:
        alerts.append(f"HALLUCINATION: {scores.hallucination_details[:120]}")
    return alerts