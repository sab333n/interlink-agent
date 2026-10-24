"""Shared PostHog client accessor for FastAPI dependencies.

Lives outside server.py so route modules can import it without a circular
import when the server is started with `python server.py`.
"""

from __future__ import annotations

from fastapi import Request
from posthog import Posthog


def get_posthog_client(request: Request) -> Posthog | None:
    """Return the shared PostHog client for FastAPI dependencies."""
    return getattr(request.app.state, "posthog_client", None)
