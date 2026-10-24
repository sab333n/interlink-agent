"""Dedicated PostHog OTLP log exporter for purpose-written service records."""

from __future__ import annotations

import logging

from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

LOGGER_NAME = "interlink.posthog"
_log_provider: LoggerProvider | None = None


def configure_posthog_log_capture(
    project_token: str | None, host: str | None
) -> tuple[logging.Logger, LoggerProvider | None]:
    """Configure a dedicated logger without exporting existing application logs."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not project_token or not host:
        return logger, None

    global _log_provider
    if _log_provider:
        return logger, _log_provider

    provider = LoggerProvider()
    set_logger_provider(provider)
    exporter = OTLPLogExporter(
        endpoint=f"{host.rstrip('/')}/i/v1/logs",
        headers={"Authorization": f"Bearer {project_token}"},
    )
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    logger.addHandler(LoggingHandler(logger_provider=provider))
    _log_provider = provider

    return logger, provider
