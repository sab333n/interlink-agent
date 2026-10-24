"""
Server-side Supabase access with the service-role key (bypasses RLS, so every
query must filter by user). Also the small JSON-over-HTTP helper the agent uses.
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional


def admin_config() -> Optional[tuple[str, str]]:
    base = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    return (base, key) if base and key else None


def _http(method: str, url: str, headers: dict[str, str], body: Any = None, raw: bool = False) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={**headers, **({"Content-Type": "application/json"} if data else {})})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            content = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {url.split('?')[0]} → {e.code}: {e.read()[:200]!r}") from None
    if raw:
        return content
    return json.loads(content) if content.strip() else None


async def http(method: str, url: str, headers: dict[str, str], body: Any = None, raw: bool = False) -> Any:
    return await asyncio.to_thread(_http, method, url, headers, body, raw)


def db_headers(prefer: str = "") -> dict[str, str]:
    _, key = admin_config() or ("", "")
    h = {"apikey": key, "Authorization": f"Bearer {key}"}
    if prefer:
        h["Prefer"] = prefer
    return h


async def db(method: str, path: str, body: Any = None, prefer: str = "") -> Any:
    base, _ = admin_config() or ("", "")
    return await http(method, f"{base}/rest/v1/{path}", db_headers(prefer), body)
