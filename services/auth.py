"""
Request auth for the hosted agent.

When AGENT_REQUIRE_AUTH=1, every protected endpoint needs a Supabase access token
(`Authorization: Bearer <jwt>`) belonging to a signed-in InterLink user. Tokens are
checked against Supabase's /auth/v1/user endpoint and cached briefly. Locally the
check is off and every request is treated as the same "local" user.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from fastapi import Header, HTTPException

TOKEN_TTL = 300
_tokens: dict[str, tuple[float, str]] = {}


def auth_required() -> bool:
    return os.getenv("AGENT_REQUIRE_AUTH", "").strip() == "1"


def _lookup_user(token: str) -> Optional[str]:
    base = os.getenv("SUPABASE_URL", "").rstrip("/")
    anon = os.getenv("SUPABASE_ANON_KEY", "")
    if not base or not anon:
        return None
    req = urllib.request.Request(
        f"{base}/auth/v1/user",
        headers={"Authorization": f"Bearer {token}", "apikey": anon},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read()).get("id")
    except (urllib.error.URLError, ValueError):
        return None


async def require_user(authorization: Optional[str] = Header(default=None)) -> str:
    if not auth_required():
        return "local"
    token = (authorization or "").removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(401, "Sign in to InterLink to use the agent.")
    hit = _tokens.get(token)
    if hit and time.time() - hit[0] < TOKEN_TTL:
        return hit[1]
    user_id = await asyncio.to_thread(_lookup_user, token)
    if not user_id:
        raise HTTPException(401, "Your session expired. Sign in again.")
    if len(_tokens) > 5000:
        _tokens.clear()
    _tokens[token] = (time.time(), user_id)
    return user_id
