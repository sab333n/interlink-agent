"""
Daily job-match alerts over iMessage for users who've opted in.

Prefs live in agent_chats.context.alerts = { "enabled": true, "last_sent": iso }.
A background loop runs once an hour and sends at most one digest per user per day.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import require_user
from imessage import load_chat, phone_for_user, save_chat, send_text
from supa import admin_config, db

router = APIRouter(prefix="/api/alerts")

DIGEST_HOUR_UTC = int(os.getenv("ALERT_HOUR_UTC", "14"))  # ~9am US Central


class AlertPrefs(BaseModel):
    enabled: bool = False


@router.get("")
async def get_alerts(user: str = Depends(require_user)) -> dict[str, Any]:
    if user == "local" or not admin_config():
        return {"enabled": False, "linked": False}
    state = await load_chat(user)
    alerts = (state["context"].get("alerts") or {})
    phone = await phone_for_user(user)
    return {
        "enabled": bool(alerts.get("enabled")),
        "linked": bool(phone),
        "last_sent": alerts.get("last_sent"),
        "phone": phone,
    }


@router.post("")
async def set_alerts(body: AlertPrefs, user: str = Depends(require_user)) -> dict[str, Any]:
    if user == "local" or not admin_config():
        raise HTTPException(503, "Alerts need a signed-in account on the hosted agent.")
    phone = await phone_for_user(user)
    if body.enabled and not phone:
        raise HTTPException(400, "Connect iMessage first so we can text you new matches.")
    state = await load_chat(user)
    alerts = dict(state["context"].get("alerts") or {})
    alerts["enabled"] = body.enabled
    state["context"]["alerts"] = alerts
    await save_chat(user, state)
    return {"enabled": body.enabled, "linked": bool(phone), "phone": phone}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _picks_digest(snapshot: dict[str, Any]) -> Optional[str]:
    picks = snapshot.get("picks") or []
    if not picks:
        return None
    lines = []
    for i, p in enumerate(picks[:5], 1):
        score = p.get("score") or p.get("match_percent") or ""
        score_s = f" ({score}% match)" if score else ""
        sponsor = " · sponsors" if p.get("sponsorsVisa") or p.get("sponsors_visa") else ""
        lines.append(f"{i}. {p.get('company')}: {p.get('role')}{score_s}{sponsor}")
    return (
        "morning — fresh matches for you:\n\n"
        + "\n".join(lines)
        + "\n\nopen interlink to apply, or text me which number to start."
    )


async def _enabled_users() -> list[tuple[str, str, dict[str, Any]]]:
    """(user_id, phone, chat_state) for users with alerts on and a linked phone."""
    if not admin_config():
        return []
    try:
        rows = await db("GET", "agent_chats?select=user_id,messages,context,sent&limit=500")
    except RuntimeError:
        return []
    out = []
    for row in rows or []:
        ctx = row.get("context") or {}
        if not (ctx.get("alerts") or {}).get("enabled"):
            continue
        uid = row["user_id"]
        phone = await phone_for_user(uid)
        if not phone:
            continue
        out.append((uid, phone, {
            "messages": row.get("messages") or [],
            "context": ctx,
            "sent": row.get("sent") or [],
        }))
    return out


async def send_digests(force: bool = False) -> int:
    """Sends today's digest to opted-in users. Returns how many texts went out."""
    now = datetime.now(timezone.utc)
    if not force and now.hour != DIGEST_HOUR_UTC:
        return 0
    today = _today()
    sent_n = 0
    for uid, phone, state in await _enabled_users():
        alerts = dict(state["context"].get("alerts") or {})
        if alerts.get("last_sent") == today and not force:
            continue
        snapshot = state["context"].get("snapshot") or {}
        text = _picks_digest(snapshot)
        if not text:
            continue
        try:
            await send_text(phone, text)
        except Exception as e:
            print(f"[alerts] send to {phone[-4:]} failed: {e}", flush=True)
            continue
        alerts["last_sent"] = today
        state["context"]["alerts"] = alerts
        await save_chat(uid, state)
        sent_n += 1
        await asyncio.sleep(0.4)
    if sent_n:
        print(f"[alerts] sent {sent_n} digests", flush=True)
    return sent_n


async def poll_forever() -> None:
    while True:
        try:
            await send_digests()
        except Exception as e:
            print(f"[alerts] loop error: {e}", flush=True)
        await asyncio.sleep(30 * 60)


def start_background() -> None:
    if not admin_config():
        return
    # Same gate as iMessage: only the hosted instance should fire digests.
    from auth import auth_required
    if not (auth_required() or os.getenv("ALERTS_POLL", "").strip() == "1"):
        return
    asyncio.get_event_loop().create_task(poll_forever())


@router.post("/send-now")
async def send_now(user: str = Depends(require_user)) -> dict[str, Any]:
    """Manual trigger for the signed-in user (testing / “send sample”)."""
    if user == "local" or not admin_config():
        raise HTTPException(503, "Alerts need the hosted agent.")
    phone = await phone_for_user(user)
    if not phone:
        raise HTTPException(400, "Connect iMessage first.")
    state = await load_chat(user)
    text = _picks_digest(state["context"].get("snapshot") or {}) or (
        "no strong matches in your snapshot yet — open InterLink so your jobs sync, then try again."
    )
    await send_text(phone, text)
    alerts = dict(state["context"].get("alerts") or {})
    alerts["last_sent"] = _today()
    alerts["enabled"] = True
    state["context"]["alerts"] = alerts
    await save_chat(user, state)
    return {"ok": True, "preview": text[:200]}
