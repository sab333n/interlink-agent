"""
Backend for the InterLink Chrome extension, which fills application forms on
the employer's own site in the user's browser.

The extension signs in once through the InterLink site: it trades the user's
Supabase session for a long-lived extension token (HMAC-signed, no table
needed), so it never has to refresh Supabase sessions itself. Profile and
resume come from the snapshot the site saves to `agent_chats`.

Answers use the same rules as the server-side filler, then `answers.suggest`
for whatever is still blank.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from pydantic import BaseModel

import answers
import queue_runner as qr
from auth import auth_required, require_user
from imessage import download_resume, load_chat
from supa import admin_config

router = APIRouter(prefix="/api/ext")

TOKEN_DAYS = 90
CONSENT = re.compile(r"acknowledge|privacy (policy|notice)|i have read|consent|agree|certify", re.I)
CONSENT_OPTION = re.compile(r"acknowledge|agree|accept|yes|i have read|consent", re.I)


def _secret() -> bytes:
    key = os.getenv("EXT_TOKEN_SECRET") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "local-dev"
    return hashlib.sha256(b"interlink-ext:" + key.encode()).digest()


def _sign(payload: str) -> str:
    return hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()[:40]


def make_token(user_id: str) -> tuple[str, int]:
    expires = int(time.time()) + TOKEN_DAYS * 86400
    payload = f"{user_id}.{expires}"
    body = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
    return f"{body}.{_sign(payload)}", expires


def read_token(token: str) -> Optional[str]:
    try:
        body, sig = token.rsplit(".", 1)
        payload = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)).decode()
        user_id, expires = payload.rsplit(".", 1)
    except ValueError:
        return None
    if not hmac.compare_digest(sig, _sign(payload)) or int(expires) < time.time():
        return None
    return user_id


async def ext_user(x_interlink_ext: Optional[str] = Header(default=None)) -> str:
    if not x_interlink_ext and not auth_required():
        return "local"
    user_id = read_token((x_interlink_ext or "").strip())
    if not user_id:
        raise HTTPException(401, "Open InterLink and sign in to reconnect the extension.")
    return user_id


async def _apply_info(user_id: str) -> dict[str, Any]:
    if not admin_config() or user_id == "local":
        raise HTTPException(409, "Profile storage isn't configured on this server.")
    state = await load_chat(user_id)
    info = (state["context"].get("snapshot") or {}).get("apply") or {}
    if not info.get("profile"):
        raise HTTPException(409, "Open InterLink once so your profile syncs, then try again.")
    return info


@router.post("/connect")
async def connect(user: str = Depends(require_user)) -> dict[str, Any]:
    token, expires = make_token(user)
    return {"token": token, "expires": expires}


@router.get("/me")
async def me(user: str = Depends(ext_user)) -> dict[str, Any]:
    info = await _apply_info(user)
    p = info["profile"]
    return {
        "name": f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
        "email": p.get("email", ""),
        "resume_name": info.get("resume_name") if info.get("resume_path") else None,
    }


@router.get("/resume")
async def resume(user: str = Depends(ext_user)) -> Response:
    info = await _apply_info(user)
    if not info.get("resume_path"):
        raise HTTPException(404, "No resume on file. Upload one on the Resume page.")
    data = await download_resume(info["resume_path"])
    name = re.sub(r"[^A-Za-z0-9._-]", "_", info.get("resume_name") or "resume.pdf")
    return Response(data, media_type="application/pdf", headers={"X-Filename": name})


class Field(BaseModel):
    id: str
    kind: str
    label: str
    required: bool = False
    value: str = ""
    options: list[str] = []


class FillRequest(BaseModel):
    fields: list[Field]
    job: dict[str, Any] = {}


def _rule_answer(f: dict[str, Any]) -> Optional[str]:
    label, kind, opts = f["label"], f["kind"], f["options"]
    if kind == "checkbox":
        return "true" if f["required"] and CONSENT.search(label) else None
    if opts:
        if qr.EEO_LABEL.search(label):
            decline = next((o for o in opts if qr.DECLINE_OPTION.search(o)), None)
            if decline:
                return decline
        if re.search(r"term|available to intern|which (term|season)|internship period", label, re.I):
            summer = [o for o in opts if re.search(r"summer", o, re.I)]
            if summer:
                return ", ".join(summer) if kind == "checkboxes" else summer[0]
        if re.search(r"how (did )?you (hear|heard)|source|referr", label, re.I):
            for pattern in (r"online job board", r"simplify", r"career", r"job board|internet|website|other"):
                hit = next((o for o in opts if re.search(pattern, o, re.I)), None)
                if hit:
                    return hit
        direct = qr.answer_for(label)
        choice = qr.pick_option(opts, direct) if direct else None
        if direct == "Online job board" and not choice:
            choice = next((o for o in opts if re.search(r"job board|online|internet|website|other", o, re.I)), None)
        if not choice and CONSENT.search(label):
            choice = next((o for o in opts if CONSENT_OPTION.search(o)), None)
        return choice
    if kind == "textarea":
        return (qr.PROFILE.get("cover_letter") or None) if re.search(r"cover letter", label, re.I) else None
    if re.search(r"if different|preferred name", label, re.I) and not f["required"]:
        return None
    return qr.answer_for(label)


def _rules(fields: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, str]:
    # PROFILE is shared with in-progress server fills; swap it only for this synchronous pass.
    saved = dict(qr.PROFILE)
    try:
        qr.PROFILE.clear()
        qr.PROFILE.update(qr.agent._PROFILE_DEFAULTS)
        qr.PROFILE.update({k: v for k, v in profile.items() if v is not None})
        out = {}
        for f in fields:
            if f["kind"] == "file" or str(f["value"]).strip():
                continue
            value = _rule_answer(f)
            if value:
                out[f["id"]] = value
        return out
    finally:
        qr.PROFILE.clear()
        qr.PROFILE.update(saved)


@router.post("/fill")
async def fill(req: FillRequest, user: str = Depends(ext_user)) -> dict[str, Any]:
    info = await _apply_info(user)
    profile = info["profile"]
    fields = [f.model_dump() for f in req.fields][:150]
    for f in fields:
        f["label"] = re.sub(r"\s*[*✱]\s*$", "", f["label"]).strip()[:400]
    out = {fid: {"answer": v, "source": "profile"} for fid, v in _rules(fields, profile).items()}
    rest = [f for f in fields if f["id"] not in out and f["label"]
            and (f["required"] or not re.search(r"if different|preferred name", f["label"], re.I))]
    out.update(await answers.suggest(rest, req.job, profile, user))
    return {"answers": out}


class LearnRequest(BaseModel):
    company: str = ""
    pairs: list[tuple[str, str]]


@router.post("/learn")
async def learn(req: LearnRequest, user: str = Depends(ext_user)) -> dict[str, Any]:
    pairs = [(q, a) for q, a in req.pairs[:60] if len(q) < 400]
    await answers.remember(user, req.company, pairs)
    return {"ok": True}
