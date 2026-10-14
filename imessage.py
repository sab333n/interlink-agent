"""
Text InterLink over iMessage, bridged by Sendblue.

Sendblue's free shared line has no webhooks, so inbound texts are polled from
GET /api/v2/messages. A phone is tied to an InterLink account by texting a
one-time "link 123456" code shown on the site; after that, texts run through
the same model, tools and memory (Supabase `agent_chats`) as the web chat.

Env: SENDBLUE_API_KEY_ID, SENDBLUE_API_SECRET, SENDBLUE_NUMBER (the line, E.164),
SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY. Polling runs only where
SENDBLUE_POLL=1 (or on the hosted server) so two servers never answer twice.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException

import chat
from auth import auth_required, require_user

router = APIRouter(prefix="/api/imessage")

SENDBLUE = "https://api.sendblue.co"
SITE_URL = os.getenv("INTERLINK_SITE_URL", "https://interlinkapp.vercel.app").rstrip("/")
POLL_SECONDS = 4
CODE_TTL = 15 * 60
CHANNEL_NOTE = (
    "CHANNEL: this conversation is over iMessage. plain text only, no markdown, no asterisks. "
    "cards you trigger with tools are sent as separate texts, so don't repeat their contents."
)

_codes: dict[str, tuple[str, float]] = {}
_locks: dict[str, asyncio.Lock] = {}
_nudged: dict[str, float] = {}
_tasks: set[asyncio.Task[None]] = set()


def log(msg: str) -> None:
    print(f"[imessage] {msg}", flush=True)


def sendblue_config() -> Optional[dict[str, str]]:
    key, secret, number = (os.getenv(k, "").strip() for k in
                           ("SENDBLUE_API_KEY_ID", "SENDBLUE_API_SECRET", "SENDBLUE_NUMBER"))
    return {"key": key, "secret": secret, "number": number} if key and secret and number else None


def admin_config() -> Optional[tuple[str, str]]:
    base = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    return (base, key) if base and key else None


def enabled() -> bool:
    return bool(sendblue_config() and admin_config() and chat.llm_config())


def _http(method: str, url: str, headers: dict[str, str], body: Any = None) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={**headers, **({"Content-Type": "application/json"} if data else {})})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {url.split('?')[0]} → {e.code}: {e.read()[:200]!r}") from None
    return json.loads(raw) if raw.strip() else None


async def http(method: str, url: str, headers: dict[str, str], body: Any = None) -> Any:
    return await asyncio.to_thread(_http, method, url, headers, body)


# ---- Sendblue ---------------------------------------------------------------

def _sb_headers() -> dict[str, str]:
    cfg = sendblue_config() or {}
    return {"sb-api-key-id": cfg.get("key", ""), "sb-api-secret-key": cfg.get("secret", "")}


async def send_text(to: str, content: str) -> None:
    cfg = sendblue_config()
    if cfg and content.strip():
        await http("POST", f"{SENDBLUE}/api/send-message", _sb_headers(),
                   {"number": to, "from_number": cfg["number"], "content": content.strip()})


async def send_typing(to: str) -> None:
    cfg = sendblue_config()
    try:
        await http("POST", f"{SENDBLUE}/api/send-typing-indicator", _sb_headers(),
                   {"number": to, "from_number": cfg["number"] if cfg else ""})
    except RuntimeError:
        pass


async def already_answered(phone: str, sent_at: str) -> bool:
    """True if we texted this number after the inbound message, e.g. from an overlapping deploy."""
    if not sent_at:
        return False
    q = urllib.parse.urlencode({"is_outbound": "true", "number": phone, "limit": 1, "created_at_gte": sent_at})
    res = await http("GET", f"{SENDBLUE}/api/v2/messages?{q}", _sb_headers())
    return bool((res or {}).get("data"))


async def inbound_since(since: datetime) -> list[dict[str, Any]]:
    q = urllib.parse.urlencode({
        "is_outbound": "false", "limit": 50, "order_direction": "asc",
        "created_at_gte": since.isoformat().replace("+00:00", "Z"),
    })
    res = await http("GET", f"{SENDBLUE}/api/v2/messages?{q}", _sb_headers())
    return (res or {}).get("data") or []


# ---- Supabase (service role, bypasses RLS; always filter by user) -----------

def _db_headers(prefer: str = "") -> dict[str, str]:
    _, key = admin_config() or ("", "")
    h = {"apikey": key, "Authorization": f"Bearer {key}"}
    if prefer:
        h["Prefer"] = prefer
    return h


async def db(method: str, path: str, body: Any = None, prefer: str = "") -> Any:
    base, _ = admin_config() or ("", "")
    return await http(method, f"{base}/rest/v1/{path}", _db_headers(prefer), body)


async def user_for_phone(phone: str) -> Optional[str]:
    rows = await db("GET", f"phone_links?select=user_id&phone=eq.{urllib.parse.quote(phone)}")
    return rows[0]["user_id"] if rows else None


async def phone_for_user(user_id: str) -> Optional[str]:
    rows = await db("GET", f"phone_links?select=phone&user_id=eq.{user_id}")
    return rows[0]["phone"] if rows else None


async def link_phone(phone: str, user_id: str) -> None:
    await db("DELETE", f"phone_links?user_id=eq.{user_id}")
    await db("POST", "phone_links?on_conflict=phone",
             {"phone": phone, "user_id": user_id}, prefer="resolution=merge-duplicates")


async def load_chat(user_id: str) -> dict[str, Any]:
    rows = await db("GET", f"agent_chats?select=messages,context,sent&user_id=eq.{user_id}")
    row = rows[0] if rows else {}
    return {"messages": row.get("messages") or [], "context": row.get("context") or {}, "sent": row.get("sent") or []}


async def save_chat(user_id: str, state: dict[str, Any]) -> None:
    await db("POST", "agent_chats?on_conflict=user_id", {
        "user_id": user_id,
        "messages": state["messages"][-120:],
        "context": state["context"],
        "sent": state["sent"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, prefer="resolution=merge-duplicates")


# ---- Conversation -----------------------------------------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _msg(sender: str, text: str, **extra: Any) -> dict[str, Any]:
    return {"id": f"{_now_ms():x}-{random.randrange(16 ** 6):06x}", "from": sender, "text": text,
            "at": _now_ms(), **extra}


def model_history(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Same shape the web chat sends: cards are summarized so the model remembers them."""
    out = []
    for m in messages[-24:]:
        content = m.get("text") or ""
        if m.get("picks"):
            content += "\n[showed picks: " + "; ".join(f"{p['company']} ({p['role']})" for p in m["picks"]) + "]"
        if m.get("draft"):
            d = m["draft"]
            content += f"\n[draft to {d['to']}, subject \"{d['subject']}\", status {d['status']}]\n{d['body']}"
        out.append({"role": "user" if m.get("from") == "me" else "assistant", "content": content.strip() or "(card)"})
    return out


def gmail_url(d: dict[str, Any]) -> str:
    q = urllib.parse.urlencode({"view": "cm", "fs": "1", "to": d["to"], "su": d["subject"], "body": d["body"]})
    return f"https://mail.google.com/mail/?{q}"


def open_draft(state: dict[str, Any]) -> Optional[dict[str, Any]]:
    draft_id = state["context"].get("draftMsgId")
    for m in reversed(state["messages"]):
        if m.get("id") == draft_id and m.get("draft", {}).get("status") == "draft":
            return m
    return None


async def draft_action(text: str, state: dict[str, Any]) -> Optional[str]:
    """"send" / "skip" on an open draft are handled here, like the web chat does."""
    m = open_draft(state)
    t = text.strip().lower()
    if not m or not re.match(r"^(send|send it|skip|no|nah|cancel)\b", t):
        return None
    d = m["draft"]
    if t.startswith("send"):
        d["status"] = "sent"
        state["sent"].append({"to": d["to"], "company": d["company"], "at": _now_ms()})
        return f"tap to send from gmail:\n{gmail_url(d)}"
    d["status"] = "skipped"
    return "skipped. want someone else?"


def picks_text(picks: list[dict[str, Any]]) -> str:
    lines = [f"{p['number']}. {p['company']} – {p['role']}" + (f" ({p['match_percent']}%)" if p.get("match_percent") else "")
             for p in picks]
    return "\n".join(lines) + "\n\nreply \"apply 1\" to start one"


def draft_text(d: dict[str, Any]) -> str:
    return f"to: {d['to']}\nsubject: {d['subject']}\n\n{d['body']}\n\nreply \"send\", \"shorter\", or \"skip\""


async def reply_to(phone: str, text: str) -> None:
    code = re.match(r"^\s*link\s+(\d{6})\s*$", text, re.I)
    if code:
        hit = _codes.pop(code.group(1), None)
        if not hit or time.time() - hit[1] > CODE_TTL:
            await send_text(phone, "that code expired. grab a fresh one on interlink → text your agent.")
            return
        await link_phone(phone, hit[0])
        await send_text(phone, "you're linked. text me anything - try \"picks\".")
        return

    user_id = await user_for_phone(phone)
    if not user_id:
        if time.time() - _nudged.get(phone, 0) > 600:
            _nudged[phone] = time.time()
            await send_text(phone, f"hey - i don't know this number yet. open {SITE_URL}/applicant/agent, "
                                   "tap connect imessage, and text me the code.")
        return

    state = await load_chat(user_id)
    state["messages"].append(_msg("me", text))
    local = await draft_action(text, state)
    if local:
        state["messages"].append(_msg("agent", local))
        await send_text(phone, local)
        await save_chat(user_id, state)
        return

    cfg = chat.llm_config()
    if not cfg:
        return
    context = dict(state["context"].get("snapshot") or {})
    context["current_person"] = state["context"].get("person")
    await send_typing(phone)

    texts: list[str] = []
    current = ""
    async for ev in chat.run_chat(model_history(state["messages"]), context, cfg, CHANNEL_NOTE):
        kind = ev["type"]
        if kind == "text":
            current += ev["delta"]
            continue
        if current.strip():
            texts.append(current.strip())
            state["messages"].append(_msg("agent", current.strip()))
        current = ""
        if kind == "picks":
            texts.append(picks_text(ev["picks"]))
            state["messages"].append(_msg("agent", "", picks=[
                {"jobId": p["job_id"], "company": p["company"], "role": p["role"],
                 "score": p.get("match_percent", 0), "reason": p.get("why", "")} for p in ev["picks"]]))
        elif kind == "draft":
            d = ev["draft"]
            m = _msg("agent", "", draft={"to": d["to"], "toName": d["to_name"], "company": d["company"],
                                         "subject": d["subject"], "body": d["body"], "status": "draft"})
            state["messages"].append(m)
            state["context"]["draftMsgId"] = m["id"]
            texts.append(draft_text(d))
        elif kind == "apply":
            link = f"{SITE_URL}/applicant/jobs?apply={urllib.parse.quote(ev['job_id'])}"
            texts.append(f"open this to review and submit:\n{link}")
        elif kind == "person":
            state["context"]["person"] = ev["person"]
        elif kind == "error":
            texts.append("hit a snag on my end. try again in a sec?")
    if current.strip():
        texts.append(current.strip())
        state["messages"].append(_msg("agent", current.strip()))

    for t in texts:
        await send_text(phone, t)
    await save_chat(user_id, state)


async def handle(phone: str, text: str, sent_at: str, backlog: bool) -> None:
    lock = _locks.setdefault(phone, asyncio.Lock())
    async with lock:
        try:
            if backlog and await already_answered(phone, sent_at):
                return
            await reply_to(phone, text)
            log(f"answered •••{phone[-4:]}")
        except Exception as e:  # one bad text must not stop the poller
            log(f"reply to •••{phone[-4:]} failed: {e}")


async def poll_forever() -> None:
    booted = datetime.now(timezone.utc)
    since = booted - timedelta(minutes=2)
    seen: dict[str, float] = {}
    log("polling Sendblue for texts")
    while True:
        try:
            for m in await inbound_since(since):
                handle_id = m.get("message_handle") or ""
                if handle_id in seen or m.get("is_outbound") or m.get("message_type") == "group":
                    continue
                seen[handle_id] = time.time()
                phone, text = m.get("from_number") or m.get("number") or "", (m.get("content") or "").strip()
                if phone and text:
                    sent_at = m.get("date_sent") or ""
                    backlog = sent_at < booted.isoformat().replace("+00:00", "Z")
                    log(f"text from •••{phone[-4:]}")
                    task = asyncio.create_task(handle(phone, text, sent_at, backlog))
                    _tasks.add(task)
                    task.add_done_callback(_tasks.discard)
            since = datetime.now(timezone.utc) - timedelta(minutes=2)
            for k in [k for k, t in seen.items() if time.time() - t > 600]:
                del seen[k]
        except Exception as e:
            log(f"poll failed: {e}")
        await asyncio.sleep(POLL_SECONDS)


async def keep_awake() -> None:
    """Render's free plan sleeps without inbound traffic, which would stop polling."""
    url = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
    if not url:
        return
    while True:
        await asyncio.sleep(600)
        try:
            await asyncio.to_thread(urllib.request.urlopen, f"{url}/api/agent/health", None, 15)
        except Exception:
            pass


def start_background() -> None:
    if not enabled():
        return
    if not (auth_required() or os.getenv("SENDBLUE_POLL", "").strip() == "1"):
        return
    asyncio.get_event_loop().create_task(poll_forever())
    asyncio.get_event_loop().create_task(keep_awake())


# ---- Site endpoints ----------------------------------------------------------

@router.get("/status")
async def status(user: str = Depends(require_user)) -> dict[str, Any]:
    cfg = sendblue_config()
    if not enabled() or user == "local":
        return {"enabled": False}
    return {"enabled": True, "number": cfg["number"] if cfg else None, "linked_phone": await phone_for_user(user)}


@router.post("/link-code")
async def link_code(user: str = Depends(require_user)) -> dict[str, Any]:
    cfg = sendblue_config()
    if not enabled() or not cfg or user == "local":
        raise HTTPException(503, "iMessage isn't set up on this server.")
    for k in [k for k, (u, t) in _codes.items() if u == user or time.time() - t > CODE_TTL]:
        del _codes[k]
    code = f"{random.SystemRandom().randrange(10 ** 6):06d}"
    _codes[code] = (user, time.time())
    return {"code": code, "number": cfg["number"], "expires_in": CODE_TTL}


@router.post("/unlink")
async def unlink(user: str = Depends(require_user)) -> dict[str, Any]:
    if not enabled() or user == "local":
        raise HTTPException(503, "iMessage isn't set up on this server.")
    await db("DELETE", f"phone_links?user_id=eq.{user}")
    return {"ok": True}
