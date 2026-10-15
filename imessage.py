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
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import chat
import inline_apply
from supa import admin_config, db, db_headers as _db_headers, http
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
# user id → {"sid", "queue": field ids still to ask, "answers", "profile"} for an application in progress
_applies: dict[str, dict[str, Any]] = {}


def log(msg: str) -> None:
    print(f"[imessage] {msg}", flush=True)


def sendblue_config() -> Optional[dict[str, str]]:
    key, secret, number = (os.getenv(k, "").strip() for k in
                           ("SENDBLUE_API_KEY_ID", "SENDBLUE_API_SECRET", "SENDBLUE_NUMBER"))
    return {"key": key, "secret": secret, "number": number} if key and secret and number else None


def enabled() -> bool:
    return bool(sendblue_config() and admin_config() and chat.llm_config())


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


async def download_resume(path: str) -> bytes:
    base, _ = admin_config() or ("", "")
    return await http("GET", f"{base}/storage/v1/object/resumes/{urllib.parse.quote(path)}", _db_headers(), raw=True)


async def record_application(user_id: str, job: dict[str, Any]) -> None:
    await db("POST", "applications?on_conflict=user_id,job_id",
             {"user_id": user_id, "job_id": job["id"], "company": job["company"], "role": job["role"],
              "stage": "applied"}, prefer="resolution=merge-duplicates")


# ---- Applying by text -------------------------------------------------------

def _spawn(coro: Any) -> None:
    task = asyncio.create_task(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


def _finish_link(job_id: str) -> str:
    return f"{SITE_URL}/applicant/jobs?apply={urllib.parse.quote(job_id)}"


async def run_apply(phone: str, user_id: str, job_id: str, snapshot: dict[str, Any]) -> None:
    """Fills the employer's form in the server browser, then texts a summary and waits for "submit"."""
    info = snapshot.get("apply") or {}
    job = (info.get("jobs") or {}).get(job_id)
    if not job or not job.get("apply_url"):
        await send_text(phone, f"i can't fill that one from here. open it:\n{_finish_link(job_id)}")
        return
    if not info.get("resume_path"):
        await send_text(phone, f"upload your resume first, then i can apply for you:\n{SITE_URL}/applicant/profile")
        return
    old = _applies.pop(user_id, None)
    if old:
        await inline_apply.close_session(old["sid"])
    try:
        resume = await download_resume(info["resume_path"])
        s = await inline_apply.open_session(
            {"id": job_id, "company": job["company"], "role": job["role"], "apply_url": job["apply_url"]},
            info.get("profile") or {}, resume, info.get("resume_name"), user_id)
    except HTTPException as e:
        await send_text(phone, f"{str(e.detail).lower()}")
        return
    if s.task:
        await s.task
    name = f"{job['company']} – {job['role']}"
    if s.status != "ready":
        await inline_apply.close_session(s.id)
        await send_text(phone, f"couldn't fill {job['company']} from here ({s.detail.lower() or s.status}). "
                               f"finish it yourself:\n{job['apply_url']}")
        return
    auto: dict[str, str] = {}
    queue: list[str] = []
    for f in s.fields:
        if f["kind"] == "file" or str(f["value"]).strip():
            continue
        if f.get("suggested"):
            # Fact, past, and essay drafts are all pre-filled; user only reviews at "submit".
            auto[f["id"]] = f["suggested"]
        elif f["required"]:
            queue.append(f["id"])
    _applies[user_id] = {"sid": s.id, "queue": queue, "answers": dict(auto), "auto": list(auto),
                         "profile": info.get("profile") or {}}
    drafts = sum(1 for f in s.fields if f.get("source") == "draft" and f["id"] in auto)
    extra = f" and answered {len(auto)} more for you" if auto else ""
    if drafts:
        extra += f" ({drafts} essay draft{'s' if drafts != 1 else ''})"
    if queue:
        await send_text(phone, f"filled {name}{extra}. {len(queue)} question"
                               f"{'s' if len(queue) > 1 else ''} left for you - reply \"cancel\" anytime.")
        await send_text(phone, ask_text(s, queue[0], 1, len(queue)))
        return
    await send_text(phone, ready_text(s, _applies[user_id]))


def _field(s: Any, fid: str) -> dict[str, Any]:
    return next(f for f in s.fields if f["id"] == fid)


def ask_text(s: Any, fid: str, n: int, total: int) -> str:
    f = _field(s, fid)
    head = f"{n}/{total}: {f['label'][:300]}"
    if f["kind"] == "checkbox":
        return f"{head}\n(yes / no)"
    opts = f.get("options") or []
    if opts:
        shown = "\n".join(f"{i}. {o[:70]}" for i, o in enumerate(opts[:8], 1))
        more = "\n(or type your answer)" if len(opts) > 8 else ""
        return f"{head}\n{shown}{more}"
    if f.get("source") == "draft" and f.get("suggested"):
        return f"{head}\n\nmy draft:\n{f['suggested']}\n\nreply \"ok\" to use it, or type your own."
    return f"{head}\n(type your answer, or \"write it\" and i'll draft one)"


def ready_text(s: Any, pending: dict[str, Any]) -> str:
    job, answers = s.job, pending["answers"]
    filled = [f for f in s.fields if f["kind"] != "file" and (answers.get(f["id"]) or str(f["value"]).strip())]
    shown = {"true": "yes", "": "no"}

    def lines(ids: list[str]) -> str:
        return "\n".join(f"• {_field(s, k)['label'][:45]}: {shown.get(answers[k], answers[k])[:60]}" for k in ids[:8])

    auto = [k for k in pending.get("auto", []) if k in answers]
    mine = [k for k in answers if k not in auto]
    return (f"{job['company']} – {job['role']} is ready: {len(filled)} answers + your resume."
            + (f"\n\ni answered these for you - check them:\n{lines(auto)}" if auto else "")
            + (f"\n\nyour answers:\n{lines(mine)}" if mine else "")
            + "\n\nreply \"submit\" to send it, or \"cancel\".")


def parse_answer(f: dict[str, Any], reply: str) -> Optional[str]:
    r = reply.strip()
    if f["kind"] == "checkbox":
        return "true" if re.match(r"^(y|yes|yeah|yep|sure|ok|true|check)", r, re.I) else (
            "" if re.match(r"^(n|no|nope|nah|false)", r, re.I) else None)
    opts = f.get("options") or []
    if not opts:
        return r or None
    if r.isdigit() and 0 < int(r) <= len(opts):
        return opts[int(r) - 1]
    low = r.lower()
    for test in (lambda o: o.lower() == low, lambda o: o.lower().startswith(low), lambda o: low in o.lower()):
        hits = [o for o in opts if test(o)]
        if len(hits) == 1:
            return hits[0]
    return r if f["kind"] == "combobox" and len(opts) > 8 else None


async def write_answer(question: str, job: dict[str, Any], profile: dict[str, Any]) -> str:
    from openai import AsyncOpenAI

    cfg = chat.llm_config()
    if not cfg:
        return ""
    client = AsyncOpenAI(api_key=cfg["key"], base_url=cfg["base"])
    facts = json.dumps({k: v for k, v in profile.items() if v}, ensure_ascii=False)[:4000]
    res = await client.chat.completions.create(
        model=cfg["model"], max_tokens=1200, temperature=0.5,
        extra_body={"reasoning_effort": "low"} if "gpt-oss" in cfg["model"] else None,
        messages=[{"role": "system", "content": (
            "Write a job application answer in the applicant's voice: 2-4 sentences, specific, first person, "
            "no fluff. Use only these facts about them; never invent experience.\nFACTS: " + facts)},
            {"role": "user", "content": f"{job['company']} – {job['role']}\nQuestion: {question}"}])
    return (res.choices[0].message.content or "").strip()


async def apply_action(phone: str, user_id: str, text: str) -> Optional[str]:
    pending = _applies.get(user_id)
    if not pending:
        return None
    t = text.strip().lower()
    sid = pending["sid"]
    s = inline_apply.SESSIONS.get(sid)
    if not s or not s.page:
        _applies.pop(user_id, None)
        return "that form timed out. say \"apply\" again and i'll refill it."
    s.touched = time.time()
    if re.match(r"^(cancel|stop)\b", t):
        _applies.pop(user_id, None)
        await inline_apply.close_session(sid)
        return "cancelled, nothing was sent."
    queue: list[str] = pending["queue"]
    if queue:
        f = _field(s, queue[0])
        if not f.get("options") and f["kind"] != "checkbox" and re.match(r"^(write it|you write|draft it|idk)", t):
            draft = await write_answer(f["label"], s.job, pending["profile"])
            if not draft:
                return "couldn't draft that one. type your answer?"
            pending["draft"] = draft
            return f"how's this?\n\n{draft}\n\nreply \"ok\" to use it, or type your own."
        offered = pending.get("draft") or (f.get("suggested") if f.get("source") == "draft" else None)
        if offered and re.match(r"^(ok|okay|yes|use it|good|perfect|looks good)\b", t):
            pending.pop("draft", None)
            value: Optional[str] = offered
        else:
            pending.pop("draft", None)
            value = parse_answer(f, text)
        done = len(pending["answers"]) - len(pending.get("auto", []))
        if value is None:
            return "didn't catch that - reply with the number or the option.\n\n" + ask_text(
                s, queue[0], done + 1, done + len(queue))
        pending["answers"][queue.pop(0)] = value
        if queue:
            return ask_text(s, queue[0], done + 2, done + 1 + len(queue))
        return ready_text(s, pending)
    if not re.match(r"^(submit|send it|yes,? submit)\b", t):
        return None
    _applies.pop(user_id, None)
    await send_text(phone, "submitting...")
    await inline_apply._submit(s, pending["answers"])
    job = s.job
    if s.status == "submitted":
        await record_application(user_id, job)
        await inline_apply.close_session(sid)
        return f"done - {job['company']} confirmed your application. it's in your tracker."
    problems = "; ".join(s.errors[:3]) or s.detail
    await inline_apply.close_session(sid)
    return f"{job['company']} didn't confirm it ({problems[:200]}). finish it here:\n{_finish_link(job['id'])}"


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
    local = await apply_action(phone, user_id, text) or await draft_action(text, state)
    if local:
        state["messages"].append(_msg("agent", local))
        await send_text(phone, local)
        await save_chat(user_id, state)
        return

    cfg = chat.llm_config()
    if not cfg:
        return
    snapshot = state["context"].get("snapshot") or {}
    context = {k: v for k, v in snapshot.items() if k != "apply"}
    if not context:
        context["note"] = (f"this user's profile hasn't synced yet, so you know nothing about their jobs, "
                           f"applications or offers. say so in one line and ask them to open {SITE_URL} once.")
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
            _spawn(run_apply(phone, user_id, ev["job_id"], snapshot))
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


def e164(raw: str) -> Optional[str]:
    digits = re.sub(r"\D", "", raw or "")
    if raw.strip().startswith("+") and 8 <= len(digits) <= 15:
        return "+" + digits
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    return None


async def add_contact(phone: str) -> None:
    """Free Sendblue lines only reply to saved contacts, so add the user's phone for them."""
    cfg = sendblue_config() or {}
    try:
        await http("POST", f"{SENDBLUE}/api/v2/contacts", _sb_headers(),
                   {"number": phone, "sendblue_number": cfg.get("number"), "tags": ["interlink"],
                    "update_if_exists": True})
    except RuntimeError as e:
        if "→ 409" in str(e):
            return
        log(f"add contact •••{phone[-4:]} failed: {e}")
        if re.search(r"→ (400|402|403|429)", str(e)):
            raise HTTPException(409, "InterLink's iMessage line is full right now. Try again later.") from None
        raise HTTPException(502, "Couldn't reach iMessage right now. Try again in a minute.") from None


class LinkRequest(BaseModel):
    phone: str = ""


@router.post("/link-code")
async def link_code(body: LinkRequest = LinkRequest(), user: str = Depends(require_user)) -> dict[str, Any]:
    cfg = sendblue_config()
    if not enabled() or not cfg or user == "local":
        raise HTTPException(503, "iMessage isn't set up on this server.")
    if body.phone:
        phone = e164(body.phone)
        if not phone:
            raise HTTPException(400, "Enter a full phone number, like (555) 123-4567.")
        await add_contact(phone)
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
