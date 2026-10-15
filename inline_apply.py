"""
In-app apply: the employer's form runs in a hidden browser; the InterLink site
shows its fields, pre-filled from the user's profile, and the user edits and
submits from InterLink. Only single-page forms (Greenhouse, Lever, Ashby and
similar) are supported — multi-step flows like Workday report `unsupported`.

Session lifecycle:  starting → filling → ready → submitting → submitted | error
                    (or unsupported / blocked, which offer the employer link instead)
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from pydantic import BaseModel

import answers
import queue_runner as qr
from auth import require_user

router = APIRouter(prefix="/api/apply")

CONFIRMATION = re.compile(
    r"thank(s| you) for (applying|your (application|interest))|application (has been |was )?(received|submitted)"
    r"|we('ve| have) received your application|successfully (submitted|applied)",
    re.I,
)
SESSION_DIR = Path(qr.agent._ROOT) / ".inline-sessions"
SESSION_TTL = 20 * 60
MAX_SESSIONS_PER_USER = 2
# Each open session holds a browser tab; small hosts (512 MB) can only afford a couple.
MAX_TOTAL_SESSIONS = int(os.environ.get("AGENT_MAX_SESSIONS", "6"))


class Session:
    def __init__(self, sid: str, job: dict, owner: str):
        self.id = sid
        self.job = job
        self.owner = owner
        self.touched = time.time()
        self.context: Optional[BrowserContext] = None
        self.status = "starting"
        self.detail = ""
        self.fields: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.screenshot: str = ""
        self.page: Optional[Page] = None
        self.resume_name = ""
        self.task: Optional[asyncio.Task[None]] = None
        self.learn: list[tuple[str, str]] = []

    def public(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "detail": self.detail,
            "fields": self.fields,
            "errors": self.errors,
            "screenshot": self.screenshot,
            "resume_name": self.resume_name,
            "apply_url": self.job.get("apply_url"),
        }


SESSIONS: dict[str, Session] = {}
_pw = None
_browser: Optional[Browser] = None
# The fill logic reads the module-level PROFILE, so only one session may fill at a time.
_fill_lock = asyncio.Lock()


async def _new_context() -> BrowserContext:
    """A fresh context per session so cookies never carry over between users."""
    global _pw, _browser
    if _browser is None or not _browser.is_connected():
        _pw = _pw or await async_playwright().start()
        _browser = await _pw.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--no-zygote",
                "--disable-extensions",
                "--js-flags=--max-old-space-size=192",
            ],
        )
    return await _browser.new_context(
            viewport={"width": 1280, "height": 1000},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            ),
        )


async def _close(s: Session) -> None:
    if s.context:
        try:
            await s.context.close()
        except Exception:
            pass
    s.context = None
    s.page = None
    for f in SESSION_DIR.glob(f"{s.id}-*"):
        f.unlink(missing_ok=True)


async def _reap() -> None:
    now = time.time()
    for sid, s in list(SESSIONS.items()):
        if now - s.touched > SESSION_TTL:
            SESSIONS.pop(sid, None)
            await _close(s)


def _owned(sid: str, user: str) -> Session:
    s = SESSIONS.get(sid)
    if not s or s.owner != user:
        raise HTTPException(404, "session not found")
    s.touched = time.time()
    return s


def _use_profile(profile: dict, resume: Optional[Path]) -> None:
    qr.PROFILE.clear()
    qr.PROFILE.update(qr.agent._PROFILE_DEFAULTS)
    qr.PROFILE.update({k: v for k, v in profile.items() if v is not None})
    qr.agent.RESUME_CANDIDATES[:] = [resume] if resume else []


async def _shot(page: Page) -> str:
    try:
        png = await page.screenshot(type="jpeg", quality=55, full_page=False)
        return "data:image/jpeg;base64," + base64.b64encode(png).decode()
    except Exception:
        return ""


async def _collect(page: Page) -> list[dict[str, Any]]:
    """Visible fields with their current (auto-filled) values and choices."""
    data = await page.evaluate(qr.COLLECT_FIELDS_JS)
    fields: list[dict[str, Any]] = []
    for f in data["fields"]:
        if f["kind"] == "input" and re.fullmatch(r"select\W*", f["label"].strip(), re.I):
            continue
        loc = page.locator(f"[data-il='{f['idx']}']")
        entry = {
            "id": str(f["idx"]),
            "kind": f["kind"],
            "label": re.sub(r"\s*[*✱]\s*$", "", f["label"]).strip(),
            "required": bool(f["required"]),
            "value": f["value"],
            "options": f["options"],
        }
        if f["kind"] == "select":
            entry["value"] = await loc.evaluate(
                "e => e.selectedIndex > 0 ? e.options[e.selectedIndex].text.trim() : ''"
            )
        elif f["kind"] == "combobox":
            try:
                iid = await loc.get_attribute("id")
                sel = f'[id^="react-select-{iid}-option"]' if iid else "[role=option]"
                await loc.focus(timeout=2000)
                await page.keyboard.press("ArrowDown")
                await asyncio.sleep(0.4)
                entry["options"] = [t.strip() for t in (await page.locator(sel).all_inner_texts())[:400] if t.strip()]
                await page.keyboard.press("Escape")
            except Exception:
                pass
        elif f["kind"] == "checkbox":
            entry["value"] = "true" if f["value"] else ""
        fields.append(entry)
    for g in data["radios"]:
        checked = next((o["option"] for o in g["opts"] if o["checked"]), "")
        fields.append({
            "id": "r" + str(g["opts"][0]["idx"]),
            "kind": "radio",
            "label": g["question"],
            "required": True,
            "value": checked,
            "options": [o["option"] for o in g["opts"]],
            "optionIds": [str(o["idx"]) for o in g["opts"]],
        })
    files = await page.locator("input[type=file]").count()
    if files:
        fields.insert(0, {"id": "resume", "kind": "file", "label": "Resume", "required": True,
                          "value": qr.resume_path().name if qr.resume_path() else "", "options": []})
    return fields


async def _set_value(page: Page, field: dict, value: str) -> None:
    kind = field["kind"]
    if kind == "radio":
        idx = field["optionIds"][field["options"].index(value)]
        await page.locator(f"[data-il='{idx}']").check(timeout=4000, force=True)
        return
    loc = page.locator(f"[data-il='{field['id']}']")
    if kind == "select":
        await loc.select_option(label=value, timeout=4000)
    elif kind == "checkbox":
        await (loc.check if value else loc.uncheck)(timeout=4000)
    elif kind == "combobox":
        iid = await loc.get_attribute("id")
        sel = f'[id^="react-select-{iid}-option"]' if iid else "[role=option]"
        await loc.focus(timeout=4000)
        await loc.fill(value, timeout=4000)
        await asyncio.sleep(1.2)
        exact = re.compile(r"^\s*" + re.escape(value) + r"\s*$")
        opt = page.locator(sel).filter(has_text=exact)
        if await opt.count() == 0:
            opt = page.locator(sel)
        await opt.first.click(timeout=4000)
    else:
        await loc.fill(value, timeout=4000)


async def _prepare(s: Session, profile: dict, resume: Optional[Path]) -> None:
    url = (s.job.get("apply_url") or "").strip()
    if not url:
        s.status, s.detail = "unsupported", "This listing has no application link."
        return
    filled = False
    async with _fill_lock:
        _use_profile(profile, resume)
        s.context = await _new_context()
        page = await s.context.new_page()
        s.page = page
        s.status = "filling"
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            await asyncio.sleep(2)
            if "workday" in page.url or "myworkdayjobs" in page.url:
                s.status = "unsupported"
                s.detail = "Workday uses a multi-step account flow — apply on the company site."
                return
            ok, detail = await qr.fill_generic(page, s.job)
            s.screenshot = await _shot(page)
            if not ok:
                s.status, s.detail = "blocked", detail
                return
            s.fields = await _collect(page)
            s.detail = detail
            filled = True
        except Exception as e:
            s.status, s.detail = "error", f"{type(e).__name__}: {e}"[:300]
            s.screenshot = await _shot(page)
    if filled:
        s.status = "answering"
        for fid, hint in (await answers.suggest(s.fields, s.job, profile, s.owner)).items():
            field = next(f for f in s.fields if f["id"] == fid)
            field["suggested"], field["source"] = hint["answer"], hint["source"]
        s.status = "ready"


async def _submit(s: Session, answers_in: dict[str, str]) -> None:
    page = s.page
    assert page is not None
    s.status, s.errors = "submitting", []
    by_id = {f["id"]: f for f in s.fields}
    s.learn = [(by_id[fid]["label"], {"true": "yes"}.get(v, v)) for fid, v in answers_in.items()
               if fid in by_id and by_id[fid]["kind"] != "file" and v.strip()
               and not str(by_id[fid]["value"]).strip()]
    for fid, value in answers_in.items():
        f = by_id.get(fid)
        if not f or f["kind"] == "file" or value == f["value"]:
            continue
        try:
            await _set_value(page, f, value)
            f["value"] = value
        except Exception as e:
            s.errors.append(f"Couldn't set “{f['label'][:60]}”: {str(e).splitlines()[0][:120]}")

    clicked = False
    for sel in ["button:has-text('Submit application')", "button:has-text('Submit Application')",
                "button[type=submit]", "input[type=submit]", "button:has-text('Submit')"]:
        btn = page.locator(sel).first
        try:
            if await btn.is_visible(timeout=1500):
                await btn.scroll_into_view_if_needed()
                await btn.click(timeout=5000)
                clicked = True
                break
        except Exception:
            continue
    if not clicked:
        s.status, s.detail = "error", "Couldn't find the form's Submit button."
        s.screenshot = await _shot(page)
        return

    for _ in range(20):
        await asyncio.sleep(1)
        try:
            text = await page.inner_text("body", timeout=3000)
        except Exception:
            text = ""
        if CONFIRMATION.search(text):
            s.status, s.detail = "submitted", "The employer confirmed your application."
            s.screenshot = await _shot(page)
            await answers.remember(s.owner, s.job.get("company", ""), s.learn)
            return
    problems = await page.evaluate(
        """() => [...document.querySelectorAll('[aria-invalid=true], .error, .field-error, [class*="error" i]')]
             .map(e => (e.innerText || e.getAttribute('aria-label') || '').trim())
             .filter(t => t && t.length < 200).slice(0, 8)"""
    )
    s.errors += [p for p in dict.fromkeys(problems)]
    if await page.locator("iframe[src*='hcaptcha'], iframe[src*='recaptcha']").count() and not problems:
        s.errors.append("The site may be asking for a CAPTCHA — finish on the company site.")
    s.screenshot = await _shot(page)
    s.fields = await _collect(page)
    s.status = "error"
    s.detail = "The form didn't confirm the submission. Fix the highlighted answers and try again."


class StartRequest(BaseModel):
    job: dict[str, Any]
    profile: dict[str, Any]
    resume_b64: Optional[str] = None
    resume_name: Optional[str] = None


class SubmitRequest(BaseModel):
    answers: dict[str, str]


async def open_session(job: dict, profile: dict, resume: Optional[bytes], resume_name: Optional[str],
                       owner: str) -> Session:
    """Registers a session and starts filling it in the background. Raises 503 when the host is full."""
    await _reap()
    mine = [x for x in SESSIONS.values() if x.owner == owner]
    for old in sorted(mine, key=lambda x: x.touched)[: max(0, len(mine) - MAX_SESSIONS_PER_USER + 1)]:
        SESSIONS.pop(old.id, None)
        await _close(old)
    if len(SESSIONS) >= MAX_TOTAL_SESSIONS:
        raise HTTPException(503, "The agent is busy with other applications. Try again in a minute.")
    sid = uuid.uuid4().hex[:12]
    s = Session(sid, job, owner)
    path = None
    if resume:
        SESSION_DIR.mkdir(exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9._-]", "_", resume_name or "resume.pdf") or "resume.pdf"
        path = SESSION_DIR / f"{sid}-{name}"
        path.write_bytes(resume)
        s.resume_name = name
    SESSIONS[sid] = s
    s.task = asyncio.create_task(_prepare(s, profile, path))
    return s


async def close_session(sid: str) -> None:
    s = SESSIONS.pop(sid, None)
    if s:
        await _close(s)


@router.post("/start")
async def start(req: StartRequest, user: str = Depends(require_user)):
    resume = base64.b64decode(req.resume_b64) if req.resume_b64 else None
    s = await open_session(req.job, req.profile, resume, req.resume_name, user)
    return s.public()


@router.get("/{sid}")
async def status(sid: str, user: str = Depends(require_user)):
    return _owned(sid, user).public()


@router.post("/{sid}/submit")
async def submit(sid: str, req: SubmitRequest, user: str = Depends(require_user)):
    s = _owned(sid, user)
    if not s.page:
        raise HTTPException(404, "session not found")
    if s.status not in ("ready", "error"):
        raise HTTPException(409, f"can't submit while {s.status}")
    asyncio.create_task(_submit(s, req.answers))
    s.status = "submitting"
    return s.public()


@router.post("/{sid}/cancel")
async def cancel(sid: str, user: str = Depends(require_user)):
    s = SESSIONS.get(sid)
    if s and s.owner == user:
        SESSIONS.pop(sid, None)
        await _close(s)
    return {"ok": True}
