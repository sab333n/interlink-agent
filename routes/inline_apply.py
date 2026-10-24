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

import services.answers as answers
import queue_runner as qr
from services.auth import require_user
from services.posthog_client import get_posthog_client

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
        self.submission_captured = False

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


_PLACEHOLDER = re.compile(r"^(select|choose)([.…\s].*)?$", re.I)


def _real_choice(value: str, options: list[str]) -> Optional[str]:
    text = (value or "").strip()
    if not text or _PLACEHOLDER.match(text):
        return None
    if options:
        return qr.pick_option(options, text) or answers._match_options(options, text)
    return text


async def _refresh_select_options(page: Page, fields: list[dict[str, Any]]) -> None:
    """Open each blank select/combobox once so Greenhouse react-select options are known."""
    for f in fields:
        if f.get("kind") not in ("select", "combobox"):
            continue
        if f.get("options") and str(f.get("value") or "").strip():
            continue
        loc = page.locator(f"[data-il='{f['id']}']")
        try:
            if f["kind"] == "select":
                opts = await loc.evaluate(
                    """e => [...e.options].filter(o => o.value !== '' && !/^select/i.test((o.text||'').trim()))
                       .map(o => (o.text||'').replace(/\\s+/g,' ').trim())"""
                )
                if opts:
                    f["options"] = opts
                continue
            iid = await loc.get_attribute("id")
            sel = f'[id^="react-select-{iid}-option"]' if iid else "[role=option]"
            await loc.scroll_into_view_if_needed(timeout=2000)
            try:
                await loc.click(timeout=2000)
            except Exception:
                await loc.focus(timeout=2000)
            await page.keyboard.press("ArrowDown")
            await asyncio.sleep(0.55)
            opts = [t.strip() for t in await page.locator(sel).all_inner_texts() if t.strip()]
            opts = [o for o in opts if not _PLACEHOLDER.match(o)]
            if opts:
                f["options"] = opts[:400]
            await page.keyboard.press("Escape")
            await asyncio.sleep(0.15)
        except Exception:
            try:
                await page.keyboard.press("Escape")
            except Exception:
                pass


async def _set_value(page: Page, field: dict, value: str) -> None:
    kind = field["kind"]
    label = str(field.get("label") or "")
    # File / resume controls are attached separately — never .fill() them.
    if kind == "file" or re.search(r"^\s*resume\b|^\s*cv\b|attach.*(resume|cv)", label, re.I):
        rp = qr.resume_path()
        if not rp:
            return
        loc = page.locator("input[type=file]").first
        if field.get("id") not in ("", None, "resume"):
            try:
                loc = page.locator(f"[data-il='{field['id']}']")
            except Exception:
                pass
        # Prefer the resume-named file input when multiple exist.
        inputs = page.locator("input[type=file]")
        n = await inputs.count()
        target = None
        for i in range(n):
            meta = await inputs.nth(i).evaluate(
                "el => (el.name + ' ' + el.id + ' ' + (el.closest('div,fieldset,label')?.innerText || '')).toLowerCase()"
            )
            if "cover" in meta:
                continue
            target = inputs.nth(i)
            if "resume" in meta or "cv" in meta:
                break
        if target is None:
            target = loc
        await target.set_input_files(str(rp))
        await asyncio.sleep(1.0)
        return

    opts = list(field.get("options") or [])
    choice = _real_choice(value, opts)
    if kind != "checkbox" and choice is None:
        raise ValueError("empty or placeholder — pick a real option")
    if kind == "radio":
        idx = field["optionIds"][field["options"].index(choice)]
        await page.locator(f"[data-il='{idx}']").check(timeout=4000, force=True)
        return
    loc = page.locator(f"[data-il='{field['id']}']")
    if kind == "select":
        try:
            await loc.select_option(label=choice, timeout=4000)
        except Exception:
            try:
                await loc.select_option(label=re.compile(re.escape(choice), re.I), timeout=4000)
            except Exception:
                # Fall back to value attribute match.
                await loc.select_option(value=re.compile(re.escape(choice), re.I), timeout=4000)
    elif kind == "checkbox":
        await (loc.check if value else loc.uncheck)(timeout=4000)
    elif kind == "combobox":
        iid = await loc.get_attribute("id")
        sel = (
            f'[id^="react-select-{iid}-option"], #{iid}-listbox [role=option], [role=listbox] [role=option]'
            if iid
            else "[role=listbox] [role=option], [role=option]"
        )
        # Click the visible control, not only the hidden input.
        clicked = False
        for sel_ctl in [
            f"#{iid}",
            f"[data-il='{field['id']}']",
        ]:
            try:
                el = page.locator(sel_ctl).first
                box = el.locator(
                    "xpath=ancestor-or-self::*[contains(@class,'select__control') or contains(@class,'select-container') or contains(@class,'Select')][1]"
                )
                if await box.count():
                    await box.first.click(timeout=2500)
                    clicked = True
                    break
                await el.click(timeout=2500)
                clicked = True
                break
            except Exception:
                continue
        if not clicked:
            await loc.focus(timeout=4000)
        await asyncio.sleep(0.2)
        try:
            await loc.fill("", timeout=2000)
            await loc.fill(choice, timeout=3000)
        except Exception:
            # Some Greenhouse comboboxes are not fillable — type via keyboard.
            await page.keyboard.type(choice, delay=20)
        await asyncio.sleep(0.9)
        exact = re.compile(r"^\s*" + re.escape(choice) + r"\s*$", re.I)
        opt = page.locator(sel).filter(has_text=exact)
        if await opt.count() == 0:
            opt = page.locator(sel).filter(has_text=re.compile(re.escape(choice), re.I))
        if await opt.count() == 0:
            opt = page.locator(sel)
        if await opt.count() == 0:
            await page.keyboard.press("ArrowDown")
            await asyncio.sleep(0.2)
            await page.keyboard.press("Enter")
        else:
            await opt.first.click(timeout=4000)
        await page.keyboard.press("Escape")
    else:
        # Never .fill() a file input that slipped through with kind=input.
        tag = await loc.evaluate("e => (e.tagName + ':' + (e.type || '')).toLowerCase()")
        if "file" in tag:
            rp = qr.resume_path()
            if rp:
                await loc.set_input_files(str(rp))
            return
        await loc.fill(choice, timeout=4000)


def _norm_label(label: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\s*[*✱]\s*$", "", label or "").strip().lower())


def _find_field(fields: list[dict[str, Any]], label: str) -> Optional[dict[str, Any]]:
    want = _norm_label(label)
    if not want:
        return None
    for f in fields:
        have = _norm_label(f.get("label") or "")
        if have == want or have.startswith(want[:48]) or want.startswith(have[:48]):
            return f
    for f in fields:
        have = _norm_label(f.get("label") or "")
        if want in have or have in want:
            return f
    return None


async def _scroll_form(page: Page) -> None:
    try:
        await page.evaluate(
            """async () => {
              const step = Math.max(300, Math.floor(window.innerHeight * 0.7));
              for (let y = 0; y < document.body.scrollHeight; y += step) {
                window.scrollTo(0, y);
                await new Promise((r) => setTimeout(r, 80));
              }
              window.scrollTo(0, 0);
            }"""
        )
    except Exception:
        pass


async def _apply_hints_to_page(
    page: Page, fields: list[dict[str, Any]], hints: dict[str, dict[str, str]]
) -> int:
    n = 0
    for fid, hint in hints.items():
        field = next((f for f in fields if f["id"] == fid), None)
        if not field or field.get("kind") == "file":
            continue
        answer = hint["answer"]
        field["suggested"], field["source"] = answer, hint["source"]
        if not str(field.get("value") or "").strip():
            field["value"] = answer
        try:
            await _set_value(page, field, answer)
            field["value"] = answer
            n += 1
        except Exception as e:
            print(f"  ! auto-fill “{field.get('label', '')[:50]}”: {e}", flush=True)
    return n


async def _fill_required_gaps(page: Page, s: Session, profile: dict) -> int:
    """Find anything the employer still marks required and fill from profile/fallbacks."""
    await _scroll_form(page)
    fields = await _collect(page)
    await _refresh_select_options(page, fields)
    # Index by current ids for suggest.
    s.fields = fields
    hints = await answers.suggest(
        [f for f in fields if not str(f.get("value") or "").strip()],
        s.job,
        profile,
        s.owner,
    )
    filled = await _apply_hints_to_page(page, fields, hints)

    left = await page.evaluate(qr.REQUIRED_LEFT_JS) or []
    fields = await _collect(page)
    await _refresh_select_options(page, [f for f in fields if not str(f.get("value") or "").strip()])
    for lab in left:
        field = _find_field(fields, lab)
        if not field:
            # Synthesize a minimal field so fallbacks still run after we locate the control.
            continue
        if field.get("kind") == "file":
            continue
        if str(field.get("value") or "").strip() and not _PLACEHOLDER.match(str(field.get("value"))):
            continue
        hint = answers._fallback(field, s.job, profile)
        if not hint:
            # Last resort for yes/no-looking labels.
            yn = answers._yn_from_profile(_norm_label(lab), profile)
            if yn and field.get("options"):
                pick = answers._pick_yes_no(field["options"], yn)
                if pick:
                    hint = {"answer": pick, "source": "fact"}
            elif yn:
                hint = {"answer": yn.capitalize(), "source": "fact"}
        if not hint:
            continue
        answer = answers._clean(field, hint["answer"]) or hint["answer"]
        try:
            await _set_value(page, field, answer)
            field["value"] = answer
            field["suggested"] = answer
            field["source"] = hint["source"]
            filled += 1
            print(f"  ✓ gap “{lab[:60]}” → {answer[:80]}", flush=True)
        except Exception as e:
            print(f"  ! gap “{lab[:50]}”: {e}", flush=True)
    s.fields = await _collect(page)
    # Carry suggestions onto the fresh collect for the review UI.
    by_label = {_norm_label(f["label"]): f for f in fields if f.get("suggested")}
    for f in s.fields:
        prev = by_label.get(_norm_label(f["label"]))
        if prev and not str(f.get("value") or "").strip():
            f["value"] = prev.get("value") or prev.get("suggested") or ""
            f["suggested"] = prev.get("suggested") or f["value"]
            f["source"] = prev.get("source")
        elif prev and prev.get("suggested"):
            f["suggested"] = prev["suggested"]
            f["source"] = prev.get("source")
            if not str(f.get("value") or "").strip():
                f["value"] = prev["suggested"]
    return filled


async def _prepare(s: Session, profile: dict, resume: Optional[Path]) -> None:
    url = (s.job.get("apply_url") or "").strip()
    if not url:
        s.status, s.detail = "unsupported", "This listing has no application link."
        return
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
            s.detail = detail
            s.status = "answering"
            await _scroll_form(page)
            s.fields = await _collect(page)
            await _refresh_select_options(page, s.fields)
            hints = await answers.suggest(s.fields, s.job, profile, s.owner)
            n = await _apply_hints_to_page(page, s.fields, hints)
            print(f"  · applied {n} suggested answers to the live form", flush=True)
            gaps = await _fill_required_gaps(page, s, profile)
            print(f"  · filled {gaps} required gaps", flush=True)
            s.screenshot = await _shot(page)
            s.status = "ready"
        except Exception as e:
            s.status, s.detail = "error", f"{type(e).__name__}: {e}"[:300]
            s.screenshot = await _shot(page)


async def _submit(s: Session, answers_in: dict[str, str]) -> None:
    page = s.page
    assert page is not None
    s.status, s.errors = "submitting", []
    # Re-collect so data-il indexes match the live DOM after earlier fills.
    s.fields = await _collect(page)
    await _refresh_select_options(page, s.fields)
    by_id = {f["id"]: f for f in s.fields}

    # Map incoming answers (possibly keyed to older field ids) onto current fields by label.
    mapped: list[tuple[dict[str, Any], str]] = []
    for fid, value in answers_in.items():
        f = by_id.get(fid)
        if not f or f.get("kind") == "file":
            continue
        mapped.append((f, value))
    # Also accept any suggested/value already on fields the UI didn't send.
    for f in s.fields:
        if f.get("kind") == "file":
            continue
        if any(f["id"] == m[0]["id"] for m in mapped):
            continue
        raw = str(f.get("value") or f.get("suggested") or "").strip()
        if raw:
            mapped.append((f, raw))

    # Ensure resume stays attached (Greenhouse sometimes clears it on re-render).
    try:
        await _set_value(page, {"id": "resume", "kind": "file", "label": "Resume"}, "resume")
    except Exception as e:
        print(f"  ! resume re-attach: {e}", flush=True)

    s.learn = []
    for f, value in mapped:
        if f.get("kind") == "file":
            continue
        choice = _real_choice(value, f.get("options") or [])
        if choice is None:
            continue
        s.learn.append((f["label"], {"true": "yes"}.get(choice, choice)))
        try:
            await _set_value(page, f, choice)
            f["value"] = choice
        except Exception as e:
            msg = str(e).splitlines()[0][:120]
            # Don't surface file-input fill noise as a user-facing blocker.
            if "Locator.fill" in msg and re.search(r"resume|file", f.get("label", ""), re.I):
                print(f"  ! skip resume fill noise: {msg}", flush=True)
                continue
            s.errors.append(f"Couldn't set “{f['label'][:60]}”: {msg}")

    # Final automatic pass for anything still required on the employer form.
    profile = qr.PROFILE
    await _fill_required_gaps(page, s, profile)

    left = await page.evaluate(qr.REQUIRED_LEFT_JS) or []
    if left:
        # One more suggest pass for stubborn blanks.
        s.fields = await _collect(page)
        await _refresh_select_options(page, s.fields)
        hints = await answers.suggest(
            [f for f in s.fields if not str(f.get("value") or "").strip()],
            s.job,
            profile,
            s.owner,
        )
        await _apply_hints_to_page(page, s.fields, hints)
        left = await page.evaluate(qr.REQUIRED_LEFT_JS) or []

    if left:
        s.errors += [f"{lab}: still required" for lab in left]
        s.screenshot = await _shot(page)
        s.fields = await _collect(page)
        for f in s.fields:
            if _PLACEHOLDER.match(str(f.get("value") or "").strip()):
                f["value"] = ""
        s.status = "error"
        s.detail = (
            "Some required employer questions are still blank after auto-fill. "
            "Edit the highlighted answers below and try again."
        )
        return

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
        """() => {
          const clean = (t) => (t || '').replace(/\\s+/g, ' ').trim();
          const labelNear = (el) => {
            const q = el.closest('.application-question, .field, [class*="field"], fieldset, label');
            const lab = q?.querySelector('.application-label, legend, label, [class*="label" i]');
            return clean(lab?.innerText || '').replace(/\\s*[*✱]\\s*$/, '').slice(0, 80);
          };
          const out = [];
          for (const e of document.querySelectorAll('[aria-invalid=true], .error, .field-error, [class*="error" i], [class*="invalid" i]')) {
            let t = clean(e.innerText || e.getAttribute('aria-label') || '');
            if (!t || t.length > 200) continue;
            if (/^(select|choose)([.…\\s].*)?$/i.test(t)) {
              const lab = labelNear(e);
              t = lab ? `${lab}: choose an option` : 'A required dropdown is still on Select…';
            }
            if (t && !out.includes(t)) out.push(t);
            if (out.length >= 8) break;
          }
          return out;
        }"""
    )
    left = await page.evaluate(qr.REQUIRED_LEFT_JS) or []
    for lab in left:
        msg = f"{lab}: still required"
        if msg not in problems:
            problems.append(msg)
    s.errors += [p for p in dict.fromkeys(problems) if p and not _PLACEHOLDER.match(p)]
    if await page.locator("iframe[src*='hcaptcha'], iframe[src*='recaptcha']").count() and not s.errors:
        s.errors.append("The site may be asking for a CAPTCHA — finish on the company site.")
    s.screenshot = await _shot(page)
    s.fields = await _collect(page)
    for f in s.fields:
        if _PLACEHOLDER.match(str(f.get("value") or "").strip()):
            f["value"] = ""
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
async def start(
    req: StartRequest,
    user: str = Depends(require_user),
    posthog_client=Depends(get_posthog_client),
):
    resume = base64.b64decode(req.resume_b64) if req.resume_b64 else None
    s = await open_session(req.job, req.profile, resume, req.resume_name, user)
    if posthog_client:
        posthog_client.capture("application_started", properties={"has_resume": bool(resume)})
    return s.public()


@router.get("/{sid}")
async def status(
    sid: str,
    user: str = Depends(require_user),
    posthog_client=Depends(get_posthog_client),
):
    s = _owned(sid, user)
    if s.status == "submitted" and not s.submission_captured and posthog_client:
        posthog_client.capture("application_submitted")
        s.submission_captured = True
    return s.public()


@router.post("/{sid}/submit")
async def submit(
    sid: str,
    req: SubmitRequest,
    user: str = Depends(require_user),
    posthog_client=Depends(get_posthog_client),
):
    s = _owned(sid, user)
    if not s.page:
        raise HTTPException(404, "session not found")
    if s.status not in ("ready", "error"):
        raise HTTPException(409, f"can't submit while {s.status}")
    asyncio.create_task(_submit(s, req.answers))
    s.status = "submitting"
    if posthog_client:
        posthog_client.capture("application_submission_requested")
    return s.public()


@router.post("/{sid}/cancel")
async def cancel(sid: str, user: str = Depends(require_user)):
    s = SESSIONS.get(sid)
    if s and s.owner == user:
        SESSIONS.pop(sid, None)
        await _close(s)
    return {"ok": True}
