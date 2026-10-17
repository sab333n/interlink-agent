"""
Auto-apply queue: fills each queued application in its own browser tab and
stops before the final Submit so the user can review and submit by hand.

Inputs (env, set by server.py):
    JOBS_PATH    JSON list of {id, company, role, apply_url}
    PROFILE_PATH the signed-in user's profile (agent format)
    RESUME_PATH  the user's uploaded resume

Progress is reported on stdout as `@@JOB {json}` lines that server.py parses:
    status: running | ready_for_review | needs_manual | failed
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import date
from pathlib import Path

os.environ["AGENT_DRY_RUN"] = "1"
os.environ.pop("AGENT_SUBMIT", None)

import agent  # noqa: E402  (reads PROFILE_PATH / RESUME_PATH at import)
from playwright.async_api import Page, async_playwright  # noqa: E402

PROFILE = agent.PROFILE
JOB_TIMEOUT_SECS = int(os.environ.get("AGENT_JOB_TIMEOUT", "420"))

CAPTCHA_TEXT = re.compile(
    r"verify you are human|are you a robot|complete the security check|checking your browser|press (and|&) hold",
    re.I,
)
EEO_LABEL = re.compile(
    r"gender|sex\b|race|ethnic|hispanic|latino|veteran|disabilit|sexual orientation|transgender|pronoun",
    re.I,
)
ATS_HOST = re.compile(r"(^|\.)(greenhouse\.io|lever\.co|ashbyhq\.com|smartrecruiters\.com)$")
DECLINE_OPTION = re.compile(r"decline|prefer not|don.?t wish|do not wish|not to (say|answer|disclose)|choose not", re.I)


def emit(job_id: str, status: str, detail: str = "") -> None:
    print("@@JOB " + json.dumps({"id": job_id, "status": status, "detail": detail}), flush=True)


def resume_path() -> Path | None:
    return next((p for p in agent.RESUME_CANDIDATES if p.exists()), None)


def full_name() -> str:
    return f"{PROFILE.get('first_name', '')} {PROFILE.get('last_name', '')}".strip()


def latest_company() -> str:
    exp = PROFILE.get("work_experience") or []
    return exp[0].get("company", "") if exp else ""


def profile_answer(label: str) -> str | None:
    """Direct profile lookups for common short fields. None = not a known field."""
    q = label.lower()
    table: list[tuple[str, str]] = [
        (r"preferred (first )?name", PROFILE.get("first_name", "")),
        (r"first name|given name", PROFILE.get("first_name", "")),
        (r"last name|family name|surname", PROFILE.get("last_name", "")),
        (r"full name|^name\b|your name|legal name", full_name()),
        (r"e-?mail", PROFILE.get("email", "")),
        (r"phone|mobile", PROFILE.get("phone", "")),
        (r"linkedin", PROFILE.get("linkedin", "")),
        (r"github", PROFILE.get("github", "")),
        (r"portfolio|website|personal site|url", PROFILE.get("portfolio", "") or PROFILE.get("github", "")),
        (r"current (company|employer)|most recent (company|employer)", latest_company()),
        (r"current (title|role)|job title", PROFILE.get("current_title", "")),
        (r"degree", PROFILE.get("degree_full", "") or PROFILE.get("degree", "")),
        (r"major|discipline|field of study|area of study", PROFILE.get("major", "")),
        (r"graduat.*(year|date)|expected graduation|class of", PROFILE.get("graduation_year", "")),
        (r"school|university|college|institution", PROFILE.get("university", "")),
        (r"\bcountry\b", PROFILE.get("country", "") or "United States"),
        (r"\bcity\b", PROFILE.get("city", "")),
        (r"\bstate\b|province", PROFILE.get("state", "")),
        (r"address (line )?2|line 2|apt|suite|unit", ""),
        (r"address|street", PROFILE.get("address", "")),
        (r"notes?$|comments?$", ""),
        (r"location|where are you (based|located)", ", ".join(
            x for x in [PROFILE.get("city", ""), PROFILE.get("state", "")] if x
        )),
        (r"years of (professional )?experience", PROFILE.get("years_experience", "")),
        (r"how (did )?you (hear|heard)|source|referr", "Online job board"),
    ]
    for pattern, value in table:
        if re.search(pattern, q):
            return value or None
    return None


def yes_no(v: str) -> str | None:
    v = str(v or "").strip().lower()
    return "Yes" if v.startswith("y") else "No" if v.startswith("n") else None


def question_answer(label: str) -> str | None:
    """Answers for full-sentence questions, where short-field lookups would misfire."""
    q = label.lower()
    visa = str(PROFILE.get("visa_status", "")).lower()
    if re.search(r"\bcpt\b|\bopt\b|f-1|curricular practical|optional practical", q):
        if not visa:
            return None
        return "Yes" if re.search(r"f-1|cpt|opt", visa) else "No"
    if re.search(r"(visa|immigration|citizenship) status", q):
        if re.search(r"citizen|permanent resident|green card", visa):
            return "Not Applicable" if "not applicable" in q else PROFILE.get("visa_status", "") or None
        return PROFILE.get("visa_status", "") or None
    if re.search(r"relatives?|family member|close personal relationship", q) and re.search(r"employ|work", q):
        return "No"
    if re.search(r"sponsor", q):
        return yes_no(PROFILE.get("sponsorship", ""))
    if re.search(r"authori[sz]ed to work|eligible to work|legally (authori|able)|right to work", q):
        return yes_no(PROFILE.get("work_auth", ""))
    yn = agent.decide_yes_no(label)
    if yn:
        return yn
    if re.search(r"what city|which city", q):
        return PROFILE.get("city", "") or None
    if re.search(r"how (did )?you (hear|heard|find|found|learn)|where did you (hear|find)", q):
        return "Online job board"
    if re.search(r"years of .{0,20}experience", q):
        return str(PROFILE.get("years_experience", "")) or None
    grad = str(PROFILE.get("graduation_year", ""))
    if re.search(r"currently enrolled|currently a student|currently pursuing", q) and not re.search(r"what|which|when|where|how", q):
        return "Yes" if grad.isdigit() and int(grad) >= date.today().year else None
    if re.search(r"(what|which).{0,40}degree", q):
        return PROFILE.get("degree_full", "") or PROFILE.get("degree", "") or None
    if re.search(r"(what|which).{0,40}(major|field of study|discipline)", q):
        return PROFILE.get("major", "") or None
    if re.search(r"graduat", q) and re.search(r"when|what|expected|year|date", q):
        return grad or None
    if re.search(r"(what|which|name of).{0,40}(school|university|college)", q):
        return PROFILE.get("university", "") or None
    return None


def is_question(label: str) -> bool:
    return "?" in label or len(label) > 70


def answer_for(label: str) -> str | None:
    return question_answer(label) if is_question(label) else (profile_answer(label) or agent.decide_yes_no(label))


def ai_answer(label: str, job: dict, options: list[str] | None = None, long: bool = False) -> str:
    public_profile = {k: v for k, v in PROFILE.items() if k != "ats_account_password"}
    if options:
        instructions = (
            "Pick the single best option for this applicant. Reply with the exact option text only.\n"
            f"OPTIONS: {json.dumps(options)}"
        )
    elif long:
        instructions = (
            "Write the applicant's answer in first person, 60-120 words, specific to this company and role, "
            "using only facts from the profile. No placeholders, no greeting, no sign-off."
        )
    else:
        instructions = "Reply with only the short value to type. If the profile has no answer, reply N/A."
    prompt = (
        f"You are filling a job application for {job.get('role')} at {job.get('company')}.\n"
        f"APPLICANT PROFILE:\n{json.dumps(public_profile)[:6000]}\n\n"
        f"FORM QUESTION: {label}\n{instructions}"
    )
    return agent.grok_complete(prompt, max_tokens=320 if long else 60)


YES_OPT = re.compile(r"^\s*(yes|i am|i do|i will|i have|i can)\b(?!\s+not)", re.I)
NO_OPT = re.compile(r"^\s*(no\b|i am not|i do not|i don.?t|i will not|i won.?t|i have not|i cannot|i can.?t)", re.I)


def _norm_opt(o: str) -> str:
    return re.sub(r"\s*\+\d+$", "", o.strip()).lower()


def pick_option(options: list[str], want: str) -> str | None:
    w = want.strip().lower()
    if not w:
        return None
    if w in ("yes", "no"):
        rx = YES_OPT if w == "yes" else NO_OPT
        return next((o for o in options if rx.search(o)), None)
    for o in options:
        if _norm_opt(o) == w:
            return o
    if w == "united states of america":
        w = "united states"
        for o in options:
            if _norm_opt(o) == w:
                return o
    for o in options:
        ol = _norm_opt(o)
        if ol.startswith(w) or (w.startswith(ol) and len(ol) > 1):
            return o
    for o in options:
        if w in o.lower():
            return o
    stop = {"of", "the", "a", "an", "in", "and", "or", "to", "degree", "s"}
    want_words = {t for t in re.findall(r"[a-z0-9]+", w) if t not in stop}
    best, best_score = None, 0
    for o in options:
        words = {t.rstrip("s") for t in re.findall(r"[a-z0-9]+", o.lower()) if t not in stop}
        score = len({t.rstrip("s") for t in want_words} & words)
        if score > best_score:
            best, best_score = o, score
    return best


COLLECT_FIELDS_JS = r"""
() => {
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' && cs.display !== 'none';
  };
  const clean = (t) => (t || '').replace(/\s+/g, ' ').trim();
  const labelFor = (el) => {
    if (el.id) {
      const l = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (l) return clean(l.innerText);
    }
    const lever = el.closest('.application-question')?.querySelector('.application-label');
    if (lever) return clean(lever.innerText);
    const wrap = el.closest('label');
    if (wrap && clean(wrap.innerText)) return clean(wrap.innerText);
    const by = el.getAttribute('aria-labelledby');
    if (by) {
      const t = by.split(/\s+/).map((id) => document.getElementById(id)?.innerText || '').join(' ');
      if (clean(t)) return clean(t);
    }
    if (el.getAttribute('aria-label')) return clean(el.getAttribute('aria-label'));
    const fs = el.closest('fieldset');
    const legend = fs?.querySelector('legend');
    if (legend && (el.type === 'radio' || el.type === 'checkbox')) return clean(legend.innerText);
    let p = el.parentElement;
    for (let i = 0; i < 5 && p; i++, p = p.parentElement) {
      const others = [...p.querySelectorAll('input:not([type=hidden]), select, textarea')]
        .filter((n) => n !== el && !(el.name && n.name === el.name));
      if (others.length > 0) break;
      const lab = [...p.querySelectorAll('label, legend, [class*="label" i], [class*="question" i]')]
        .find((n) => !n.contains(el) && clean(n.innerText));
      if (lab) return clean(lab.innerText);
    }
    p = el.parentElement;
    for (let i = 0; i < 4 && p; i++, p = p.parentElement) {
      const t = clean(p.innerText);
      if (t && t.length < 300) return t;
    }
    return clean(el.placeholder || el.name || '');
  };
  const out = [];
  let idx = 0;
  const radioGroups = new Map();
  for (const el of document.querySelectorAll('input, textarea, select')) {
    const type = (el.getAttribute('type') || el.tagName).toLowerCase();
    if (['hidden', 'submit', 'button', 'image', 'reset', 'file', 'search'].includes(type)) continue;
    if (el.disabled || el.readOnly) continue;
    if (type === 'radio') {
      if (!visible(el) && !visible(el.parentElement || el)) continue;
      const g = el.name || labelFor(el);
      if (!radioGroups.has(g)) radioGroups.set(g, []);
      el.setAttribute('data-il', String(idx));
      const own = el.id ? document.querySelector(`label[for="${CSS.escape(el.id)}"]`) : el.closest('label');
      radioGroups.get(g).push({ idx: idx++, option: clean(own?.innerText || el.value), checked: el.checked,
        question: clean(el.closest('fieldset')?.querySelector('legend')?.innerText || '') ||
          clean(el.closest('[role=radiogroup], .field, .application-question, div')?.parentElement?.innerText || '').slice(0, 250) });
      continue;
    }
    if (!visible(el)) continue;
    el.setAttribute('data-il', String(idx));
    const combo = el.getAttribute('role') === 'combobox' || !!el.closest('.select__control, [class*="select__control"]');
    let chosen = combo ? clean(el.closest('.select__control, [class*="select__control"]')?.querySelector('[class*="single-value"], [class*="multi-value"]')?.innerText || '') : '';
    // react-select placeholders ("Select…") are not real answers
    if (/^select(\b|[.…\s]|$)/i.test(chosen) || /^choose(\b|[.…\s]|$)/i.test(chosen)) chosen = '';
    let selectVal = '';
    if (el.tagName === 'SELECT' && el.selectedIndex >= 0) {
      const t = clean(el.options[el.selectedIndex]?.text || '');
      if (t && !/^select(\b|[.…\s]|$)/i.test(t) && el.selectedIndex > 0) selectVal = t;
    }
    out.push({
      idx: idx++,
      kind: combo ? 'combobox' : el.tagName === 'SELECT' ? 'select' : el.tagName === 'TEXTAREA' ? 'textarea' : type,
      label: labelFor(el).slice(0, 300),
      value: combo ? chosen : el.tagName === 'SELECT' ? selectVal : (type === 'checkbox' ? (el.checked ? 'on' : '') : el.value),
      required: el.required || el.getAttribute('aria-required') === 'true' || /\*\s*$/.test(labelFor(el)),
      options: el.tagName === 'SELECT' ? [...el.options].filter((o) => o.value !== '' && !/^select(\b|[.…\s]|$)/i.test(clean(o.text))).map((o) => clean(o.text)) : [],
    });
  }
  const radios = [...radioGroups.values()].map((opts) => ({ question: opts[0].question, opts, answered: opts.some((o) => o.checked) }));
  return { fields: out, radios };
}
"""

REQUIRED_LEFT_JS = r"""
() => {
  const clean = (t) => (t || '').replace(/\s+/g, ' ').trim();
  const labelOf = (el) => {
    const lever = el.closest('.application-question')?.querySelector('.application-label');
    if (lever) return clean(lever.innerText).replace(/\s*✱$/, '').slice(0, 80);
    const l = el.id && document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
    return clean(l?.innerText || el.getAttribute('aria-label') || el.closest('fieldset')?.querySelector('legend')?.innerText || el.name).slice(0, 80);
  };
  const seen = new Set();
  const left = [];
  for (const el of document.querySelectorAll('input[required], textarea[required], select[required], [aria-required=true]')) {
    if (el.type === 'hidden' || el.type === 'file') continue;
    let empty;
    if (el.type === 'radio') {
      if (seen.has(el.name)) continue;
      seen.add(el.name);
      empty = !document.querySelector(`input[name="${CSS.escape(el.name)}"]:checked`);
    } else if (el.type === 'checkbox') {
      empty = !el.checked;
    } else {
      const ctl = el.closest('.select__control, [class*="select__control"]');
      if (ctl) {
        const picked = clean(ctl.querySelector('[class*="single-value"], [class*="multi-value"]')?.innerText || '');
        empty = !picked || /^select(\b|[.…\s]|$)/i.test(picked) || /^choose(\b|[.…\s]|$)/i.test(picked);
      } else if (el.tagName === 'SELECT') {
        const t = clean(el.options[el.selectedIndex]?.text || '');
        empty = el.selectedIndex <= 0 || !t || /^select(\b|[.…\s]|$)/i.test(t);
      } else {
        empty = !el.value;
      }
      if (el.getBoundingClientRect().width === 0 && !ctl) continue;
    }
    const lab = labelOf(el);
    if (empty && lab && !/^cards\[/.test(lab) && !left.includes(lab)) left.push(lab);
  }
  return left;
}
"""


async def pick_autocomplete(page: Page) -> None:
    """Location fields often only accept a value chosen from their suggestion list."""
    await asyncio.sleep(1.5)
    for sel in ["[role=option]", ".dropdown-location", ".pac-item", "[class*='dropdown-results'] div", "[class*='suggestion']"]:
        try:
            opt = page.locator(sel).first
            if await opt.is_visible(timeout=500):
                await opt.click(timeout=2000)
                return
        except Exception:
            continue


async def open_application_form(page: Page) -> None:
    """Moves from a job description page to the actual application form where the URL shape is known."""
    url = page.url
    if "jobs.lever.co" in url and not re.search(r"/apply/?$", url.split("?")[0]):
        await page.goto(url.split("?")[0].rstrip("/") + "/apply", wait_until="domcontentloaded")
    elif "jobs.ashbyhq.com" in url and "/application" not in url:
        await page.goto(url.split("?")[0].rstrip("/") + "/application", wait_until="domcontentloaded")
    elif not ATS_HOST.search(re.sub(r"^https?://", "", url).split("/")[0]):
        frame_src = await page.evaluate(
            """() => {
              for (const f of document.querySelectorAll('iframe')) {
                try {
                  const u = new URL(f.src);
                  if (/(^|\\.)(greenhouse\\.io|lever\\.co|ashbyhq\\.com|smartrecruiters\\.com)$/.test(u.hostname)
                      && !/captcha/.test(u.hostname)) return f.src;
                } catch {}
              }
              return '';
            }"""
        )
        if frame_src:
            await page.goto(frame_src, wait_until="domcontentloaded")
    try:
        await page.wait_for_selector("input:not([type=hidden]), textarea", state="visible", timeout=15000)
    except Exception:
        pass
    await asyncio.sleep(1)
    if await page.locator("form input, form textarea").count() == 0:
        for sel in ["a:has-text('Apply for this job')", "button:has-text('Apply for this job')",
                    "a:has-text('Apply now')", "button:has-text('Apply now')",
                    "a:has-text('Apply')", "button:has-text('Apply')"]:
            try:
                btn = page.locator(sel).first
                if await btn.is_visible(timeout=1200):
                    await btn.click()
                    await asyncio.sleep(3)
                    break
            except Exception:
                continue


async def blocked_reason(page: Page) -> str | None:
    try:
        text = await page.inner_text("body", timeout=5000)
    except Exception:
        text = ""
    # Invisible reCAPTCHA/hCaptcha frames are always present on Greenhouse/Lever; only a
    # visible challenge blocks the form.
    visible_challenge = await page.evaluate(
        """() => [...document.querySelectorAll('iframe')].some((f) => {
          if (!/hcaptcha|recaptcha|challenges\\.cloudflare|turnstile/.test(f.src) || /anchor|enclave/.test(f.src)) return false;
          const r = f.getBoundingClientRect();
          return r.width > 250 && r.height > 250 && getComputedStyle(f).visibility !== 'hidden';
        })"""
    )
    if CAPTCHA_TEXT.search(text) or visible_challenge:
        return "CAPTCHA or bot check"
    fillable = await page.locator("input:not([type=hidden]):visible, textarea:visible").count()
    if await page.locator("input[type=password]:visible").count() and fillable <= 3:
        return "Site requires signing in to an account"
    if fillable == 0:
        return "No application form found on the page"
    return None


async def fill_generic(page: Page, job: dict) -> tuple[bool, str]:
    await open_application_form(page)
    reason = await blocked_reason(page)
    if reason:
        return False, reason

    rp = resume_path()
    if rp:
        inputs = page.locator("input[type=file]")
        n = await inputs.count()
        target = None
        for i in range(n):
            meta = (await inputs.nth(i).evaluate(
                "el => (el.name + ' ' + el.id + ' ' + (el.closest('div,fieldset')?.innerText || '')).toLowerCase()"
            ))
            if "cover" in meta:
                continue
            target = inputs.nth(i)
            if "resume" in meta or "cv" in meta:
                break
        if target is not None:
            try:
                await target.set_input_files(str(rp))
                print(f"  ✓ Attached resume ({rp.name})")
                await asyncio.sleep(2.5)
            except Exception as e:
                print(f"  ! Resume upload failed: {e}")

    data = await page.evaluate(COLLECT_FIELDS_JS)
    filled = 0
    for f in data["fields"]:
        if f["value"]:
            continue
        label, kind = f["label"], f["kind"]
        loc = page.locator(f"[data-il='{f['idx']}']")
        if re.search(r"if different|preferred name", label, re.I) and not f["required"]:
            continue
        try:
            if kind in ("select", "combobox"):
                opts = f["options"]
                opt_sel = "[role=option]"
                if kind == "combobox":
                    iid = await loc.get_attribute("id")
                    if iid:
                        opt_sel = f'[id^="react-select-{iid}-option"], #{iid}-listbox [role=option]'
                    await loc.scroll_into_view_if_needed(timeout=3000)
                    await loc.focus(timeout=3000)
                    await page.keyboard.press("ArrowDown")
                    await asyncio.sleep(0.7)
                    opts = [t.strip() for t in await page.locator(opt_sel).all_inner_texts() if t.strip()]
                choice = None
                if EEO_LABEL.search(label):
                    choice = next((o for o in opts if DECLINE_OPTION.search(o)), None)
                if choice is None:
                    direct = answer_for(label)
                    if direct:
                        choice = pick_option(opts, direct) if opts else direct
                    if direct == "Online job board" and opts and not choice:
                        choice = next((o for o in opts if re.search(r"job board|online|internet|website|simplify|other", o, re.I)), None)
                if choice is None and re.search(r"acknowledge|privacy (policy|notice)|i have read|consent", label, re.I):
                    choice = next((o for o in opts if re.search(r"acknowledge|agree|accept|yes|i have read|consent", o, re.I)), None)
                if choice is None and f["required"] and opts:
                    choice = pick_option(opts, ai_answer(label, job, options=opts))
                if not choice:
                    if kind == "combobox":
                        await page.keyboard.press("Escape")
                    continue
                if kind == "select":
                    await loc.select_option(label=choice, timeout=3000)
                elif opts:
                    exact = re.compile(r"^\s*" + re.escape(choice) + r"\s*$")
                    await page.locator(opt_sel).filter(has_text=exact).first.click(timeout=3000)
                else:
                    # Async-search comboboxes (e.g. location) only show options after typing.
                    await loc.fill(choice, timeout=3000)
                    await asyncio.sleep(2)
                    typed = page.locator(opt_sel)
                    if await typed.count():
                        await typed.first.click(timeout=3000)
                    else:
                        await page.keyboard.press("Enter")
                filled += 1
                print(f"  ✓ {label[:60]} → {choice}")
            elif kind == "checkbox":
                if f["required"] and re.search(r"agree|acknowledge|certify|confirm|consent|privacy", label, re.I):
                    await loc.check(timeout=3000)
                    filled += 1
            else:
                value = answer_for(label) if kind != "textarea" else None
                if not value and re.search(r"cover letter", label, re.I):
                    value = PROFILE.get("cover_letter", "") or None
                if not value and (kind == "textarea" or f["required"]):
                    long = kind == "textarea" or len(label) > 60
                    value = ai_answer(label, job, long=long)
                    if value.strip().upper() == "N/A":
                        value = ""
                if value:
                    await loc.fill(value, timeout=3000)
                    if re.search(r"location|city", label, re.I):
                        await pick_autocomplete(page)
                    filled += 1
                    print(f"  ✓ {label[:60]} → {value[:50]}")
        except Exception as e:
            print(f"  ! Could not fill '{label[:50]}': {e}")

    for group in data["radios"]:
        if group["answered"]:
            continue
        q = group["question"]
        opts = [o["option"] for o in group["opts"]]
        want = None
        if EEO_LABEL.search(q):
            want = next((o for o in opts if DECLINE_OPTION.search(o)), None)
        if want is None:
            ans = question_answer(q) or agent.decide_yes_no(q)
            want = pick_option(opts, ans) if ans else None
        if want is None:
            want = pick_option(opts, ai_answer(q, job, options=opts))
        if want:
            idx = group["opts"][opts.index(want)]["idx"]
            try:
                await page.locator(f"[data-il='{idx}']").check(timeout=3000, force=True)
                filled += 1
                print(f"  ✓ {q[:60]} → {want}")
            except Exception as e:
                print(f"  ! Could not choose '{want}' for '{q[:40]}': {e}")

    left_labels = await page.evaluate(REQUIRED_LEFT_JS)
    left = len(left_labels)
    for lab in left_labels:
        print(f"  · left for you: {lab}")
    for sel in ["button[type=submit]", "input[type=submit]", "button:has-text('Submit')"]:
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=1000):
                await btn.scroll_into_view_if_needed()
                break
        except Exception:
            continue
    if filled == 0:
        return False, "Could not fill any fields on this form"
    detail = f"Filled {filled} fields"
    if left:
        detail += f" · {left} required field{'s' if left != 1 else ''} left for you: " + "; ".join(
            x for x in left_labels[:3] if x
        ) + ("…" if left > 3 else "")
    return True, detail


async def process(context, job: dict) -> tuple[str, str, Page | None]:
    url = (job.get("apply_url") or "").strip()
    if not url:
        return "needs_manual", "Listing has no apply link", None
    page = await context.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await asyncio.sleep(2)
        is_workday = "myworkdayjobs" in page.url or "workday" in page.url
        if is_workday:
            os.environ["WORKDAY_APPLY_URL"] = url
            ok = await agent.handle_workday_page(page)
            if ok:
                return "ready_for_review", "Workday application filled up to the final Submit", page
            return "needs_manual", "Workday flow could not reach the final step", page
        ok, detail = await fill_generic(page, job)
        return ("ready_for_review" if ok else "needs_manual"), detail, page
    except Exception as e:
        return "failed", f"{type(e).__name__}: {e}"[:300], page


async def main() -> None:
    jobs = json.loads(Path(os.environ["JOBS_PATH"]).read_text(encoding="utf-8"))
    print(f"Auto-apply queue: {len(jobs)} job(s). Review mode — nothing is submitted.")
    if not os.getenv("XAI_API_KEY"):
        print("  ! XAI_API_KEY not set — open-ended questions will be left blank for you.")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=os.environ.get("AGENT_HEADLESS") == "1", slow_mo=40, args=["--disable-blink-features=AutomationControlled"]
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            ),
        )
        ready = 0
        for i, job in enumerate(jobs, 1):
            print(f"\n[{i}/{len(jobs)}] {job.get('role')} — {job.get('company')}")
            emit(job["id"], "running")
            try:
                status, detail, page = await asyncio.wait_for(process(context, job), JOB_TIMEOUT_SECS)
            except asyncio.TimeoutError:
                status, detail, page = "needs_manual", f"Timed out after {JOB_TIMEOUT_SECS}s", None
            emit(job["id"], status, detail)
            print(f"  → {status}: {detail}")
            if status == "ready_for_review":
                ready += 1
            elif page is not None:
                await page.close()

        print(f"\n@@DONE {json.dumps({'ready': ready, 'total': len(jobs)})}", flush=True)
        if ready == 0 or os.environ.get("AGENT_HEADLESS") == "1":
            await browser.close()
            return
        print("Review each open tab and click Submit yourself. Close the browser window when finished.")
        closed = asyncio.Event()
        browser.on("disconnected", lambda _: closed.set())
        await closed.wait()


if __name__ == "__main__":
    asyncio.run(main())
