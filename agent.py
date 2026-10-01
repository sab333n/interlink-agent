import asyncio
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from playwright.async_api import Page, TimeoutError as PWTimeout, async_playwright

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")

# ── Grok client (uses OpenAI-compatible API) ──
XAI_MODEL = os.getenv("XAI_MODEL", "grok-3-mini")
_grok_client: OpenAI | None = None


def _grok() -> OpenAI | None:
    global _grok_client
    if _grok_client is None and os.getenv("XAI_API_KEY"):
        _grok_client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
    return _grok_client


def grok_complete(prompt: str, max_tokens: int = 200) -> str:
    """Returns "" when no XAI_API_KEY is configured or the call fails."""
    global XAI_MODEL
    client = _grok()
    if client is None:
        return ""
    try:
        resp = client.chat.completions.create(
            model=XAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        if "model" in str(e).lower():
            try:
                names = sorted(m.id for m in client.models.list().data if "grok" in m.id)
                fallback = next((n for n in names if "mini" in n or "fast" in n), names[-1] if names else "")
                if fallback and fallback != XAI_MODEL:
                    print(f"  ! Model {XAI_MODEL} unavailable — switching to {fallback}")
                    XAI_MODEL = fallback
                    return grok_complete(prompt, max_tokens)
            except Exception:
                pass
        print(f"  ! Grok call failed ({XAI_MODEL}): {e}")
        return ""

# ── Load profile ──
# The server passes the signed-in user's profile via PROFILE_PATH; profile.json
# is only a fallback for running this script by hand.
_PROFILE_FILE = Path(os.environ.get("PROFILE_PATH") or (_ROOT / "profile.json"))
_PROFILE_DEFAULTS = {
    "first_name": "", "last_name": "", "email": "", "phone": "", "phone_local": "",
    "phone_country_code": "+1", "phone_device_type": "Personal Mobile", "address": "",
    "city": "", "state": "", "state_abbr": "", "zip": "", "country": "United States of America",
    "linkedin": "", "github": "", "portfolio": "", "work_auth": "", "sponsorship": "",
    "relocation": "", "years_experience": "", "current_title": "", "university": "",
    "degree": "", "degree_full": "", "major": "", "secondary_major": "", "graduation_year": "",
    "gpa": "", "salary_expectation": "", "start_date": "", "availability_choice": "",
    "referral": "", "cover_letter": "", "skills_summary": "", "skills_tags": [],
    "work_experience": [], "education": [], "prior_employment_at_target": "No",
    "projects": [], "leadership": [], "honors": [], "availability": "",
    "remote_preference": "", "ats_account_password": "",
}
_PROFILE_RAW = json.loads(_PROFILE_FILE.read_text(encoding="utf-8")) if _PROFILE_FILE.exists() else {}
PROFILE = {**_PROFILE_DEFAULTS, **{k: v for k, v in _PROFILE_RAW.items() if v is not None}}

_RESUME_ENV = os.environ.get("RESUME_PATH", "").strip()
RESUME_CANDIDATES = (
    [Path(_RESUME_ENV)]
    if _RESUME_ENV
    else [_ROOT / "resume.pdf", _ROOT / "resume.txt", _ROOT.parent / "resume.pdf"]
)

INTERLINK_URL = os.environ.get("INTERLINK_URL", "http://localhost:5173/applicant/jobs")

# ── Submission safety ──
# By default the agent is in DRY RUN mode — it fills every field and clicks
# through every Save/Continue button, but STOPS at the final "Submit" /
# "Submit Application" button so you can test as many times as you want.
#
# To actually submit, set either:
#   AGENT_SUBMIT=1 python agent.py
#   AGENT_DRY_RUN=0 python agent.py
AGENT_SUBMIT = os.environ.get("AGENT_SUBMIT", "0") == "1"
AGENT_DRY_RUN = os.environ.get("AGENT_DRY_RUN", "1") == "1" and not AGENT_SUBMIT

# Force account flow on Workday. "auto" (default) = detect from page.
# "create" = always try to create a new account (click Create Account toggle if needed).
# "signin" = always try to sign in with existing credentials.
WORKDAY_MODE = os.environ.get("WORKDAY_MODE", "auto").lower()
WORKDAY_STEP_RETRY_BUDGET = 2

WORKDAY_PRIORITY_FIELD_SPECS = [
    {
        "key": "availability",
        "kind": "checkbox",
        "question_keywords": [
            "notice period",
            "available to start work",
            "when would you be available to start work",
            "when can you start",
            "start work",
        ],
        "answer": str(PROFILE.get("availability_choice", "Available Immediately")),
    },
    {
        "key": "salary",
        "kind": "text",
        "question_keywords": [
            "salary expectations",
            "salary expectation",
            "salary",
            "compensation",
            "pay expectation",
            "gross",
        ],
        "answer": str(PROFILE.get("salary_expectation", "")),
    },
    {
        "key": "relocation",
        "kind": "select_one",
        "question_keywords": [
            "relocate",
            "relocation",
            "other cities",
            "work in other cities",
        ],
        "answer": str(PROFILE.get("relocation", "Yes")),
    },
    {
        "key": "work_auth",
        "kind": "select_one",
        "question_keywords": [
            "legally authorized",
            "authorized to work",
            "work authorization",
            "eligible to work",
            "right to work",
        ],
        "answer": str(PROFILE.get("work_auth", "Yes")),
    },
    {
        "key": "sponsorship",
        "kind": "select_one",
        "question_keywords": [
            "require visa sponsorship",
            "visa sponsorship",
            "require sponsorship",
            "need sponsorship",
            "future require visa sponsorship",
        ],
        "answer": str(PROFILE.get("sponsorship", "No")),
    },
]


def ask_grok(question: str, context: str = "") -> str:
    """Ask Grok a question about how to fill a form field."""
    prompt = f"""You are helping autofill a job application form.

USER PROFILE:
{json.dumps({k: v for k, v in PROFILE.items() if k != "ats_account_password"}, indent=2)}

{f"PAGE CONTEXT: {context}" if context else ""}

QUESTION: {question}

Answer with ONLY the value to type/select. No explanation. No quotes.
If unsure, use the most appropriate value from the profile.
For yes/no questions, answer Yes or No.
For dropdowns, give the exact option text to select.
"""
    return grok_complete(prompt)


async def ask_grok_async(question: str, context: str = "") -> str:
    """Non-blocking Grok call from async code."""
    return await asyncio.to_thread(ask_grok, question, context)


async def fill_field(page: Page, selector: str, value: str, log: str = ""):
    """Try to fill a field. Returns True if successful."""
    try:
        el = page.locator(selector).first
        if await el.is_visible(timeout=2000):
            await el.click()
            await asyncio.sleep(0.2)
            await el.clear()
            await el.fill(value)
            print(f"  ✓ {log or selector}: {value[:50]}")
            return True
    except Exception:
        pass
    return False


async def select_option(page: Page, selector: str, value: str, log: str = ""):
    """Try to select a dropdown option."""
    try:
        el = page.locator(selector).first
        if await el.is_visible(timeout=2000):
            try:
                await el.select_option(label=value, timeout=2000)
            except Exception:
                try:
                    await el.select_option(value=value, timeout=2000)
                except Exception:
                    # Click and try to find matching option
                    await el.click()
                    await asyncio.sleep(0.5)
                    option = page.locator(f"option:has-text('{value}')").first
                    if await option.count() > 0:
                        await option.click()
            print(f"  ✓ {log}: {value}")
            return True
    except Exception:
        pass
    return False


async def pick_workday_dropdown(page: Page, button, desired_value: str) -> bool:
    """Click a Workday custom dropdown button and pick an option containing desired_value."""
    try:
        await button.scroll_into_view_if_needed(timeout=2000)
        await button.click(timeout=3500)
        await asyncio.sleep(0.5)

        # Workday renders options as <div role="option"> or <li role="option">
        # inside a popup. Search by partial text (case-insensitive).
        wanted = desired_value.lower()
        options = page.locator("[role='option']:visible, [data-automation-id='promptOption']:visible")
        n = await options.count()
        for j in range(n):
            opt = options.nth(j)
            try:
                text = (await opt.text_content() or "").strip()
            except Exception:
                continue
            if not text:
                continue
            if wanted in text.lower():
                try:
                    await opt.click(timeout=2500)
                    return True
                except Exception:
                    continue
        # No match — close dropdown
        try:
            await page.keyboard.press("Escape")
        except Exception:
            pass
    except Exception:
        pass
    return False


async def handle_select_one_questions(page: Page):
    """
    Specialised handler for Workday's "Application Questions" page.

    Every unanswered dropdown on that page visibly displays the literal text
    "Select One". We use JavaScript to find every such button in the DOM,
    climb up to the nearest question text (any ancestor element whose text
    ends with '?' or has a Workday `questionText` attribute), and return
    both as a pair of opaque ids. We then iterate those pairs, click to open
    the menu, read the options, and pick the profile-appropriate answer.
    """
    try:
        pairs = await page.evaluate(
            """() => {
                const out = [];
                const all = document.querySelectorAll('button, div[role="combobox"], div[role="button"]');
                let idx = 0;
                for (const el of all) {
                    const t = (el.innerText || '').trim();
                    if (t !== 'Select One') continue;
                    // Climb up to find the question label.
                    let question = '';
                    let p = el.parentElement;
                    for (let i = 0; i < 14 && p; i++) {
                        const q = p.querySelector(
                            "[data-automation-id='questionText'], " +
                            "[data-automation-id='formLabel'], " +
                            "[data-automation-id*='promptLabel'], " +
                            "label:not([for=''])"
                        );
                        if (q && q.innerText) {
                            const tx = q.innerText.trim();
                            if (tx) { question = tx; break; }
                        }
                        const kids = Array.from(p.children || []);
                        for (const k of kids) {
                            const tx = (k.innerText || '').trim();
                            if (tx && tx.length < 250 && /\\?\\*?\\s*$/.test(tx)) {
                                question = tx; break;
                            }
                        }
                        if (question) break;
                        p = p.parentElement;
                    }
                    if (!question) continue;
                    // Tag the element so we can find it again from Python.
                    const tag = `__il_dd_${idx++}`;
                    el.setAttribute('data-il-agent-id', tag);
                    out.push({ id: tag, question });
                }
                return out;
            }"""
        )
    except Exception as e:
        print(f"  [select-one] probe failed: {e}")
        return

    if not pairs:
        return

    print(f"  [select-one] {len(pairs)} 'Select One' dropdown(s) found")

    for p in pairs:
        tag = p["id"]
        question = (p["question"] or "").strip()
        qlower = question.lower()
        qnorm = _norm_text(question)
        if not question:
            continue

        # Decide the answer locally (fast path). These three cover the bulk
        # of Workday application questions.
        if "sponsor" in qlower:
            desired = str(PROFILE.get("sponsorship", "Yes"))
        elif (
            "authori" in qlower
            or "eligible to work" in qlower
            or "right to work" in qlower
        ):
            desired = str(PROFILE.get("work_auth", "Yes"))
        elif (
            "relocate" in qlower
            or "willing to move" in qlower
            or "willing to relocate" in qlower
            or "other cities" in qnorm
        ):
            desired = str(PROFILE.get("relocation", "Yes"))
        elif (
            "previously" in qlower
            and ("worked at" in qlower or "employee" in qlower or "intern" in qlower)
        ):
            desired = str(PROFILE.get("prior_employment_at_target", "No"))
        else:
            desired = ""  # let Grok decide below

        btn = page.locator(f"[data-il-agent-id='{tag}']")
        if await btn.count() == 0:
            continue

        # Open the menu.
        try:
            await btn.first.scroll_into_view_if_needed(timeout=2000)
            await btn.first.click(timeout=2500)
        except Exception as e:
            print(f"  [select-one] could not open '{question[:70]}': {e}")
            continue

        await asyncio.sleep(0.35)

        # Read available option labels.
        try:
            options_loc = page.locator(
                "[role='option']:visible, "
                "[data-automation-id='promptOption']:visible, "
                "li[role='option']:visible"
            )
            n_opts = await options_loc.count()
            opts = []
            for k in range(n_opts):
                try:
                    t = (await options_loc.nth(k).text_content() or "").strip()
                    if t:
                        opts.append(t)
                except Exception:
                    continue
        except Exception:
            opts = []

        if not opts:
            print(f"  [select-one] no options visible for '{question[:70]}'")
            try:
                await page.keyboard.press("Escape")
            except Exception:
                pass
            continue

        # If we didn't have a local answer, ask Grok with the option list.
        if not desired:
            try:
                prompt = (
                    f"Question: {question}\nOptions: {opts}\n\n"
                    "Pick the SINGLE best option that an otherwise-qualified "
                    "candidate would choose for a legitimate job application. "
                    "Respond with EXACTLY the option text, nothing else."
                )
                guess = await ask_grok_async(prompt)
                desired = (guess or "").strip().strip('"').strip("'")
            except Exception:
                desired = ""

        # Match desired against available options (exact, then substring).
        chosen = ""
        if desired:
            wanted = desired.lower()
            for o in opts:
                if o.lower() == wanted:
                    chosen = o
                    break
            if not chosen:
                for o in opts:
                    if wanted in o.lower() or o.lower() in wanted:
                        chosen = o
                        break

        if not chosen:
            print(
                f"  [select-one] no matching option for '{question[:60]}' "
                f"(wanted '{desired}', options {opts[:6]})"
            )
            try:
                await page.keyboard.press("Escape")
            except Exception:
                pass
            continue

        # Click the matching option.
        try:
            target = page.locator(
                f"[role='option']:has-text('{chosen}'), "
                f"[data-automation-id='promptOption']:has-text('{chosen}'), "
                f"li[role='option']:has-text('{chosen}')"
            ).first
            await target.click(timeout=2500)
            print(f"  ✓ {question[:60]}: {chosen}")
        except Exception as e:
            print(f"  [select-one] click failed for '{chosen}': {e}")
            try:
                await page.keyboard.press("Escape")
            except Exception:
                pass

        await asyncio.sleep(0.2)


async def handle_workday_dropdowns(page: Page):
    """Fill the Workday combobox-style dropdowns (buttons with aria-haspopup='listbox')."""
    # These are the most common ones on Workday applications.
    # Map from a keyword that appears in the button label → profile value to pick.
    dropdown_map = [
        # Country / phone
        (["country phone code", "phone code"], PROFILE.get("phone_country_code", "+1")),
        (
            ["phone device type", "device type", "phone type"],
            PROFILE.get("phone_device_type", "Personal Mobile"),
        ),
        (["country", "country/region"], PROFILE.get("country", "United States of America")),

        # Referral / source
        (
            ["how did you hear about us", "how did you hear", "source", "referral source", "referred by"],
            PROFILE.get("referral", "Social Media"),
        ),

        # Location
        (["state", "province", "region"], PROFILE.get("state", "Louisiana")),

        # Yes/No application question dropdowns (Workday sometimes uses Select One
        # dropdowns for these instead of radios)
        (
            ["are you legally authorized", "legally authorized to work", "work authorization"],
            PROFILE.get("work_auth", "Yes"),
        ),
        (
            ["require sponsorship", "need sponsorship", "visa sponsorship"],
            PROFILE.get("sponsorship", "No"),
        ),
        (
            ["are you open to relocate", "willing to relocate", "relocation", "other cities"],
            PROFILE.get("relocation", "Yes"),
        ),
        (
            ["have you worked at", "previously employed", "former employee", "worked here previously"],
            PROFILE.get("prior_employment_at_target", "No"),
        ),

        # Education
        (["degree", "level of education", "education level"], PROFILE.get("degree", "Bachelors")),

        # Notice / availability
        (
            ["notice period", "when can you start", "availability", "start date", "available to start work"],
            PROFILE.get("availability_choice", "Available Immediately"),
        ),
    ]

    # Workday dropdowns are usually <button aria-haspopup="listbox"> with
    # an inner <span> holding the current value or placeholder. The application-
    # questions page also uses role="combobox" or data-automation-id="promptOption".
    buttons = page.locator(
        "button[aria-haspopup='listbox']:visible, "
        "button[aria-haspopup='menu']:visible, "
        "button[aria-haspopup='true']:visible, "
        "button[role='combobox']:visible, "
        "button[data-automation-id*='promptOption']:visible, "
        "button[data-automation-id*='selectWidget']:visible"
    )
    count = await buttons.count()
    if count:
        print(f"  [dropdowns] found {count} candidate button(s)")
    for i in range(count):
        btn = buttons.nth(i)
        try:
            if not await btn.is_visible(timeout=800):
                continue

            # Get the current value — if something is already selected, skip
            btn_text = (await btn.text_content() or "").strip()

            # Find the label. Workday application-questions pages put the
            # question in a sibling/ancestor, NOT in aria-labelledby, so we
            # use a multi-strategy JS walker.
            label = await btn.evaluate(
                """(el) => {
                    // 1. aria-labelledby
                    const lby = el.getAttribute('aria-labelledby');
                    if (lby) {
                        const parts = lby.split(/\\s+/).map(id => {
                            const n = document.getElementById(id);
                            return n ? n.innerText.trim() : '';
                        }).filter(Boolean);
                        if (parts.length) return parts.join(' ');
                    }
                    // 2. aria-label
                    const al = el.getAttribute('aria-label');
                    if (al && !/^(select one|select|choose)$/i.test(al.trim())) {
                        return al.trim();
                    }
                    // 3. Walk up ancestors, searching for labels / question text.
                    //    Workday uses [data-automation-id='questionText'],
                    //    [data-automation-id='formLabel'], or <label>.
                    let p = el.parentElement;
                    for (let i = 0; i < 12 && p; i++) {
                        const q = p.querySelector(
                            "[data-automation-id='questionText'], " +
                            "[data-automation-id='formLabel'], " +
                            "[data-automation-id*='promptLabel'], " +
                            "label:not([for='']), " +
                            "legend"
                        );
                        if (q) {
                            const t = q.innerText && q.innerText.trim();
                            if (t) return t;
                        }
                        // Any heading or div whose text ends with '?' is probably a question.
                        const kids = Array.from(p.children || []);
                        for (const k of kids) {
                            const t = (k.innerText || '').trim();
                            if (t && t.length < 250 && /\\?\\*?\\s*$/.test(t)) {
                                return t;
                            }
                        }
                        p = p.parentElement;
                    }
                    return '';
                }"""
            )
            label = (label or "").strip()

            label_lower = label.lower()
            btn_text_lower = btn_text.lower()

            # Skip if the button clearly has a non-placeholder value selected
            placeholder_like = (
                not btn_text
                or btn_text_lower in ("select one", "select", "choose", "--", "")
                or btn_text_lower == label_lower
            )
            if not placeholder_like:
                continue

            if not label:
                # No resolvable question → can't match against dropdown_map.
                print(
                    f"  [dropdowns] skip #{i}: no label found "
                    f"(btn_text='{btn_text[:40]}')"
                )
                continue

            matched = False
            for keywords, desired in dropdown_map:
                if any(k in label_lower for k in keywords):
                    matched = True
                    print(f"  ▸ Dropdown '{label.strip()[:60]}' → trying '{desired}'")
                    ok = await pick_workday_dropdown(page, btn, desired)
                    if ok:
                        print(f"  ✓ {label.strip()[:60]}: {desired}")
                    else:
                        print(f"  ! Could not pick '{desired}' in '{label.strip()[:60]}'")
                    await asyncio.sleep(0.15)
                    break

            if not matched:
                # Ask Grok as a fallback — for application questions like
                # "Are you open to relocate?" where the keyword map missed.
                print(
                    f"  [dropdowns] no keyword match for '{label.strip()[:80]}' "
                    "— asking Grok"
                )
                try:
                    guess = await _dropdown_grok_fallback(page, btn, label)
                    if guess:
                        ok = await pick_workday_dropdown(page, btn, guess)
                        if ok:
                            print(
                                f"  ✓ {label.strip()[:60]}: {guess} (Grok)"
                            )
                        else:
                            print(
                                f"  ! Grok said '{guess}' but no matching option in "
                                f"'{label.strip()[:60]}'"
                            )
                except Exception:
                    pass
        except Exception as e:
            print(f"  [dropdowns] exception: {e}")
            continue


async def _dropdown_grok_fallback(page: Page, btn, label: str) -> str:
    """Open the dropdown, read the available options, ask Grok which one
    best answers the given question based on PROFILE. Returns the selected
    label text (or '' if we should give up and close the menu)."""
    # Open the menu briefly to enumerate options
    try:
        await btn.click(timeout=2500)
        await asyncio.sleep(0.4)
    except Exception:
        return ""

    try:
        options = await page.locator(
            "[role='option']:visible, "
            "div[data-automation-id='promptOption']:visible, "
            "li[role='option']:visible"
        ).all_text_contents()
    except Exception:
        options = []

    # Close the menu
    try:
        await page.keyboard.press("Escape")
    except Exception:
        pass

    opts = [o.strip() for o in options if o and o.strip()]
    opts = list(dict.fromkeys(opts))  # de-dup, preserve order
    if not opts:
        return ""

    ll = label.lower()
    # Fast local answers for the common yes/no patterns
    if "sponsor" in ll:
        want = PROFILE.get("sponsorship", "Yes")
    elif "authori" in ll or "legally allowed" in ll:
        want = PROFILE.get("work_auth", "Yes")
    elif "relocate" in ll or "willing to move" in ll:
        want = PROFILE.get("relocation", "Yes")
    else:
        want = None

    if want:
        for o in opts:
            if o.lower() == want.lower():
                return o
        # fallback: first option containing the word
        for o in opts:
            if want.lower() in o.lower():
                return o

    # Otherwise delegate to Grok
    try:
        prompt_text = (
            f"Question: {label}\nOptions: {opts}\n\n"
            "Pick the SINGLE best option that a typical qualified candidate "
            f"(profile summary: {PROFILE.get('skills_summary', '')}) would "
            "choose. Respond with EXACTLY the option text, nothing else."
        )
        answer = await ask_grok_async(prompt_text)
        answer = (answer or "").strip().strip('"').strip("'")
        if answer:
            for o in opts:
                if o.lower() == answer.lower():
                    return o
            for o in opts:
                if answer.lower() in o.lower() or o.lower() in answer.lower():
                    return o
    except Exception:
        pass

    return ""


def decide_yes_no(question: str) -> str | None:
    """Decide Yes/No for common application questions based on PROFILE.

    Returns "Yes", "No", or None if we shouldn't auto-answer.
    """
    q = question.lower()

    # Prior-employment questions → No (you're applying fresh)
    if any(
        k in q
        for k in [
            "worked at",
            "previously employed",
            "former employee",
            "previous employee",
            "currently employed",
            "current employee",
            "former intern",
            "previously as an employee",
        ]
    ):
        return "No"

    # Work authorization → Yes (or whatever profile says)
    if any(
        k in q
        for k in [
            "authorized to work",
            "work authorization",
            "legally authorized",
            "eligible to work",
            "right to work",
        ]
    ):
        return "Yes" if str(PROFILE.get("work_auth", "Yes")).lower().startswith("y") else "No"

    # Sponsorship → No (or profile)
    if any(
        k in q
        for k in [
            "require sponsorship",
            "need sponsorship",
            "visa sponsorship",
            "will you now or in the future require",
        ]
    ):
        return "Yes" if str(PROFILE.get("sponsorship", "No")).lower().startswith("y") else "No"

    # Relocation
    if "relocate" in q or "willing to relocate" in q:
        return "Yes" if str(PROFILE.get("relocation", "Yes")).lower().startswith("y") else "No"

    # 18 years or older
    if "18 years" in q or "at least 18" in q or "of legal age" in q:
        return "Yes"

    # Criminal record / felony — default No
    if any(k in q for k in ["convicted", "felony", "criminal record"]):
        return "No"

    # Non-compete / NDA with current employer
    if "non-compete" in q or "non compete" in q or "nda" in q:
        return "No"

    return None


def _norm_text(value: str) -> str:
    """Normalize visible labels/question text for loose matching."""
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _matches_question_keywords(question: str, keywords: list[str]) -> bool:
    qnorm = _norm_text(question)
    return any(_norm_text(keyword) in qnorm for keyword in keywords)


def _is_select_placeholder(value: str, question: str = "") -> bool:
    vnorm = _norm_text(value)
    if vnorm in ("", "select one", "select", "choose"):
        return True
    return bool(question and vnorm == _norm_text(question))


async def snapshot_workday_priority_fields(page: Page) -> dict[str, dict]:
    """Capture whether the priority Workday fields are present and answered."""
    snapshot = {
        spec["key"]: {
            "key": spec["key"],
            "kind": spec["kind"],
            "answer": spec["answer"],
            "question": "",
            "value": "",
            "present": False,
            "answered": False,
        }
        for spec in WORKDAY_PRIORITY_FIELD_SPECS
    }

    try:
        raw = await page.evaluate(
            """() => {
                const isVisible = (el) => !!(
                    el &&
                    (el.offsetWidth || el.offsetHeight || el.getClientRects().length)
                );
                const text = (el) => ((el && el.innerText) || '').trim();
                const findQuestion = (el) => {
                    let p = el.parentElement;
                    for (let i = 0; i < 14 && p; i++) {
                        const q = p.querySelector(
                            "[data-automation-id='questionText'], " +
                            "[data-automation-id='formLabel'], " +
                            "[data-automation-id*='promptLabel'], " +
                            "label:not([for='']), legend"
                        );
                        if (q) {
                            const t = text(q);
                            if (t) return t;
                        }
                        const kids = Array.from(p.children || []);
                        for (const k of kids) {
                            const t = text(k);
                            if (t && t.length < 250 && /\\?\\*?\\s*$/.test(t)) {
                                return t;
                            }
                        }
                        let sib = p.previousElementSibling;
                        for (let j = 0; j < 4 && sib; j++) {
                            const q2 = sib.querySelector?.(
                                "[data-automation-id='questionText'], " +
                                "[data-automation-id='formLabel'], " +
                                "[data-automation-id*='promptLabel'], " +
                                "label:not([for='']), legend"
                            );
                            const t2 = q2 ? text(q2) : text(sib);
                            if (t2 && t2.length < 250 && /[\\?\\*]\\s*$/.test(t2)) {
                                return t2;
                            }
                            sib = sib.previousElementSibling;
                        }
                        p = p.parentElement;
                    }
                    return '';
                };
                const findCheckboxLabel = (el) => {
                    const parentLabel = el.closest('label');
                    if (parentLabel) {
                        const t = text(parentLabel);
                        if (t) return t;
                    }
                    let sib = el.nextElementSibling;
                    while (sib) {
                        const t = text(sib);
                        if (t) return t;
                        sib = sib.nextElementSibling;
                    }
                    let p = el.parentElement;
                    for (let i = 0; i < 4 && p; i++) {
                        const t = text(p);
                        if (t && t.length < 120) return t;
                        p = p.parentElement;
                    }
                    return '';
                };

                const dropdowns = [];
                document.querySelectorAll("button, div[role='combobox'], div[role='button']").forEach((el) => {
                    if (!isVisible(el)) return;
                    const question = findQuestion(el);
                    if (!question) return;
                    dropdowns.push({ question, value: text(el) });
                });

                const textInputs = [];
                document.querySelectorAll(
                    "input[type='text'], input:not([type]), textarea"
                ).forEach((el) => {
                    if (!isVisible(el) || el.disabled || el.readOnly) return;
                    const question = findQuestion(el);
                    if (!question) return;
                    textInputs.push({ question, value: (el.value || '').trim() });
                });

                const checkboxes = [];
                document.querySelectorAll("input[type='checkbox']").forEach((el) => {
                    if (!isVisible(el)) return;
                    const question = findQuestion(el);
                    if (!question) return;
                    checkboxes.push({
                        question,
                        label: findCheckboxLabel(el),
                        checked: !!el.checked,
                    });
                });

                return { dropdowns, textInputs, checkboxes };
            }"""
        )
    except Exception:
        return snapshot

    for spec in WORKDAY_PRIORITY_FIELD_SPECS:
        key = spec["key"]
        if spec["kind"] == "select_one":
            for item in raw.get("dropdowns", []):
                if _matches_question_keywords(item.get("question", ""), spec["question_keywords"]):
                    snapshot[key]["present"] = True
                    snapshot[key]["question"] = item.get("question", "")
                    snapshot[key]["value"] = item.get("value", "")
                    if not _is_select_placeholder(
                        item.get("value", ""),
                        item.get("question", ""),
                    ):
                        snapshot[key]["answered"] = True
                    break
        elif spec["kind"] == "text":
            for item in raw.get("textInputs", []):
                if _matches_question_keywords(item.get("question", ""), spec["question_keywords"]):
                    snapshot[key]["present"] = True
                    snapshot[key]["question"] = item.get("question", "")
                    snapshot[key]["value"] = item.get("value", "")
                    snapshot[key]["answered"] = bool((item.get("value") or "").strip())
                    break
        elif spec["kind"] == "checkbox":
            desired_norm = _norm_text(spec["answer"])
            for item in raw.get("checkboxes", []):
                if _matches_question_keywords(item.get("question", ""), spec["question_keywords"]):
                    snapshot[key]["present"] = True
                    snapshot[key]["question"] = item.get("question", "")
                    if _norm_text(item.get("label", "")) == desired_norm:
                        snapshot[key]["value"] = item.get("label", "")
                        snapshot[key]["answered"] = bool(item.get("checked"))
                        break
            if not snapshot[key]["present"]:
                for item in raw.get("checkboxes", []):
                    if _norm_text(item.get("label", "")) == desired_norm:
                        snapshot[key]["present"] = True
                        snapshot[key]["question"] = item.get("question", "")
                        snapshot[key]["value"] = item.get("label", "")
                        snapshot[key]["answered"] = bool(item.get("checked"))
                        break

    return snapshot


def log_workday_priority_snapshot(snapshot: dict[str, dict], prefix: str = "  [priority]") -> None:
    for item in snapshot.values():
        if not item.get("present"):
            continue
        state = "filled" if item.get("answered") else "missing"
        current = (item.get("value") or "").strip() or "<blank>"
        print(f"{prefix} {item['key']}: {state} (current: {current[:80]})")


async def collect_workday_validation_errors(page: Page) -> list[str]:
    errors: list[str] = []
    try:
        err_loc = page.locator(
            "[data-automation-id='errorMessage']:visible, "
            ".error:visible, "
            "[aria-invalid='true']:visible"
        )
        err_count = await err_loc.count()
        for i in range(min(err_count, 8)):
            try:
                t = (await err_loc.nth(i).text_content() or "").strip()
                if t and t not in errors:
                    errors.append(t)
            except Exception:
                continue
    except Exception:
        pass
    return errors


async def get_workday_step_signature(page: Page) -> str:
    try:
        parts = await page.evaluate(
            """() => {
                const isVisible = (el) => !!(
                    el &&
                    (el.offsetWidth || el.offsetHeight || el.getClientRects().length)
                );
                const seen = new Set();
                const out = [];
                const nodes = document.querySelectorAll(
                    "h1, h2, " +
                    "[data-automation-id='pageTitle'], " +
                    "[data-automation-id='questionText'], " +
                    "[data-automation-id='formLabel'], " +
                    "[data-automation-id*='promptLabel'], legend"
                );
                for (const el of nodes) {
                    if (!isVisible(el)) continue;
                    const t = (el.innerText || '').trim();
                    if (!t || t.length > 250 || seen.has(t)) continue;
                    seen.add(t);
                    out.push(t);
                    if (out.length >= 10) break;
                }
                return out;
            }"""
        )
    except Exception:
        parts = []

    normalized = [_norm_text(part) for part in parts if part]
    if normalized:
        return " || ".join(normalized)
    return _norm_text(page.url)


async def is_workday_final_confirmation(page: Page, page_text: str = "") -> bool:
    body = _norm_text(page_text or await page.inner_text("body"))
    url = page.url.lower()

    confirmation_phrases = [
        "thank you for applying",
        "thanks for applying",
        "your application has been submitted",
        "application has been submitted",
        "your application was submitted",
        "we have received your application",
        "application received",
        "received your application",
        "application submitted",
    ]
    if any(phrase in body for phrase in confirmation_phrases):
        return True

    if "/apply/" not in url and any(
        marker in url
        for marker in ["/submitted", "/submission", "/confirmation", "/thank-you"]
    ):
        return True

    return False


async def fill_workday_salary_expectation(page: Page) -> bool:
    desired = str(PROFILE.get("salary_expectation", "")).strip()
    if not desired:
        return False

    fields = page.locator(
        "input[type='text']:visible, input:not([type]):visible, textarea:visible"
    )
    count = await fields.count()
    for i in range(count):
        field = fields.nth(i)
        try:
            question = await field.evaluate(
                """(el) => {
                    let p = el.parentElement;
                    for (let i = 0; i < 12 && p; i++) {
                        const q = p.querySelector(
                            "[data-automation-id='questionText'], " +
                            "[data-automation-id='formLabel'], " +
                            "[data-automation-id*='promptLabel'], " +
                            "label:not([for='']), legend"
                        );
                        if (q && q.innerText) {
                            const t = q.innerText.trim();
                            if (t) return t;
                        }
                        const kids = Array.from(p.children || []);
                        for (const k of kids) {
                            const t = (k.innerText || '').trim();
                            if (t && t.length < 250 && /\\?\\*?\\s*$/.test(t)) {
                                return t;
                            }
                        }
                        p = p.parentElement;
                    }
                    return '';
                }"""
            )
            if not _matches_question_keywords(
                question or "",
                next(spec["question_keywords"] for spec in WORKDAY_PRIORITY_FIELD_SPECS if spec["key"] == "salary"),
            ):
                continue
            await field.click(timeout=2500)
            await field.fill("", timeout=1500)
            await field.fill(desired, timeout=2500)
            print(f"  ↻ Retried salary expectation: {desired}")
            return True
        except Exception:
            continue
    return False


async def fill_workday_availability_choice(page: Page) -> bool:
    desired = str(PROFILE.get("availability_choice", "")).strip()
    if not desired:
        return False

    checkboxes = page.locator("input[type='checkbox']:visible")
    count = await checkboxes.count()
    desired_norm = _norm_text(desired)
    for i in range(count):
        cb = checkboxes.nth(i)
        try:
            label = await cb.evaluate(
                """(el) => {
                    const parentLabel = el.closest('label');
                    if (parentLabel) {
                        const t = (parentLabel.innerText || '').trim();
                        if (t) return t;
                    }
                    let sib = el.nextElementSibling;
                    while (sib) {
                        const t = (sib.innerText || '').trim();
                        if (t) return t;
                        sib = sib.nextElementSibling;
                    }
                    let p = el.parentElement;
                    for (let i = 0; i < 4 && p; i++) {
                        const t = (p.innerText || '').trim();
                        if (t && t.length < 120) return t;
                        p = p.parentElement;
                    }
                    return '';
                }"""
            )
            label_norm = _norm_text(label or "")
            if label_norm != desired_norm:
                continue
            if not await cb.is_checked():
                try:
                    await cb.check(force=True, timeout=2500)
                except Exception:
                    await cb.evaluate("n => n.click()")
            print(f"  ↻ Retried availability choice: {desired}")
            return True
        except Exception:
            continue
    return False


async def refill_unresolved_workday_priority_fields(
    page: Page, unresolved: list[dict]
) -> bool:
    unresolved_keys = {item["key"] for item in unresolved}
    refilled = False

    if "salary" in unresolved_keys:
        refilled = await fill_workday_salary_expectation(page) or refilled

    if "availability" in unresolved_keys:
        refilled = await fill_workday_availability_choice(page) or refilled

    if unresolved_keys & {"relocation", "work_auth", "sponsorship"}:
        before = await snapshot_workday_priority_fields(page)
        await handle_select_one_questions(page)
        await handle_workday_dropdowns(page)
        after = await snapshot_workday_priority_fields(page)
        for key in ("relocation", "work_auth", "sponsorship"):
            if (
                key in unresolved_keys
                and after.get(key, {}).get("answered")
                and not before.get(key, {}).get("answered")
            ):
                print(f"  ↻ Retried dropdown '{key}': {after[key]['answer']}")
                refilled = True

    return refilled


async def click_workday_button(page: Page, selector: str, label_for_log: str) -> bool:
    try:
        btn = page.locator(selector).last
        if await btn.count() == 0 or not await btn.is_visible(timeout=1200):
            return False
        return await force_click(page, btn, f"'{label_for_log}'")
    except Exception:
        return False


async def attempt_workday_advance_with_retry(
    page: Page,
    selector: str,
    label_for_log: str,
    current_url: str,
    step_signature_before: str,
    priority_guard_enabled: bool,
) -> dict[str, object]:
    unresolved: list[dict] = []
    errors: list[str] = []

    for attempt in range(WORKDAY_STEP_RETRY_BUDGET + 1):
        if attempt > 0:
            print(
                f"  ↻ Retry {attempt}/{WORKDAY_STEP_RETRY_BUDGET} for '{label_for_log}'"
            )
            if unresolved:
                print(
                    "  [priority] unresolved before retry: "
                    + ", ".join(item["key"] for item in unresolved)
                )
            refilled = await refill_unresolved_workday_priority_fields(page, unresolved)
            if not refilled and unresolved:
                print("  ! Retry aborted — unresolved fields could not be refilled.")
                break

        if not await click_workday_button(page, selector, label_for_log):
            return {
                "clicked": False,
                "completed": False,
                "advanced": False,
                "retry_exhausted": attempt > 0,
                "errors": errors,
                "unresolved": unresolved,
            }

        await asyncio.sleep(1.5)
        new_text = await page.inner_text("body")
        if await is_workday_final_confirmation(page, new_text):
            print("\n✅ APPLICATION SUBMITTED SUCCESSFULLY!")
            return {
                "clicked": True,
                "completed": True,
                "advanced": True,
                "retry_exhausted": False,
                "errors": [],
                "unresolved": [],
            }

        errors = await collect_workday_validation_errors(page)
        if errors:
            print(f"  ! Page returned {len(errors)} validation error(s):")
            for error in errors:
                print(f"     • {error}")

        priority_snapshot = await snapshot_workday_priority_fields(page)
        unresolved = [
            item for item in priority_snapshot.values()
            if item.get("present") and not item.get("answered")
        ]
        if unresolved:
            log_workday_priority_snapshot(priority_snapshot, "  [priority post-click]")

        step_signature_after = await get_workday_step_signature(page)
        advanced = page.url != current_url or step_signature_after != step_signature_before
        if advanced:
            print("  ✓ Page advanced to the next Workday step.")
            return {
                "clicked": True,
                "completed": False,
                "advanced": True,
                "retry_exhausted": False,
                "errors": errors,
                "unresolved": unresolved,
            }

        print("  ! Page stayed on the same Workday step.")
        if not priority_guard_enabled and not errors and not unresolved:
            print("  ↻ No priority required fields detected on this step; continuing.")
            return {
                "clicked": True,
                "completed": False,
                "advanced": True,
                "retry_exhausted": False,
                "errors": [],
                "unresolved": [],
            }
        if attempt >= WORKDAY_STEP_RETRY_BUDGET or not unresolved:
            return {
                "clicked": True,
                "completed": False,
                "advanced": False,
                "retry_exhausted": True,
                "errors": errors,
                "unresolved": unresolved,
            }

    return {
        "clicked": True,
        "completed": False,
        "advanced": False,
        "retry_exhausted": True,
        "errors": errors,
        "unresolved": unresolved,
    }


async def handle_radio_groups(page: Page, page_text: str):
    """Answer radio/yes-no question groups on Workday forms.

    Handles both native <input type="radio"> groups and Workday's custom
    [role="radio"] button groups.
    """
    # ── 1. Native radio groups, grouped by name attribute ──
    try:
        radios = page.locator("input[type='radio']:visible")
        r_count = await radios.count()

        seen_groups: set[str] = set()

        for i in range(r_count):
            r = radios.nth(i)
            try:
                if not await r.is_visible(timeout=500):
                    continue
                group_name = await r.get_attribute("name") or ""
                if not group_name or group_name in seen_groups:
                    continue
                seen_groups.add(group_name)

                # If any radio in this group is already checked, skip.
                group_sel = f"input[type='radio'][name='{group_name}']"
                checked_count = await page.locator(f"{group_sel}:checked").count()
                if checked_count > 0:
                    continue

                # Get the question: fieldset legend or closest label/heading.
                question = await r.evaluate(
                    """(el) => {
                        // Try fieldset > legend
                        const fs = el.closest('fieldset');
                        if (fs) {
                            const lg = fs.querySelector('legend');
                            if (lg) return lg.innerText.trim();
                        }
                        // Try aria-labelledby
                        const labelledBy = el.getAttribute('aria-labelledby');
                        if (labelledBy) {
                            const lbl = document.getElementById(labelledBy);
                            if (lbl) return lbl.innerText.trim();
                        }
                        // Try preceding sibling label
                        let p = el.parentElement;
                        for (let i = 0; i < 6 && p; i++) {
                            const lbl = p.querySelector('label, [role="label"]');
                            if (lbl && lbl.innerText) return lbl.innerText.trim();
                            p = p.parentElement;
                        }
                        return '';
                    }"""
                )
                question = (question or "").strip()
                if not question:
                    continue

                answer = decide_yes_no(question)
                if not answer:
                    # Let Grok pick from the available option values
                    options = await page.locator(group_sel).evaluate_all(
                        "nodes => nodes.map(n => n.value || (n.labels && n.labels[0] && n.labels[0].innerText) || '').filter(Boolean)"
                    )
                    if options:
                        print(f"  ? Asking Grok about radio group: '{question[:80]}'")
                        try:
                            answer = await asyncio.wait_for(
                                ask_grok_async(
                                    f"For the question '{question}', pick one of: {options}",
                                    context=page_text[:400],
                                ),
                                timeout=15,
                            )
                        except Exception:
                            answer = None

                if not answer:
                    continue

                # Click the matching radio
                picked = False
                opts_loc = page.locator(group_sel)
                opt_count = await opts_loc.count()
                answer_lower = answer.strip().lower()
                for j in range(opt_count):
                    opt = opts_loc.nth(j)
                    try:
                        val = (await opt.get_attribute("value") or "").lower()
                        opt_id = await opt.get_attribute("id") or ""
                        lbl_text = ""
                        if opt_id:
                            lbl = page.locator(f"label[for='{opt_id}']")
                            if await lbl.count() > 0:
                                lbl_text = (await lbl.text_content() or "").strip().lower()
                        if answer_lower in (val, lbl_text) or (
                            lbl_text and answer_lower in lbl_text
                        ):
                            try:
                                await opt.check(force=True, timeout=2500)
                            except Exception:
                                try:
                                    await opt.click(force=True, timeout=2500)
                                except Exception:
                                    continue
                            picked = True
                            break
                    except Exception:
                        continue

                if picked:
                    print(f"  ✓ Radio '{question[:70]}': {answer}")
                else:
                    print(f"  ! Could not pick '{answer}' for radio: {question[:70]}")
            except Exception:
                continue
    except Exception:
        pass

    # ── 2. Workday custom radios ([role="radio"]) ──
    try:
        custom_radios = page.locator("[role='radio']:visible")
        cr_count = await custom_radios.count()
        # Group by their containing [role="radiogroup"] via parent chain
        seen_group_ids: set[str] = set()
        for i in range(cr_count):
            r = custom_radios.nth(i)
            try:
                group_el = r.locator(
                    "xpath=ancestor::*[@role='radiogroup'][1]"
                ).first
                if await group_el.count() == 0:
                    continue
                gid = await group_el.get_attribute("id") or await group_el.evaluate(
                    "n => n.outerHTML.slice(0, 80)"
                )
                if gid in seen_group_ids:
                    continue
                seen_group_ids.add(gid)

                # Skip if any option in this group is already aria-checked=true
                already_checked = await group_el.locator(
                    "[role='radio'][aria-checked='true']"
                ).count()
                if already_checked > 0:
                    continue

                question = (
                    (await group_el.get_attribute("aria-label") or "").strip()
                    or (await group_el.evaluate(
                        """(g) => {
                            const lid = g.getAttribute('aria-labelledby');
                            if (lid) {
                                const l = document.getElementById(lid);
                                if (l) return l.innerText.trim();
                            }
                            const lbl = g.querySelector('label, legend');
                            if (lbl) return lbl.innerText.trim();
                            return '';
                        }"""
                    ))
                )
                if not question:
                    continue

                answer = decide_yes_no(question)
                if not answer:
                    continue

                options = group_el.locator("[role='radio']")
                opt_count = await options.count()
                picked = False
                for j in range(opt_count):
                    opt = options.nth(j)
                    try:
                        opt_label = (
                            (await opt.get_attribute("aria-label") or "").strip()
                            or (await opt.text_content() or "").strip()
                        )
                        if answer.lower() in opt_label.lower():
                            try:
                                await opt.click(force=True, timeout=2500)
                            except Exception:
                                continue
                            picked = True
                            break
                    except Exception:
                        continue
                if picked:
                    print(f"  ✓ Radio '{question[:70]}': {answer}")
            except Exception:
                continue
    except Exception:
        pass


async def toggle_workday_account_mode(page: Page, target: str) -> bool:
    """Toggle between Workday Sign In and Create Account forms.

    target: "create" to switch to Create Account, "signin" to switch to Sign In.
    Returns True if the toggle link/button was found and clicked.
    """
    # Links / buttons Workday exposes for the opposite action:
    # - "Create Account" link when on Sign In page
    # - "Sign In" link when on Create Account page
    if target == "create":
        selectors = [
            "[data-automation-id='createAccountLink']",
            "a:has-text('Create Account')",
            "button:has-text('Create Account')",
            "[data-automation-id*='createAccount']:not([data-automation-id*='SubmitButton'])",
        ]
    else:
        selectors = [
            "[data-automation-id='signInLink']",
            "a:has-text('Sign In')",
            "button:has-text('Sign In')",
            "[data-automation-id*='signIn']:not([data-automation-id*='SubmitButton'])",
        ]

    for sel in selectors:
        try:
            loc = page.locator(sel)
            n = await loc.count()
            for i in range(n):
                el = loc.nth(i)
                try:
                    if not await el.is_visible(timeout=600):
                        continue
                    txt = (await el.text_content() or "").strip()
                    # Skip the submit button which is not a toggle
                    if txt.lower() in ("submit", ""):
                        continue
                    print(f"  → Toggling account mode via: '{txt[:60]}'")
                    try:
                        await el.click(timeout=3000)
                    except Exception:
                        # Force-click via JS if the normal click fails
                        try:
                            await el.evaluate("node => node.click()")
                        except Exception:
                            continue
                    return True
                except Exception:
                    continue
        except Exception:
            continue
    return False


async def force_click(page: Page, locator, label: str = "") -> bool:
    """Click with aggressive fallbacks: normal → force → JS click."""
    try:
        await locator.click(timeout=4000)
        return True
    except Exception as e1:
        try:
            await locator.click(timeout=4000, force=True)
            print(f"  ↻ force-clicked {label}")
            return True
        except Exception as e2:
            try:
                await locator.evaluate("node => node.click()")
                print(f"  ↻ JS-clicked {label}")
                return True
            except Exception as e3:
                print(f"  ✗ Click failed for {label}: {str(e1).splitlines()[0][:80]}")
                return False


async def recover_from_workday_error(
    page: Page, apply_url: str, attempt: int
) -> bool:
    """
    Try multiple recovery strategies against the Workday
    "Something went wrong — please refresh the page and then try again." banner.

    Returns True if the banner is gone after recovery, False otherwise.
    Uses a different strategy each attempt so we don't spin on the same one.
    """
    strategies = [
        "wait_then_reload",     # attempt 1 — quick reload
        "hard_nav_apply_url",   # attempt 2 — re-GET the original apply URL
        "long_wait_reload",     # attempt 3 — Workday sometimes needs ~10s
        "hard_nav_apply_url",   # attempt 4 — one more try at the apply URL
    ]
    strategy = strategies[min(attempt, len(strategies) - 1)]
    print(f"  ↻ Recovery strategy: {strategy}")

    try:
        if strategy == "wait_then_reload":
            await asyncio.sleep(3)
            await page.reload(wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(5)

        elif strategy == "long_wait_reload":
            await asyncio.sleep(8)
            await page.reload(wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(6)

        elif strategy == "hard_nav_apply_url":
            if apply_url:
                print(f"  → Navigating back to apply URL: {apply_url[:90]}")
                await page.goto(
                    apply_url, wait_until="domcontentloaded", timeout=45000
                )
                await asyncio.sleep(6)
            else:
                # No apply URL on record — fall back to reload.
                await page.reload(wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(5)

    except Exception as e:
        print(f"  ! Recovery error: {e}")
        return False

    # Check if the banner is gone.
    try:
        body_text = (await page.inner_text("body")).lower()
        still_broken = (
            "something went wrong" in body_text
            and ("refresh the page" in body_text or "try again" in body_text)
        )
        return not still_broken
    except Exception:
        return False


async def handle_workday_page(page: Page):
    """Fill a Workday multi-step application."""
    print("\n[WORKDAY] Detected Workday ATS")

    # Wait for page to load
    await asyncio.sleep(1.2)

    # Remember the initial apply URL. Prefer the explicit env var
    # (WORKDAY_APPLY_URL) if provided, otherwise fall back to whatever URL
    # the page is currently sitting on. This is what we re-navigate to when
    # Workday loses its apply-flow context after sign-in.
    initial_apply_url = os.environ.get("WORKDAY_APPLY_URL", "").strip() or page.url
    refresh_budget = 4  # total recovery attempts

    # ── Pre-flight: if we land right into the error banner after sign-in,
    # recover BEFORE entering the step loop so step 1 starts clean.
    try:
        pre_text = (await page.inner_text("body")).lower()
        if (
            "something went wrong" in pre_text
            and ("refresh the page" in pre_text or "try again" in pre_text)
        ):
            print("  ⚠  Error banner present on page load — recovering first...")
            attempt = 0
            while refresh_budget > 0:
                refresh_budget -= 1
                ok = await recover_from_workday_error(
                    page, initial_apply_url, attempt
                )
                attempt += 1
                if ok:
                    print("  ✓ Banner cleared — continuing.")
                    break
                print(f"  ... still broken, retries left: {refresh_budget}")
    except Exception:
        pass

    for step in range(1, 12):  # max 12 steps
        print(f"\n[STEP {step}] Processing page...")
        await asyncio.sleep(0.8)

        current_url = page.url
        page_text = await page.inner_text("body")
        print(f"  URL: {current_url[:80]}")

        # ── Workday generic error banner recovery ────────────────────────────
        pt_lower = (page_text or "").lower()
        if (
            "something went wrong" in pt_lower
            and ("refresh the page" in pt_lower or "try again" in pt_lower)
        ):
            if refresh_budget > 0:
                print(
                    "  ⚠  Workday error banner detected mid-flow "
                    f"(retries left: {refresh_budget})"
                )
                attempt = 4 - refresh_budget  # rotates through strategies
                refresh_budget -= 1
                ok = await recover_from_workday_error(
                    page, initial_apply_url, attempt
                )
                if ok:
                    print("  ✓ Banner cleared — retrying this step.")
                # Either way, restart this step with fresh page state.
                continue
            else:
                print("  ✗ Recovery budget exhausted — taking screenshot and stopping.")
                try:
                    await page.screenshot(
                        path=str(_ROOT / "workday_error.png"), full_page=True
                    )
                except Exception:
                    pass
                return False

        # ── Detect & handle account-gate mode: create / signin / none ──
        # Workday redirects to `/login?redirect=...` when the email already has
        # an account. When that happens, forget WORKDAY_MODE=create and sign in.
        account_mode = "none"
        try:
            pw_count = await page.locator("input[type='password']:visible").count()
            body_lower = (page_text or "").lower()
            url_lower = current_url.lower()

            account_exists_redirect = (
                "/login" in url_lower
                or "redirect=" in url_lower
                or "an account already exists" in body_lower
                or "already have an account" in body_lower
            )

            if pw_count > 0:
                current_is_create = pw_count >= 2

                # Decide desired mode (ignore env var if Workday already told us
                # the account exists via /login redirect).
                if account_exists_redirect:
                    print(
                        "  [ACCOUNT] Workday redirected to /login — account already exists, "
                        "switching to SIGN IN."
                    )
                    desired = "signin"
                elif WORKDAY_MODE == "signin":
                    desired = "signin"
                elif WORKDAY_MODE == "create":
                    desired = "create"
                else:
                    desired = "create" if not current_is_create else "create"

                # Toggle if we're not in the desired mode
                if desired == "create" and not current_is_create:
                    print("  [ACCOUNT] Page shows Sign In — toggling to Create Account...")
                    toggled = await toggle_workday_account_mode(page, "create")
                    if toggled:
                        await asyncio.sleep(0.8)
                        pw_count = await page.locator("input[type='password']:visible").count()
                        current_is_create = pw_count >= 2
                    else:
                        print("  ! Could not find Create Account toggle — staying on Sign In")

                elif desired == "signin" and current_is_create:
                    print("  [ACCOUNT] Page shows Create Account — toggling to Sign In...")
                    toggled = await toggle_workday_account_mode(page, "signin")
                    if toggled:
                        await asyncio.sleep(0.8)
                        pw_count = await page.locator("input[type='password']:visible").count()
                        current_is_create = pw_count >= 2

                account_mode = "create" if current_is_create else "signin"
                print(
                    f"  [ACCOUNT] Mode: {account_mode.upper()} "
                    f"(WORKDAY_MODE={WORKDAY_MODE}, password fields visible: {pw_count})"
                )
        except Exception as e:
            print(f"  ! Account-mode detection error: {e}")

        # ── Fill visible text inputs (incl. password for Workday account creation) ──
        inputs = page.locator(
            "input[type='text']:visible, input[type='email']:visible, "
            "input[type='tel']:visible, input[type='password']:visible, "
            "input[type='url']:visible, input:not([type]):visible"
        )
        count = await inputs.count()

        for i in range(count):
            inp = inputs.nth(i)
            try:
                if not await inp.is_visible(timeout=800):
                    continue

                # ── Gather attributes ──
                inp_id = await inp.get_attribute("id") or ""
                inp_name = await inp.get_attribute("name") or ""
                inp_placeholder = await inp.get_attribute("placeholder") or ""
                inp_aria = await inp.get_attribute("aria-label") or ""
                inp_autocomplete = await inp.get_attribute("autocomplete") or ""
                inp_type = (await inp.get_attribute("type") or "").lower()
                inp_readonly = await inp.get_attribute("readonly")
                inp_role = (await inp.get_attribute("role") or "").lower()
                inp_aria_haspopup = (await inp.get_attribute("aria-haspopup") or "").lower()
                inp_auto_id = (await inp.get_attribute("data-automation-id") or "").lower()

                # Skip Workday combobox inputs — they're handled by handle_workday_dropdown
                # (these have aria-haspopup, role=combobox, or are promptOption inputs).
                if (
                    inp_readonly is not None
                    or inp_role in ("combobox", "listbox")
                    or inp_aria_haspopup in ("listbox", "menu", "true")
                    or "promptoption" in inp_auto_id
                    or "prompt" in inp_auto_id
                ):
                    continue

                current_val = await inp.input_value()
                if current_val:  # already filled
                    continue

                label_text = ""
                if inp_id:
                    lbl = page.locator(f"label[for='{inp_id}']")
                    if await lbl.count() > 0:
                        label_text = (await lbl.text_content() or "").strip()

                # Workday's "Application Questions" page often has NO label[for]
                # and no useful aria-label. The question text sits in an
                # ancestor element (e.g. <div data-automation-id='questionText'>).
                # Walk up the DOM to find it so keyword matching actually works.
                if not label_text:
                    try:
                        ancestor_label = await inp.evaluate(
                            """(el) => {
                                let p = el.parentElement;
                                for (let i = 0; i < 12 && p; i++) {
                                    const q = p.querySelector(
                                        "[data-automation-id='questionText'], " +
                                        "[data-automation-id='formLabel'], " +
                                        "[data-automation-id*='promptLabel'], " +
                                        "label:not([for=''])"
                                    );
                                    if (q && q.innerText) {
                                        const t = q.innerText.trim();
                                        if (t) return t;
                                    }
                                    const kids = Array.from(p.children || []);
                                    for (const k of kids) {
                                        const t = (k.innerText || '').trim();
                                        if (t && t.length < 250 && /\\?\\*?\\s*$/.test(t)) {
                                            return t;
                                        }
                                    }
                                    p = p.parentElement;
                                }
                                return '';
                            }"""
                        )
                        if ancestor_label:
                            label_text = ancestor_label.strip()
                    except Exception:
                        pass

                field_hint = label_text or inp_aria or inp_placeholder or inp_name or inp_id
                fl = field_hint.lower()
                ac = inp_autocomplete.lower()
                nm = (inp_name or "").lower()

                # ── Skip honeypot / anti-bot fields ──
                # Workday embeds a hidden "website" input ("for robots only, do not enter").
                if "robot" in fl or "do not enter" in fl or "for robots" in fl:
                    continue
                if nm == "website" and ("robot" in (inp_aria or "").lower() or "robot" in fl):
                    continue

                # Map common fields directly. Order matters — more specific checks first.
                value = None

                if inp_type == "password" or "password" in fl:
                    value = PROFILE.get("ats_account_password") or os.environ.get("ATS_ACCOUNT_PASSWORD") or "InterLink!Apply$2026"
                elif "extension" in fl or "ext." in fl or "ext " in fl:
                    # Phone extension: leave blank unless you actually have one
                    continue
                elif any(x in fl for x in ["first name", "firstname", "given"]) or ac == "given-name":
                    value = PROFILE["first_name"]
                elif any(x in fl for x in ["last name", "lastname", "family", "surname"]) or ac == "family-name":
                    value = PROFILE["last_name"]
                elif "middle" in fl:
                    continue  # no middle name in profile
                elif "email" in fl or ac == "email" or "email" in nm:
                    value = PROFILE["email"]
                elif "phone number" in fl or fl.strip() == "phone" or fl.endswith("phone*") or ac == "tel" or (
                    "phone" in fl and "code" not in fl and "country" not in fl and "type" not in fl
                ):
                    # Workday has a separate Country Phone Code dropdown — the
                    # phone field wants only the local number. Strip "+1-", "+1 ",
                    # country-code leading digits, parentheses, and spaces.
                    raw = PROFILE["phone"]
                    local = re.sub(r"^\s*\+\d{1,3}[\s\-\.]*", "", raw)
                    local = local.replace("(", "").replace(")", "").strip()
                    value = local
                elif any(x in fl for x in ["linkedin", "linked in"]):
                    value = PROFILE["linkedin"]
                elif "github" in fl:
                    value = PROFILE["github"]
                elif any(x in fl for x in ["portfolio", "personal website", "personal site"]):
                    value = PROFILE["portfolio"]
                elif "website" in fl or fl.strip() == "url":
                    # Most ATS "website" fields are honeypots OR optional URL fields.
                    # Use portfolio if label makes it clearly legit; else skip.
                    continue
                elif any(x in fl for x in ["city"]):
                    value = PROFILE["city"]
                elif any(x in fl for x in ["zip", "postal"]):
                    value = PROFILE["zip"]
                elif any(x in fl for x in ["address line 1", "address 1", "street"]):
                    value = PROFILE["address"]
                elif "address line 2" in fl or "address 2" in fl or "apt" in fl or "suite" in fl:
                    continue
                elif any(
                    x in fl
                    for x in [
                        "salary",
                        "compensation",
                        "pay",
                        "salary expectations",
                        "salary expectation",
                    ]
                ):
                    value = PROFILE["salary_expectation"]
                elif any(x in fl for x in ["gpa", "grade"]):
                    value = PROFILE["gpa"]
                elif any(x in fl for x in ["university", "school", "college", "institution"]):
                    value = PROFILE["university"]
                elif any(x in fl for x in ["major", "field of study", "concentration"]):
                    value = PROFILE["major"]
                elif any(x in fl for x in ["graduation", "grad year", "class of"]):
                    value = PROFILE["graduation_year"]
                elif any(x in fl for x in ["title", "position", "role", "job title"]):
                    value = PROFILE["current_title"]
                elif any(x in fl for x in ["years of experience", "years experience"]):
                    value = PROFILE["years_experience"]
                elif field_hint and len(field_hint) < 120:
                    # Ask Grok for unknown fields (guard against huge chunks that aren't labels)
                    print(f"  ? Asking Grok about field: '{field_hint}'")
                    try:
                        value = await asyncio.wait_for(
                            ask_grok_async(
                                f"What value should I enter for the field labeled '{field_hint}'?",
                                context=f"Current page section: {page_text[:500]}",
                            ),
                            timeout=20,
                        )
                    except Exception:
                        value = None

                if value:
                    try:
                        # Short timeout — don't waste 30s per uncooperative field
                        await inp.click(timeout=2500)
                        await inp.fill("", timeout=1500)
                        await inp.fill(value, timeout=2500)
                        display_val = "•" * len(value) if inp_type == "password" else value[:50]
                        print(f"  ✓ {field_hint}: {display_val}")
                    except Exception as fill_err:
                        short_err = str(fill_err).splitlines()[0][:90]
                        print(f"  ! Skip '{field_hint}': {short_err}")

            except Exception:
                continue

        # ── First-pass: specific handler for the Application Questions page
        # where every dropdown literally displays "Select One" and the
        # question text lives in an ancestor. This handler is extremely
        # reliable because it matches by visible text.
        await handle_select_one_questions(page)

        # ── Handle Workday custom dropdowns (buttons with aria-haspopup) ──
        await handle_workday_dropdowns(page)

        # ── Handle radio / yes-no question groups ──
        await handle_radio_groups(page, page_text or "")

        # ── Fill textareas ──
        textareas = page.locator("textarea:visible")
        ta_count = await textareas.count()
        for i in range(ta_count):
            ta = textareas.nth(i)
            try:
                current = await ta.input_value()
                if current:
                    continue
                ta_id = await ta.get_attribute("id") or ""
                ta_label = ""
                if ta_id:
                    lbl = page.locator(f"label[for='{ta_id}']")
                    if await lbl.count() > 0:
                        ta_label = (await lbl.text_content() or "").strip()

                fl = ta_label.lower()
                if any(x in fl for x in ["salary", "compensation", "pay expectation", "salary expectation"]):
                    value = PROFILE.get(
                        "salary_expectation",
                        "Competitive / Open to discussion based on standard intern rates",
                    )
                elif any(x in fl for x in ["cover", "letter", "motivation", "why"]):
                    value = PROFILE["cover_letter"]
                elif "role description" in fl or "job description" in fl or "responsibilities" in fl:
                    # Workday Work Experience "Role Description". We can't know
                    # which entry index this is without scoping — fall through to
                    # a generic answer from skills_summary as a safe default.
                    value = PROFILE.get("skills_summary", "")
                elif any(x in fl for x in ["skill", "background", "summary", "describe yourself"]):
                    value = PROFILE["skills_summary"]
                elif ta_label:
                    print(f"  ? Asking Grok about textarea: '{ta_label}'")
                    try:
                        value = await asyncio.wait_for(
                            ask_grok_async(
                                f"What should I write for the textarea field labeled '{ta_label}'? Keep it under 100 words.",
                                context=f"Job application for software engineering role. {page_text[:300]}",
                            ),
                            timeout=20,
                        )
                    except Exception:
                        value = None
                else:
                    continue

                await ta.fill(value)
                print(f"  ✓ Textarea '{ta_label}': {value[:60]}...")
                await asyncio.sleep(0.3)
            except Exception:
                continue

        # ── Handle selects / dropdowns ──
        selects = page.locator("select:visible")
        sel_count = await selects.count()
        for i in range(sel_count):
            sel = selects.nth(i)
            try:
                sel_id = await sel.get_attribute("id") or ""
                sel_name = await sel.get_attribute("name") or ""

                sel_label = ""
                if sel_id:
                    lbl = page.locator(f"label[for='{sel_id}']")
                    if await lbl.count() > 0:
                        sel_label = (await lbl.text_content() or "").strip()

                fl = sel_label.lower()
                value = None

                if any(x in fl for x in ["country", "nation"]):
                    value = PROFILE["country"]
                elif any(x in fl for x in ["state", "province", "region"]):
                    value = PROFILE["state"]
                elif any(
                    x in fl
                    for x in [
                        "legally authorized",
                        "authorized to work",
                        "work authorization",
                        "work auth",
                    ]
                ):
                    value = PROFILE["work_auth"]
                elif any(x in fl for x in ["sponsor", "sponsorship", "visa"]):
                    value = PROFILE["sponsorship"]
                elif any(x in fl for x in ["relocate", "relocation", "other cities"]):
                    value = PROFILE["relocation"]
                elif any(x in fl for x in ["degree", "education level"]):
                    value = "Bachelor's Degree"
                elif any(x in fl for x in ["gender"]):
                    value = "Prefer not to say"
                elif any(x in fl for x in ["ethnicity", "race"]):
                    value = "Prefer not to say"
                elif any(x in fl for x in ["veteran"]):
                    value = "No"
                elif any(x in fl for x in ["disability"]):
                    value = "No"
                elif sel_label:
                    # Get all options and ask Grok
                    options = await sel.locator("option").all_text_contents()
                    options_clean = [o.strip() for o in options if o.strip() and o.strip() != ""]
                    if options_clean:
                        print(f"  ? Asking Grok about select '{sel_label}'")
                        value = await ask_grok_async(
                            f"For dropdown '{sel_label}', pick the best option. Options: {options_clean}",
                        )

                if value:
                    await select_option(
                        page,
                        f"#{sel_id}" if sel_id else f"select[name='{sel_name}']",
                        value,
                        sel_label,
                    )
                    await asyncio.sleep(0.3)
            except Exception:
                continue

        # ── Handle Workday-specific custom dropdowns (not native <select>) ──
        # Workday uses custom div-based dropdowns with data-automation-id
        workday_inputs = page.locator(
            "[data-automation-id*='Input']:visible, [data-automation-id*='input']:visible"
        )
        wd_count = await workday_inputs.count()
        for i in range(min(wd_count, 20)):
            el = workday_inputs.nth(i)
            try:
                auto_id = await el.get_attribute("data-automation-id") or ""
                current = await el.input_value()
                if current:
                    continue

                fl = auto_id.lower()
                value = None

                if "firstname" in fl or "legalname_first" in fl:
                    value = PROFILE["first_name"]
                elif "lastname" in fl or "legalname_last" in fl:
                    value = PROFILE["last_name"]
                elif "email" in fl:
                    value = PROFILE["email"]
                elif "phone" in fl:
                    value = PROFILE["phone"]
                elif "address" in fl:
                    value = PROFILE["address"]
                elif "city" in fl:
                    value = PROFILE["city"]
                elif "postalcode" in fl or "zipcode" in fl:
                    value = PROFILE["zip"]

                if value:
                    await el.fill(value)
                    print(f"  ✓ Workday [{auto_id}]: {value}")
                    await asyncio.sleep(0.3)
            except Exception:
                continue

        # ── Upload resume if file input exists ──
        resume_path = next((p for p in RESUME_CANDIDATES if p.exists()), None)

        if resume_path:
            file_inputs = page.locator("input[type='file']")
            fi_count = await file_inputs.count()
            if fi_count > 0:
                try:
                    # Only upload if no file has been attached yet — otherwise
                    # we'd re-upload every step and Workday would reject it.
                    already_uploaded_text = (page_text or "").lower()
                    if (
                        "successfully" in already_uploaded_text
                        and (
                            "upload" in already_uploaded_text
                            or resume_path.name.lower() in already_uploaded_text
                        )
                    ):
                        pass  # resume is already on the page
                    else:
                        await file_inputs.first.set_input_files(str(resume_path))
                        print(f"  ✓ Uploaded resume: {resume_path.name}")
                        # Workday's autofillWithResume flow parses the PDF
                        # server-side; give it a few seconds to populate the
                        # form before we continue. We poll for either the
                        # "successful upload" toast or a real text input.
                        for _ in range(12):  # up to ~6s
                            try:
                                body_lc = (
                                    await page.inner_text("body", timeout=1500)
                                ).lower()
                                visible_text_inputs = await page.locator(
                                    "input[type='text']:visible, input[type='email']:visible"
                                ).count()
                                if (
                                    "successfully" in body_lc
                                    or visible_text_inputs > 0
                                ):
                                    break
                            except Exception:
                                pass
                            await asyncio.sleep(0.5)
                except Exception as e:
                    print(f"  ! Resume upload error: {e}")

        # ── Scroll any "consent text" containers to the bottom first ──
        # Workday sometimes requires the user to scroll through the agreement
        # text before the consent checkbox is enabled.
        try:
            await page.evaluate(
                """() => {
                    const candidates = document.querySelectorAll(
                        "[data-automation-id*='gdprText'], " +
                        "[data-automation-id*='consentNotice'], " +
                        "[data-automation-id*='AgreementText'], " +
                        "[data-automation-id*='legalConsent'], " +
                        "[class*='consent'], [class*='agreement'], [class*='termsText']"
                    );
                    candidates.forEach(el => {
                        try {
                            if (el.scrollHeight > el.clientHeight) {
                                el.scrollTop = el.scrollHeight;
                            }
                        } catch (e) {}
                    });
                    // Also scroll the main window to bottom in case the agreement
                    // is just long body text.
                    try { window.scrollTo(0, document.body.scrollHeight); } catch (e) {}
                }"""
            )
            await asyncio.sleep(0.25)
        except Exception:
            pass

        # ── Handle the priority Workday agreement checkbox first ──
        try:
            agreement = page.locator(
                "[data-automation-id='agreementCheckbox'], "
                "input[data-automation-id='agreementCheckbox'], "
                "input[type='checkbox'][data-automation-id*='greement']:visible"
            ).first
            if await agreement.count() > 0 and await agreement.is_visible(timeout=800):
                if not await agreement.is_checked():
                    try:
                        await agreement.check(force=True, timeout=3000)
                        print("  ✓ Checked: Agreement (data-automation-id='agreementCheckbox')")
                    except Exception:
                        try:
                            await agreement.evaluate("n => n.click()")
                            print("  ✓ Checked agreement via JS click")
                        except Exception as e:
                            print(f"  ! Could not check agreement: {e}")
        except Exception:
            pass

        # ── Tick any other "I agree / terms / consent" checkboxes ──
        try:
            checkboxes = page.locator("input[type='checkbox']:visible")
            cb_count = await checkboxes.count()
            for i in range(cb_count):
                cb = checkboxes.nth(i)
                try:
                    if await cb.is_visible(timeout=500) and not await cb.is_checked():
                        cb_id = await cb.get_attribute("id") or ""
                        cb_name = await cb.get_attribute("name") or ""
                        cb_aria = await cb.get_attribute("aria-label") or ""
                        cb_auto = (await cb.get_attribute("data-automation-id") or "").lower()
                        lbl_text = ""
                        if cb_id:
                            l = page.locator(f"label[for='{cb_id}']")
                            if await l.count() > 0:
                                lbl_text = (await l.text_content() or "").strip()

                        # Fallback: Workday's checkbox group items use a wrapping
                        # <label> OR an adjacent <span> instead of label[for=...].
                        # Walk up the DOM to find the closest text that looks
                        # like a label (e.g. "More than 3 Months").
                        if not lbl_text:
                            try:
                                lbl_text = await cb.evaluate(
                                    """(el) => {
                                        // parent <label>
                                        const parentLabel = el.closest('label');
                                        if (parentLabel) {
                                            const t = parentLabel.innerText.trim();
                                            if (t) return t;
                                        }
                                        // sibling text
                                        let sib = el.nextElementSibling;
                                        while (sib) {
                                            const t = (sib.innerText || '').trim();
                                            if (t) return t;
                                            sib = sib.nextElementSibling;
                                        }
                                        // ancestor with short text
                                        let p = el.parentElement;
                                        for (let i = 0; i < 4 && p; i++) {
                                            const t = (p.innerText || '').trim();
                                            if (t && t.length < 80) return t;
                                            p = p.parentElement;
                                        }
                                        return '';
                                    }"""
                                )
                                lbl_text = (lbl_text or "").strip()
                            except Exception:
                                pass
                        option_text = lbl_text
                        question_text = ""
                        try:
                            question_text = await cb.evaluate(
                                """(el) => {
                                    let p = el.parentElement;
                                    for (let i = 0; i < 10 && p; i++) {
                                        const q = p.querySelector(
                                            "[data-automation-id='questionText'], " +
                                            "[data-automation-id='formLabel'], " +
                                            "[data-automation-id*='promptLabel'], " +
                                            "legend"
                                        );
                                        if (q) {
                                            const t = (q.innerText || '').trim();
                                            if (t) return t;
                                        }
                                        let sib = p.previousElementSibling;
                                        for (let j = 0; j < 4 && sib; j++) {
                                            const q2 = sib.querySelector?.(
                                                "[data-automation-id='questionText'], " +
                                                "[data-automation-id='formLabel'], " +
                                                "[data-automation-id*='promptLabel'], " +
                                                "legend"
                                            );
                                            const t2 = ((q2 && q2.innerText) || sib.innerText || '').trim();
                                            if (t2 && t2.length < 250 && /[\\?\\*]\\s*$/.test(t2)) {
                                                return t2;
                                            }
                                            sib = sib.previousElementSibling;
                                        }
                                        p = p.parentElement;
                                    }
                                    return '';
                                }"""
                            )
                            question_text = (question_text or "").strip()
                        except Exception:
                            pass
                        hint = f"{cb_id} {cb_name} {cb_aria} {lbl_text} {cb_auto}".lower()
                        if any(
                            k in hint
                            for k in [
                                "agree",
                                "accept",
                                "terms",
                                "privacy",
                                "acknowledge",
                                "consent",
                                "confirm",
                                "disclos",
                            ]
                        ):
                            try:
                                await cb.check(force=True, timeout=2500)
                                print(f"  ✓ Checked: {lbl_text[:80] or hint[:80]}")
                            except Exception:
                                try:
                                    await cb.evaluate("n => n.click()")
                                except Exception:
                                    pass
                            await asyncio.sleep(0.2)
                        else:
                            # Notice-period / availability checkbox group.
                            # Only tick the specific option that matches
                            # PROFILE["availability_choice"] (case-insensitive).
                            desired_avail = str(
                                PROFILE.get("availability_choice", "")
                            ).strip().lower()
                            avail_label = _norm_text(option_text)
                            avail_question = _norm_text(question_text)
                            desired_avail_norm = _norm_text(desired_avail)
                            if (
                                desired_avail_norm
                                and avail_label
                                and (
                                    desired_avail_norm == avail_label
                                    or desired_avail_norm in avail_label
                                    or avail_label in desired_avail_norm
                                )
                                and (
                                    any(
                                        k in avail_question
                                        for k in [
                                            "notice period",
                                            "available to start work",
                                            "start work",
                                            "availability",
                                        ]
                                    )
                                    or avail_label == desired_avail_norm
                                )
                            ):
                                    try:
                                        await cb.check(force=True, timeout=2500)
                                        print(
                                            f"  ✓ Checked (availability): {option_text[:80]}"
                                        )
                                    except Exception:
                                        try:
                                            await cb.evaluate("n => n.click()")
                                            print(
                                                f"  ✓ Checked (availability, JS): {option_text[:80]}"
                                            )
                                        except Exception:
                                            pass
                                    await asyncio.sleep(0.2)
                except Exception:
                    continue
        except Exception:
            pass

        # ── Try to advance to next step ──
        await asyncio.sleep(0.4)
        step_signature_before = await get_workday_step_signature(page)
        priority_snapshot_before = await snapshot_workday_priority_fields(page)
        if any(item.get("present") for item in priority_snapshot_before.values()):
            log_workday_priority_snapshot(priority_snapshot_before)

        # Look for Create Account / Sign In / Next / Continue / Submit buttons.
        # Order matters: try account-creation & navigation before Submit, so we
        # don't prematurely submit an incomplete application.
        # Order the button selectors based on the detected account mode so we
        # don't click "Create Account" when the account already exists, and
        # vice-versa.
        if account_mode == "signin":
            account_gate = [
                "[data-automation-id='signInSubmitButton']",
                "button:has-text('Sign In')",
                "button:has-text('Log In')",
            ]
        elif account_mode == "create":
            account_gate = [
                "[data-automation-id='createAccountSubmitButton']",
                "button:has-text('Create Account')",
            ]
        else:
            account_gate = []

        next_selectors = [
            *account_gate,
            "[data-automation-id='click_filter']",
            "button:has-text('Apply Now')",
            "button:has-text('Apply manually')",
            # Workday multi-step navigation
            "[data-automation-id='bottom-navigation-next-button']",
            "[data-automation-id='bottom-navigation-forward-button']",
            "button:has-text('Save and Continue')",
            "button:has-text('Continue')",
            "button:has-text('Next')",
            "button:has-text('Review')",
            # Final submit (AGENT_DRY_RUN will skip these)
            "button:has-text('Submit Application')",
            "button:has-text('Submit')",
            "button[type='submit']",
            "input[type='submit']",
        ]

        clicked_next = False
        disabled_button_text = ""
        for sel in next_selectors:
            try:
                btn = page.locator(sel).last
                if not await btn.is_visible(timeout=1500):
                    continue

                btn_text = (await btn.text_content() or "").strip()
                aria_label = (await btn.get_attribute("aria-label") or "").strip()
                auto_id = (await btn.get_attribute("data-automation-id") or "").strip()

                # A trustworthy button must have SOME identifying text. Skip
                # anonymous `button[type='submit']` on pages like the interstitial
                # login redirect (that was clicking '').
                label_for_log = btn_text or aria_label or auto_id
                if not label_for_log:
                    continue

                # Log disabled buttons so the user can see what's being skipped.
                is_enabled = await btn.is_enabled(timeout=800)
                is_disabled_attr = await btn.get_attribute("disabled")
                aria_disabled = await btn.get_attribute("aria-disabled")

                if not is_enabled or is_disabled_attr is not None or aria_disabled == "true":
                    disabled_button_text = label_for_log
                    print(
                        f"  ⊗ '{label_for_log}' is disabled "
                        f"(enabled={is_enabled}, disabled_attr={is_disabled_attr}, "
                        f"aria-disabled={aria_disabled})"
                    )
                    continue

                # Only trip DRY RUN on terminal-submit-style buttons.
                # We exclude "Save and Continue" and similar intermediate actions.
                label_lower = label_for_log.lower().strip()
                is_final_submit = (
                    label_lower in ("submit", "submit application", "send application")
                    or "submit application" in label_lower
                    or "submit my application" in label_lower
                )
                if AGENT_DRY_RUN and is_final_submit:
                    print(f"  🛟  [DRY RUN] Stopping here — would click '{label_for_log}'")
                    print("      Re-run with AGENT_SUBMIT=1 to actually submit.")
                    # Leave the browser open a bit so the user can visually review
                    return True

                print(f"\n  → Clicking: '{label_for_log}'")
                outcome = await attempt_workday_advance_with_retry(
                    page,
                    sel,
                    label_for_log,
                    current_url,
                    step_signature_before,
                    any(item.get("present") for item in priority_snapshot_before.values()),
                )

                if not outcome.get("clicked"):
                    continue

                clicked_next = True
                if outcome.get("completed"):
                    return True

                unresolved = outcome.get("unresolved") or []
                if outcome.get("retry_exhausted") and not outcome.get("advanced"):
                    if unresolved:
                        print(
                            "  ! Required Workday fields still unresolved: "
                            + ", ".join(item["key"] for item in unresolved)
                        )
                    else:
                        print("  ! Workday step did not advance after retries.")
                    stuck_path = _ROOT / "stuck.png"
                    await page.screenshot(path=str(stuck_path), full_page=True)
                    print(f"  Screenshot saved to {stuck_path}")
                    return False
                break
            except Exception:
                continue

        if not clicked_next:
            print("  ! No clickable next/submit button found.")
            if disabled_button_text:
                print(
                    f"    (The '{disabled_button_text}' button exists but is disabled — "
                    "likely missing required fields or unchecked Terms checkbox.)"
                )
            body = await page.inner_text("body")
            if await is_workday_final_confirmation(page, body):
                print("\n✅ Application appears complete!")
                return True
            else:
                print("  ! Stuck — taking screenshot")
                stuck_path = _ROOT / "stuck.png"
                await page.screenshot(path=str(stuck_path), full_page=True)
                print(f"  Screenshot saved to {stuck_path}")
                return False

    return False


async def handle_generic_form(page: Page):
    """Generic form handler for non-Workday ATS."""
    print("  Running generic form fill...")
    await asyncio.sleep(2)

    # Fill common fields
    field_map = {
        "input[name*='first_name' i], input[id*='first_name' i], input[placeholder*='first name' i]": PROFILE[
            "first_name"
        ],
        "input[name*='last_name' i], input[id*='last_name' i], input[placeholder*='last name' i]": PROFILE[
            "last_name"
        ],
        "input[name*='email' i], input[id*='email' i], input[type='email']": PROFILE["email"],
        "input[name*='phone' i], input[id*='phone' i], input[type='tel']": PROFILE["phone"],
        "input[name*='linkedin' i], input[id*='linkedin' i], input[placeholder*='linkedin' i]": PROFILE["linkedin"],
        "input[name*='github' i], input[id*='github' i]": PROFILE["github"],
        "input[name*='website' i], input[id*='website' i], input[name*='portfolio' i]": PROFILE["portfolio"],
    }

    for selector, value in field_map.items():
        for sel in selector.split(", "):
            await fill_field(page, sel.strip(), value)
            break

    # Upload resume
    resume_path = next((p for p in RESUME_CANDIDATES if p.exists()), None)
    if resume_path:
        try:
            await page.locator("input[type='file']").first.set_input_files(str(resume_path))
            print("  ✓ Uploaded resume")
            await asyncio.sleep(2)
        except Exception:
            pass

    # Submit
    for sel in ["button[type='submit']", "button:has-text('Submit')", "input[type='submit']"]:
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=2000):
                st = (await btn.text_content() or "").strip()
                print("  → Submitting...")
                if AGENT_DRY_RUN and "submit" in st.lower():
                    print("  [DRY RUN] Would submit here — skipping")
                    return True
                await btn.click()
                await asyncio.sleep(3)
                return True
        except Exception:
            continue

    return False


async def main():
    print("=" * 60)
    print("InterLink AI Agent — Auto Apply")
    print("=" * 60)
    if AGENT_DRY_RUN:
        print("🛟  DRY RUN — will fill every field and click Save/Continue,")
        print("    but will STOP at the final Submit button (safe to re-run).")
        print("    To actually submit, run with: AGENT_SUBMIT=1 python agent.py")
    else:
        print("🚀  LIVE MODE — will click Submit Application at the final step.")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=False,  # set False to watch the browser
            slow_mo=50,  # 50ms — just enough to look human; 0 = max speed
            args=["--disable-blink-features=AutomationControlled"],
        )

        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        apply_url = os.environ.get("APPLY_URL", "").strip()
        clicked = False
        if apply_url:
            print(f"\n[1] Opening apply link: {apply_url}")
            await page.goto(apply_url, wait_until="domcontentloaded", timeout=30000)
            if "workday" in apply_url.lower():
                os.environ["WORKDAY_APPLY_URL"] = apply_url
            clicked = True
        else:
            # ── Step 1: Inject session into localStorage then navigate ──
            # The app checks localStorage for 'interlinkApplicant' and 'interlinkRole'.
            # We must visit the origin first (even a blank page) before localStorage is writable.
            base_url = INTERLINK_URL.split("/applicant")[0]  # e.g. http://localhost:5173
            print(f"\n[1] Setting session at: {base_url}")
            await page.goto(base_url, wait_until="domcontentloaded", timeout=30000)
            # The layout checks localStorage['interlinkApplicant'] === APPLICANT_EMAIL
            # (see src/applicant/config.ts and src/layouts/ApplicantLayout.tsx).
            applicant_email = os.environ.get("APPLICANT_EMAIL", PROFILE.get("email", ""))
            await page.evaluate(
                """(email) => {
                    localStorage.setItem('interlinkApplicant', email);
                    localStorage.setItem('interlinkRole', 'applicant');
                }""",
                applicant_email,
            )
            print(f"  ✓ Session injected (interlinkApplicant={applicant_email})")

            print(f"\n[2] Navigating to: {INTERLINK_URL}")
            await page.goto(INTERLINK_URL, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(3)
            print("  ✓ Jobs page loaded")

            # ── Step 3: Find the target job card and click its Apply button ──
            # TARGET_COMPANY can be overridden via env var. Defaults to Revvity.
            target_company = os.environ.get("TARGET_COMPANY", "Revvity")
            print(f"\n[3] Looking for Apply button on '{target_company}' card...")

            clicked = False

            # Primary strategy: find the job-card containing the target company name,
            # then click the Apply button inside it.
            try:
                card = page.locator(".job-card", has_text=target_company).first
                if await card.count() > 0:
                    # Scroll into view so the button is clickable
                    await card.scroll_into_view_if_needed()
                    await asyncio.sleep(0.3)

                    apply_btn = card.locator("button:has-text('Apply')").first
                    if await apply_btn.is_visible(timeout=3000):
                        role_text = ""
                        try:
                            role_text = (
                                await card.locator(".job-card__role").first.text_content()
                                or ""
                            ).strip()
                        except Exception:
                            pass
                        print(f"  ✓ Found '{target_company}' card: {role_text}")
                        print("  → Clicking Apply — waiting for new tab...")

                        # Apply for jobs with applyUrl opens a new tab via window.open
                        try:
                            async with context.expect_page(timeout=8000) as new_page_info:
                                await apply_btn.click()
                            new_page = await new_page_info.value
                            await new_page.wait_for_load_state(
                                "domcontentloaded", timeout=20000
                            )
                            page = new_page
                            # Pin the ORIGINAL apply URL as soon as we land on it,
                            # before Workday can redirect us to /login?... or a
                            # broken dashboard. handle_workday_page reads this
                            # env var to recover from "Something went wrong".
                            try:
                                from urllib.parse import urlparse, parse_qs, unquote

                                initial_url = page.url or ""
                                pinned = ""
                                if "workday" in initial_url.lower():
                                    # If Workday already redirected us to
                                    # /login?redirect=<URL-encoded apply URL>,
                                    # unwrap the real apply URL from the query.
                                    parsed = urlparse(initial_url)
                                    qs = parse_qs(parsed.query)
                                    redirect_val = (qs.get("redirect") or [""])[0]
                                    if redirect_val:
                                        decoded = unquote(redirect_val)
                                        if decoded.startswith("/"):
                                            decoded = f"{parsed.scheme}://{parsed.netloc}{decoded}"
                                        pinned = decoded
                                    elif "/login" not in parsed.path.lower():
                                        pinned = initial_url
                                if pinned:
                                    os.environ["WORKDAY_APPLY_URL"] = pinned
                                    print(f"  ✓ Pinned apply URL: {pinned[:90]}")
                            except Exception as e:
                                print(f"  ! Could not pin apply URL: {e}")
                            clicked = True
                        except PWTimeout:
                            # No new tab opened — fallback to same-page navigation
                            print("  ! No new tab opened (clicked but no window.open fired)")
                            await asyncio.sleep(2)
                            clicked = True
            except Exception as e:
                print(f"  ! Error targeting '{target_company}' card: {e}")

            # Fallback: click the first visible Apply button on the page
            if not clicked:
                print(f"  ! Could not find '{target_company}' — trying first Apply button")
                for sel in [
                    "button:has-text('Apply →')",
                    "button:has-text('Apply')",
                    "a:has-text('Apply')",
                ]:
                    try:
                        btn = page.locator(sel).first
                        if await btn.is_visible(timeout=1500):
                            print(f"  → Falling back to first visible: {sel}")
                            try:
                                async with context.expect_page(timeout=5000) as new_page_info:
                                    await btn.click()
                                new_page = await new_page_info.value
                                await new_page.wait_for_load_state(
                                    "domcontentloaded", timeout=20000
                                )
                                page = new_page
                            except PWTimeout:
                                await asyncio.sleep(2)
                            clicked = True
                            break
                    except Exception:
                        continue

        if not clicked:
            print("  ! Could not find Apply button. Taking screenshot...")
            no_apply = _ROOT / "no_apply_btn.png"
            await page.screenshot(path=str(no_apply))
            await browser.close()
            return

        await asyncio.sleep(3)
        current_url = page.url
        print(f"\n[4] Redirected to: {current_url}")

        # ── Step 3: Detect ATS and fill the form ──
        page_content = await page.content()

        if "workday" in current_url or "myworkdayjobs" in current_url or "workday" in page_content.lower():
            print("\n[5] Detected: Workday ATS")
            success = await handle_workday_page(page)
        elif "greenhouse" in current_url or "boards.greenhouse" in current_url:
            print("\n[5] Detected: Greenhouse ATS")
            success = await handle_generic_form(page)
        elif "lever.co" in current_url:
            print("\n[5] Detected: Lever ATS")
            success = await handle_generic_form(page)
        else:
            print("\n[5] Unknown ATS — attempting generic fill")
            success = await handle_generic_form(page)

        if success:
            print("\n✅ DONE — Application processed!")
        else:
            print("\n⚠ Agent completed but could not confirm submission.")
            print("  Check the browser window and complete manually if needed.")

        # Keep browser open briefly so you can inspect the result.
        # Override with AGENT_REVIEW_SECONDS=N (e.g. 0 for instant close).
        review_secs = int(os.environ.get("AGENT_REVIEW_SECONDS", "15"))
        if review_secs > 0:
            print(f"\nKeeping browser open for {review_secs} seconds so you can review...")
            await asyncio.sleep(review_secs)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
