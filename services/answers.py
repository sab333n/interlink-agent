"""
Answers the form questions the rule-based filler leaves blank.

The model sees the applicant's profile facts and every answer they've given on
past applications (Supabase `apply_answers`), and returns per-field answers
tagged by source: "fact" (from the profile), "past" (reused from an earlier
application) or "draft" (written for an open-ended question; always shown to
the user for review). Goal: leave as few blanks as possible — essays included.
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Optional

import routes.chat as chat
from agent import ASK_USER  # questions answered only from past answers or profile facts
from services.supa import admin_config, db

MAX_PAST = 80

PROMPT = """You fill out EVERY blank field on a job application for one applicant.

Use ONLY the FACTS and PAST ANSWERS below. Never invent employers, titles, degrees, dates, or legal status.

For each question return {"id", "answer", "source"}:
- "fact": follows directly from FACTS (sponsorship, work auth, school, graduation, location, links, skills).
- "past": reuse a past answer (swap the company name when needed).
- "draft": open-ended / essay / "why us" / "tell us about" / cover letter. Write 3-5 specific
  first-person sentences grounded in FACTS (skills, projects, school). No fluff, no exclamation marks,
  no greeting or sign-off. ALWAYS draft these — never leave them blank. Each essay must answer its
  own question; never give two questions the same text. Mention only experiences in FACTS — no
  invented upbringing, languages or anecdotes.
- Omit GPA, name pronunciation, and accounts/usernames on the company's own product unless FACTS or
  PAST ANSWERS contain them. The applicant fills those in themselves.
- For OPTIONS / checkboxes / radios: copy the option text EXACTLY. For multi-select checkboxes,
  return a comma-separated list of the options to check.
- Prefer answering over leaving blank. Only return null for questions you truly cannot answer
  (e.g. salary history with no data, a specific employee referral name you don't have).
- Required fields must be answered whenever a reasonable choice exists in OPTIONS or FACTS.
- Internship term / availability: pick every term that fits (usually Summer and/or Winter/Spring
  around their graduation year). Prefer Summer if unsure.
- Area of interest: pick the option closest to their skills (Backend/Systems for Python/Go,
  Frontend for React, Full-stack if both, Security/Infra if relevant).
- "How did you hear": Online job board / Simplify / Career website / Other — never invent a person.
- EEO / gender / race / veteran: pick Decline / Prefer not to say when available.
- Consent / privacy / acknowledge: Yes / Agree / Acknowledge.

Return JSON: {"answers": [...]} — one entry per question you can answer."""

ESSAY_PROMPT = """Write the applicant's answer to ONE open-ended question on a job application.

3-5 specific first-person sentences grounded only in FACTS (skills, projects, experience, school).
Answer exactly what QUESTION asks, and don't reuse sentences from OTHER ANSWERS.
Never invent employers, titles, dates, results, languages, upbringing or personal anecdotes:
mention only experiences that appear in FACTS. No greeting, sign-off or exclamation marks.

Return JSON: {"answer": "..."}"""

PLACEHOLDER = re.compile(r"^\s*(n/?a|none|not (specified|applicable|provided)|-+|tbd|unknown)\s*$", re.I)
NOT_ESSAY = re.compile(r"if different|preferred name|pronoun|salary|compensation|referral name|employee name", re.I)


def _is_essay(field: dict[str, Any]) -> bool:
    label = field.get("label") or ""
    if field.get("options") or NOT_ESSAY.search(label) or ASK_USER.search(label):
        return False
    return field["kind"] == "textarea" or (field["kind"] in ("text", "input") and (len(label) > 80 or "?" in label))


def question_key(label: str, company: str) -> str:
    key = re.sub(r"\s*[*✱]\s*$", "", label or "").strip().lower()
    if company:
        key = re.sub(re.escape(company.lower()), "{company}", key)
    return re.sub(r"\s+", " ", key)[:300]


async def load_past(user_id: str) -> list[dict[str, str]]:
    if not admin_config() or not re.fullmatch(r"[0-9a-f-]{36}", user_id or ""):
        return []
    try:
        rows = await db("GET", f"apply_answers?select=question,answer&user_id=eq.{user_id}"
                               f"&order=updated_at.desc&limit={MAX_PAST}")
    except RuntimeError:
        return []
    return rows or []


async def remember(user_id: str, company: str, pairs: list[tuple[str, str]]) -> None:
    """Saves the user's answers to non-standard questions so the next form can reuse them."""
    if not pairs or not admin_config() or not re.fullmatch(r"[0-9a-f-]{36}", user_id or ""):
        return
    rows = [{"user_id": user_id, "question": question_key(q, company), "answer": a[:2000]}
            for q, a in pairs if q.strip() and a.strip()]
    try:
        await db("POST", "apply_answers?on_conflict=user_id,question", rows,
                 prefer="resolution=merge-duplicates")
    except RuntimeError as e:
        print(f"[answers] couldn't save answers: {e}", flush=True)


def _match_options(options: list[str], text: str) -> Optional[str]:
    """Exact, then startswith, then contains — returns the option's exact wording."""
    want = text.strip().lower()
    if not want or not options:
        return None
    for test in (
        lambda o: o.strip().lower() == want,
        lambda o: o.strip().lower().startswith(want),
        lambda o: want.startswith(o.strip().lower()) and len(o.strip()) > 2,
        lambda o: want in o.lower(),
    ):
        hits = [o for o in options if test(o)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            # Prefer shortest exact-ish match.
            return min(hits, key=lambda o: abs(len(o) - len(text)))
    return None


def _clean(field: dict[str, Any], answer: Any) -> Optional[str]:
    if answer is None:
        return None
    if isinstance(answer, list):
        answer = ", ".join(str(a) for a in answer if a)
    text = str(answer).strip()
    if not text:
        return None
    if field["kind"] == "checkbox":
        return "true" if text.lower().startswith(("y", "t", "check", "on")) else ""
    options = field.get("options") or []
    if field["kind"] == "checkboxes" and options:
        parts = re.split(r"\s*,\s*", text)
        picked = []
        for part in parts:
            match = _match_options(options, part)
            if match and match not in picked:
                picked.append(match)
        return ", ".join(picked) if picked else None
    if options:
        match = _match_options(options, text)
        if match:
            return match
        if field["kind"] == "combobox" and len(options) > 8:
            return text
        return None
    return text


def _skills(profile: dict[str, Any]) -> str:
    tags = profile.get("skills_tags") or []
    if isinstance(tags, list):
        return " ".join(str(t) for t in tags).lower()
    return str(profile.get("skills_summary") or "").lower()


def _pick_yes_no(opts: list[str], yn: str) -> Optional[str]:
    hit = next((o for o in opts if re.match(rf"^\s*{yn}\b", o, re.I)), None)
    if hit:
        return hit
    return next((o for o in opts if re.search(rf"\b{yn}\b", o, re.I)), None)


def _yn_from_profile(q: str, profile: dict[str, Any]) -> Optional[str]:
    visa = str(profile.get("visa_status", "")).lower()
    if re.search(r"sponsor|h-?1b|immigration", q):
        return "yes" if str(profile.get("sponsorship", "")).lower().startswith("y") else "no"
    if re.search(r"authori[sz]ed|lawfully|eligible to work|legally (authori|able)|right to work", q):
        auth = str(profile.get("work_auth", "yes")).lower()
        return "no" if auth in ("no", "n") or "not authorized" in auth else "yes"
    if re.search(r"willing to work|work from the office|in.?office|on.?site|hybrid|relocat", q):
        rel = str(profile.get("relocation", "yes")).lower()
        return "no" if rel.startswith("n") else "yes"
    if re.search(r"optional practical|stem\s*opt|\bopt\b|\bcpt\b|f-1", q):
        if re.search(r"f-1|cpt|opt|stem", visa):
            return "yes"
        if re.search(r"citizen|permanent|green card", visa):
            return "no"
        # Unknown visa — default No so the form isn't blocked; user can edit.
        return "no"
    return None


def _fallback(field: dict[str, Any], job: dict[str, Any], profile: dict[str, Any]) -> Optional[dict[str, str]]:
    """Rule-based last pass so required / common blanks don't stay empty."""
    label = field.get("label") or ""
    q = label.lower()
    opts = field.get("options") or []
    kind = field["kind"]
    skills = _skills(profile)
    grad = str(profile.get("graduation_year") or "")
    loc = str(job.get("location") or "")

    # Short profile fields (Greenhouse often leaves these blank until suggest).
    if kind in ("text", "input", "email", "tel") and not opts:
        table = [
            (r"preferred (first )?name", profile.get("first_name")),
            (r"first name|given name", profile.get("first_name")),
            (r"last name|family name|surname", profile.get("last_name")),
            (r"e-?mail", profile.get("email")),
            (r"phone|mobile", profile.get("phone")),
            (r"linkedin", profile.get("linkedin")),
            (r"github", profile.get("github")),
            (r"website|portfolio|personal site", profile.get("portfolio") or profile.get("github")),
            (r"undergraduate gpa|college gpa|\bgpa\b", profile.get("gpa")),
            (r"end date year|graduation year|to year", grad),
        ]
        for pattern, value in table:
            if value and not PLACEHOLDER.match(str(value)) and re.search(pattern, q):
                return {"answer": str(value), "source": "fact"}

    if kind == "checkbox" and re.search(r"agree|acknowledge|certify|confirm|consent|privacy", q):
        return {"answer": "true", "source": "fact"}

    if opts and re.search(r"gender|sex\b|race|ethnic|hispanic|latino|veteran|disabilit|orientation|pronoun", q):
        decline = next((o for o in opts if re.search(
            r"decline|prefer not|don.?t wish|do not wish|not to (say|answer|disclose)|choose not|i don't wish", o, re.I)), None)
        if decline:
            return {"answer": decline, "source": "fact"}

    if opts and re.search(r"how (did|do) you (hear|heard|find|learn)|where did you hear|source|referr", q):
        for pattern in (r"online job board", r"simplify", r"linkedin", r"career (fair|website)", r"job board",
                        r"internet|website|other"):
            hit = next((o for o in opts if re.search(pattern, o, re.I)), None)
            if hit:
                return {"answer": hit, "source": "fact"}

    # Graduation timeline / expected graduation (term + year style options).
    if opts and re.search(r"graduation timeline|expected graduation|graduat.*(term|date|when|timeline)", q):
        year = grad if grad.isdigit() else ""
        for pattern in (
            rf"summer\s*{year}" if year else None,
            rf"{year}" if year else None,
            r"summer\s*2027|summer\s*2026|spring\s*2027|fall\s*2026",
            r"summer",
            r"spring|winter|fall",
        ):
            if not pattern:
                continue
            hit = next((o for o in opts if re.search(pattern, o, re.I)), None)
            if hit:
                return {"answer": hit, "source": "fact"}
        if opts:
            return {"answer": opts[0], "source": "fact"}

    if opts and re.search(r"term|available to intern|which (term|season)|internship period", q):
        picks = [o for o in opts if re.search(r"summer", o, re.I)]
        if not picks:
            picks = [o for o in opts if re.search(r"winter|spring|fall|autumn", o, re.I)]
        if not picks and opts:
            picks = opts[:1]
        if picks:
            return {"answer": ", ".join(picks) if kind == "checkboxes" else picks[0], "source": "fact"}

    # Office location preference (Pittsburgh / NYC / Remote…).
    if opts and re.search(r"office location|which (office|location|city)|most interested in", q):
        loc_l = loc.lower()
        for city in ("pittsburgh", "new york", "nyc", "san francisco", "seattle", "austin", "remote"):
            if city in loc_l:
                hit = next((o for o in opts if re.search(re.escape(city), o, re.I)), None)
                if hit:
                    return {"answer": hit, "source": "fact"}
        hit = next((o for o in opts if re.search(r"pittsburgh|new york|nyc", o, re.I)), None)
        if hit:
            return {"answer": hit, "source": "fact"}
        if opts:
            return {"answer": opts[0], "source": "fact"}

    if opts and re.search(r"end date month|start date month|from month|to month|\bmonth\b", q):
        # Education end month — May is the common US undergrad default.
        for m in ("may", "june", "december", "may"):
            hit = next((o for o in opts if re.search(rf"^{m}$|{m}", o, re.I)), None)
            if hit:
                return {"answer": hit, "source": "fact"}
        if opts:
            return {"answer": opts[min(4, len(opts) - 1)], "source": "fact"}  # often May is index ~4

    if opts and re.search(r"area of interest|focus area|team interest|which team|engineering track", q):
        prefer = []
        if re.search(r"python|go\b|golang|rust|c\+\+|backend|postgres|kubernetes|infra|data|ml|stat", skills):
            prefer += [r"data|machine learning|ml|ai\b|analytics", r"backend|systems|infrastructure|platform"]
        if re.search(r"react|typescript|frontend|css|ui\b", skills):
            prefer += [r"frontend|full.?stack|web"]
        prefer += [r"full.?stack|software|general|data"]
        for pattern in prefer:
            hit = next((o for o in opts if re.search(pattern, o, re.I)), None)
            if hit:
                return {"answer": hit, "source": "fact"}

    if opts and re.search(r"available.*(start|full.?time)|when would you be available|start date", q):
        for pattern in (r"upon graduation|after graduation|need to return to school",
                        r"graduat", r"flexible|immediately|asap"):
            hit = next((o for o in opts if re.search(pattern, o, re.I)), None)
            if hit:
                return {"answer": hit, "source": "fact"}
        if grad:
            hit = next((o for o in opts if grad in o), None)
            if hit:
                return {"answer": hit, "source": "fact"}

    # Yes/No style selects (work auth, sponsorship, OPT, office willingness).
    yn = _yn_from_profile(q, profile)
    if yn and (opts or kind in ("select", "combobox", "radio")):
        if opts:
            hit = _pick_yes_no(opts, yn)
            if hit:
                return {"answer": hit, "source": "fact"}
        else:
            return {"answer": yn.capitalize(), "source": "fact"}

    if opts and re.search(r"degree|bachelor|master", q):
        degree = str(profile.get("degree_full") or profile.get("degree") or "")
        if degree:
            hit = _match_options(opts, degree) or _match_options(opts, "Bachelor")
            if hit:
                return {"answer": hit, "source": "fact"}

    # Essays are drafted by the model, one per question (see _draft_essays); no canned text.
    return None


async def _draft_essays(fields: list[dict[str, Any]], job: dict[str, Any], facts: dict[str, Any],
                        cfg: dict[str, str], out: dict[str, dict[str, str]]) -> None:
    """Drafts each still-blank essay with its own model call, so every answer fits its question."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=cfg["key"], base_url=cfg["base"])
    for f in fields:
        others = [v["answer"] for v in out.values() if v["source"] == "draft"]
        try:
            res = await client.chat.completions.create(
                model=cfg["model"], temperature=0.4, max_tokens=1500,
                response_format={"type": "json_object"},
                extra_body={"reasoning_effort": "low"} if "gpt-oss" in cfg["model"] else None,
                messages=[
                    {"role": "system", "content": ESSAY_PROMPT},
                    {"role": "user", "content": json.dumps({
                        "company": job.get("company"), "role": job.get("role"), "QUESTION": f["label"][:600],
                        "FACTS": facts, "OTHER ANSWERS": others,
                    }, ensure_ascii=False)[:20000]},
                ])
            answer = str(json.loads(res.choices[0].message.content or "{}").get("answer") or "").strip()
        except Exception as e:
            print(f"[answers] essay draft failed: {str(e)[:200]}", flush=True)
            continue
        if answer and answer not in others:
            out[f["id"]] = {"answer": answer, "source": "draft"}


async def suggest(fields: list[dict[str, Any]], job: dict[str, Any], profile: dict[str, Any],
                  user_id: str) -> dict[str, dict[str, str]]:
    """{field id: {"answer", "source"}} for blank fields the model / heuristics could answer."""
    cfg = chat.llm_config()
    blank = [f for f in fields if f["kind"] != "file" and not str(f.get("value") or "").strip()]
    if not blank:
        return {}
    past = await load_past(user_id)
    out: dict[str, dict[str, str]] = {}
    by_id = {f["id"]: f for f in blank}

    if cfg:
        questions = [{"id": f["id"], "question": f["label"][:400], "kind": f["kind"], "required": f["required"],
                      **({"options": f["options"][:80]} if f.get("options") else {})} for f in blank]
        facts = {k: v for k, v in profile.items() if v not in (None, "", [])}
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=cfg["key"], base_url=cfg["base"])
        try:
            res = await client.chat.completions.create(
                model=cfg["model"], temperature=0.25, max_tokens=5000,
                response_format={"type": "json_object"},
                extra_body={"reasoning_effort": "low"} if "gpt-oss" in cfg["model"] else None,
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": json.dumps({
                        "company": job.get("company"), "role": job.get("role"),
                        "FACTS": facts, "PAST ANSWERS": past, "QUESTIONS": questions,
                    }, ensure_ascii=False)[:24000]},
                ])
            data = json.loads(res.choices[0].message.content or "{}")
            for item in data.get("answers") or []:
                f = by_id.get(str(item.get("id")))
                source = item.get("source")
                if not f or source not in ("fact", "past", "draft"):
                    continue
                if ASK_USER.search(f["label"]) and source != "past":
                    continue
                answer = _clean(f, item.get("answer"))
                if answer is not None:
                    out[f["id"]] = {"answer": answer, "source": source}
        except Exception as e:
            print(f"[answers] suggest failed: {str(e)[:200]}", flush=True)

        # Essays with no answer or a copy of another essay get their own draft.
        essays = [f for f in blank if _is_essay(f)]
        seen: set[str] = set()
        for f in essays:
            hint = out.get(f["id"])
            if hint and hint["source"] != "past" and hint["answer"] in seen:
                out.pop(f["id"])
            elif hint:
                seen.add(hint["answer"])
        await _draft_essays([f for f in essays if f["id"] not in out], job, facts, cfg, out)

    for f in blank:
        if f["id"] in out:
            continue
        hint = _fallback(f, job, profile)
        if not hint:
            continue
        answer = _clean(f, hint["answer"])
        if answer is not None:
            out[f["id"]] = {"answer": answer, "source": hint["source"]}
    return out
