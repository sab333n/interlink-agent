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

import chat
from supa import admin_config, db

MAX_PAST = 80

PROMPT = """You fill out EVERY blank field on a job application for one applicant.

Use ONLY the FACTS and PAST ANSWERS below. Never invent employers, titles, degrees, dates, or legal status.

For each question return {"id", "answer", "source"}:
- "fact": follows directly from FACTS (sponsorship, work auth, school, graduation, location, links, skills).
- "past": reuse a past answer (swap the company name when needed).
- "draft": open-ended / essay / "why us" / "tell us about" / cover letter. Write 3-5 specific
  first-person sentences grounded in FACTS (skills, projects, school). No fluff, no exclamation marks,
  no greeting or sign-off. ALWAYS draft these — never leave them blank.
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


def _fallback(field: dict[str, Any], job: dict[str, Any], profile: dict[str, Any]) -> Optional[dict[str, str]]:
    """Rule-based last pass so required / common blanks don't stay empty."""
    label = field.get("label") or ""
    q = label.lower()
    opts = field.get("options") or []
    kind = field["kind"]
    skills = _skills(profile)
    grad = str(profile.get("graduation_year") or "")
    company = job.get("company") or "the company"
    role = job.get("role") or "this role"

    if kind == "checkbox" and re.search(r"agree|acknowledge|certify|confirm|consent|privacy", q):
        return {"answer": "true", "source": "fact"}

    if opts and re.search(r"gender|sex\b|race|ethnic|hispanic|latino|veteran|disabilit|orientation|pronoun", q):
        decline = next((o for o in opts if re.search(
            r"decline|prefer not|don.?t wish|do not wish|not to (say|answer|disclose)|choose not|i don't wish", o, re.I)), None)
        if decline:
            return {"answer": decline, "source": "fact"}

    if opts and re.search(r"how (did|do) you (hear|heard|find|learn)|source|referr", q):
        for pattern in (r"online job board", r"simplify", r"career (fair|website)", r"job board",
                        r"internet|website|other"):
            hit = next((o for o in opts if re.search(pattern, o, re.I)), None)
            if hit:
                return {"answer": hit, "source": "fact"}

    if opts and re.search(r"term|available to intern|which (term|season)|internship period", q):
        picks = [o for o in opts if re.search(r"summer", o, re.I)]
        if not picks:
            picks = [o for o in opts if re.search(r"winter|spring|fall|autumn", o, re.I)]
        if not picks and opts:
            picks = opts[:1]
        if picks:
            return {"answer": ", ".join(picks) if kind == "checkboxes" else picks[0], "source": "fact"}

    if opts and re.search(r"area of interest|focus area|team interest|which team|engineering track", q):
        prefer = []
        if re.search(r"python|go\b|golang|rust|c\+\+|backend|postgres|kubernetes|infra", skills):
            prefer += [r"backend|systems|infrastructure|platform|distributed"]
        if re.search(r"react|typescript|frontend|css|ui\b", skills):
            prefer += [r"frontend|full.?stack|web"]
        if re.search(r"security|crypto|network", skills):
            prefer += [r"security|network"]
        prefer += [r"full.?stack|software|general"]
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

    if opts and re.search(r"\b(yes|no)\b", " ".join(opts).lower()) and len(opts) <= 6:
        if re.search(r"sponsor|visa|immigration", q):
            yn = "yes" if str(profile.get("sponsorship", "")).lower().startswith("y") else "no"
        elif re.search(r"authori[sz]ed|eligible to work|legally", q):
            yn = "yes" if str(profile.get("work_auth", "yes")).lower() not in ("no", "n") else "no"
        elif re.search(r"relocat|in.?office|on.?site|hybrid|40 hour|full.?time|enrolled|return to", q):
            yn = "yes"
        elif re.search(r"\bcpt\b|\bopt\b|f-1", q):
            visa = str(profile.get("visa_status", "")).lower()
            yn = "yes" if re.search(r"f-1|cpt|opt", visa) else "no"
        else:
            yn = None
        if yn:
            hit = next((o for o in opts if re.match(rf"^\s*{yn}\b", o, re.I)), None)
            if hit:
                return {"answer": hit, "source": "fact"}

    if opts and re.search(r"degree|bachelor|master", q):
        degree = str(profile.get("degree_full") or profile.get("degree") or "")
        if degree:
            hit = _match_options(opts, degree) or _match_options(opts, "Bachelor")
            if hit:
                return {"answer": hit, "source": "fact"}

    if kind in ("textarea",) or (kind in ("text", "input") and (len(label) > 80 or "?" in label)):
        if re.search(r"if different|preferred name|pronoun|salary|compensation|referral name|employee name", q):
            return None
        exp = (profile.get("work_experience") or [{}])
        latest = exp[0] if isinstance(exp, list) and exp else {}
        proof = (latest.get("description") or profile.get("skills_summary") or "").strip()
        if not proof and not profile.get("university"):
            return None
        draft = (
            f"I'm excited about {role} at {company} because it lines up with the work I've already done. "
            f"{proof[:280].rstrip('.')}. "
            f"I'm currently studying {profile.get('major') or 'computer science'} at "
            f"{profile.get('university') or 'university'} "
            f"(class of {grad or 'soon'}) and want to ship real systems with your team."
        )
        return {"answer": re.sub(r"\s+", " ", draft).strip(), "source": "draft"}

    return None


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
                answer = _clean(f, item.get("answer"))
                if answer is not None:
                    out[f["id"]] = {"answer": answer, "source": source}
        except Exception as e:
            print(f"[answers] suggest failed: {str(e)[:200]}", flush=True)

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
