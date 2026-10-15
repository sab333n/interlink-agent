"""
Answers the form questions the rule-based filler leaves blank.

The model sees the applicant's profile facts and every answer they've given on
past applications (Supabase `apply_answers`), and returns per-field answers
tagged by source: "fact" (from the profile), "past" (reused from an earlier
application) or "draft" (written for an open-ended question; always shown to
the user for review). Anything it isn't sure about comes back unanswered.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

import chat
from supa import admin_config, db

MAX_PAST = 80

PROMPT = """You fill out job application questions for one applicant.

Use ONLY the FACTS and PAST ANSWERS below. Never invent experience, dates, or legal status.
For each question return an object {"id", "answer", "source"}:
- source "fact": the answer follows directly from FACTS (e.g. sponsorship, work authorization,
  graduation date, school, location, links).
- source "past": the applicant answered the same or an equivalent question before. Reuse that
  answer, adjusted only for a different company name.
- source "draft": an open-ended question (why this company, describe a project, etc.). Write
  2-4 specific first-person sentences grounded in FACTS. No fluff, no exclamation marks.
- answer null when you are not confident. Unsure is better than wrong.
For questions with OPTIONS, the answer must be copied exactly from OPTIONS.
For checkboxes, answer "yes" or "no".
Return JSON: {"answers": [...]}"""


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


def _clean(field: dict[str, Any], answer: Any) -> Optional[str]:
    if answer is None:
        return None
    text = str(answer).strip()
    if not text:
        return None
    if field["kind"] == "checkbox":
        return "true" if text.lower().startswith("y") else ""
    options = field.get("options") or []
    if options:
        match = next((o for o in options if o.strip().lower() == text.lower()), None)
        if match is None and field["kind"] == "combobox" and len(options) > 8:
            return text
        return match
    return text


async def suggest(fields: list[dict[str, Any]], job: dict[str, Any], profile: dict[str, Any],
                  user_id: str) -> dict[str, dict[str, str]]:
    """{field id: {"answer", "source"}} for blank fields the model could answer."""
    cfg = chat.llm_config()
    blank = [f for f in fields if f["kind"] != "file" and not str(f.get("value") or "").strip()]
    if not cfg or not blank:
        return {}
    past = await load_past(user_id)
    questions = [{"id": f["id"], "question": f["label"][:400], "kind": f["kind"], "required": f["required"],
                  **({"options": f["options"][:60]} if f.get("options") else {})} for f in blank]
    facts = {k: v for k, v in profile.items() if v not in (None, "", [])}
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=cfg["key"], base_url=cfg["base"])
    try:
        res = await client.chat.completions.create(
            model=cfg["model"], temperature=0.2, max_tokens=4000,
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
    except Exception as e:
        print(f"[answers] suggest failed: {str(e)[:200]}", flush=True)
        return {}
    by_id = {f["id"]: f for f in blank}
    out: dict[str, dict[str, str]] = {}
    for item in data.get("answers") or []:
        f = by_id.get(str(item.get("id")))
        source = item.get("source")
        if not f or source not in ("fact", "past", "draft"):
            continue
        answer = _clean(f, item.get("answer"))
        if answer is not None:
            out[f["id"]] = {"answer": answer, "source": source}
    return out
