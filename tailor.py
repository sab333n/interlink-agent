"""
Job-specific resume rewrite and cover letter, powered by the same Groq model as chat.

Résumés always use Jake's Resume layout (the classic single-column Overleaf ATS template):
  Name
  contact | line
  Education → Experience → Projects → Technical Skills
No summary block, no multi-column layout.

POST /api/tailor  →  { resume_text, cover_letter, highlights[], template: "jakes" }
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import chat
from auth import require_user

router = APIRouter(prefix="/api/tailor")

# Jake's Resume (Overleaf) section order and contact style.
RESUME_PROMPT = """You rewrite one applicant's résumé for ONE specific job.

Rules:
- Use ONLY facts in PROFILE. Never invent employers, titles, degrees, dates, projects, or metrics.
- Keep every real employer and title. You may reorder bullets and rephrase for the job's keywords.
- Prefer quantified bullets from PROFILE. If a metric isn't in PROFILE, don't add one.
- Do NOT write a Summary / Objective section.
- Return JSON only (no markdown fences):
{
  "name": "Full Name",
  "contact": ["email", "phone", "City, ST", "linkedin url or handle", "github url or handle"],
  "education": [{"school": "", "location": "", "degree": "", "dates": "", "details": ["optional coursework line"]}],
  "experience": [{"title": "", "company": "", "location": "", "dates": "", "bullets": ["...", "..."]}],
  "projects": [{"name": "", "stack": "", "dates": "", "bullets": ["..."]}],
  "skills": {"Languages": "...", "Frameworks": "...", "Tools": "..."},
  "highlights": ["3 short reasons this rewrite fits the job"]
}
- contact: omit empty values. Prefer 3-5 items.
- experience: 1-4 roles, 3-5 bullets each, tailored to the job.
- projects: 0-3 real projects from PROFILE only (omit the array if none).
- skills: 2-4 category keys; put job-relevant skills first inside each value.
- education details may include one "Relevant Coursework: ..." line when PROFILE has majors/skills."""

COVER_PROMPT = """You write a cover letter for one applicant and one job.

Rules:
- Use ONLY facts in PROFILE. Never invent experience.
- 3 short paragraphs, first person, under 220 words.
- Paragraph 1: why this company/role (specific).
- Paragraph 2: one proof point from real experience (numbers only if in PROFILE).
- Paragraph 3: ask for a conversation; no fluff, no exclamation marks.
- No greeting line beyond "Dear Hiring Team," and no mailing address block.
Return JSON: {"cover_letter": "..."}"""


def format_jakes(data: dict[str, Any]) -> str:
    """Render structured résumé data into Jake's Resume plain-text layout."""
    lines: list[str] = []
    name = str(data.get("name") or "").strip() or "Applicant"
    lines.append(name)

    contact = [str(c).strip() for c in (data.get("contact") or []) if str(c).strip()]
    # Strip URL schemes for a cleaner Jake-style contact line.
    cleaned = []
    for c in contact:
        c = re.sub(r"^https?://(www\.)?", "", c).rstrip("/")
        if c and c not in cleaned:
            cleaned.append(c)
    if cleaned:
        lines.append(" | ".join(cleaned[:6]))
    lines.append("")

    edu = data.get("education") or []
    if edu:
        lines.append("Education")
        lines.append("")
        for e in edu[:3]:
            school = str(e.get("school") or "").strip()
            loc = str(e.get("location") or "").strip()
            degree = str(e.get("degree") or "").strip()
            dates = str(e.get("dates") or "").strip()
            head = school
            if loc:
                head = f"{school}, {loc}" if school else loc
            if dates:
                lines.append(f"{head}{' ' * max(2, 66 - len(head) - len(dates))}{dates}" if head else dates)
            elif head:
                lines.append(head)
            if degree:
                lines.append(degree)
            for d in e.get("details") or []:
                t = str(d).strip()
                if t:
                    lines.append(t if t.startswith(("•", "-", "*")) else f"• {t}")
            lines.append("")

    exp = data.get("experience") or []
    if exp:
        lines.append("Experience")
        lines.append("")
        for e in exp[:5]:
            title = str(e.get("title") or "").strip()
            company = str(e.get("company") or "").strip()
            loc = str(e.get("location") or "").strip()
            dates = str(e.get("dates") or "").strip()
            if title and dates:
                lines.append(f"{title}{' ' * max(2, 66 - len(title) - len(dates))}{dates}")
            elif title:
                lines.append(title)
            company_line = company
            if loc:
                company_line = f"{company} — {loc}" if company else loc
            if company_line:
                lines.append(company_line)
            for b in e.get("bullets") or []:
                t = str(b).strip().lstrip("•-* ").strip()
                if t:
                    lines.append(f"• {t}")
            lines.append("")

    projects = data.get("projects") or []
    if projects:
        lines.append("Projects")
        lines.append("")
        for p in projects[:4]:
            pname = str(p.get("name") or "").strip()
            stack = str(p.get("stack") or "").strip()
            dates = str(p.get("dates") or "").strip()
            left = f"{pname} | {stack}" if pname and stack else (pname or stack)
            if left and dates:
                lines.append(f"{left}{' ' * max(2, 66 - len(left) - len(dates))}{dates}")
            elif left:
                lines.append(left)
            for b in p.get("bullets") or []:
                t = str(b).strip().lstrip("•-* ").strip()
                if t:
                    lines.append(f"• {t}")
            lines.append("")

    skills = data.get("skills") or {}
    if isinstance(skills, dict) and skills:
        lines.append("Technical Skills")
        lines.append("")
        for key, val in skills.items():
            v = str(val).strip()
            k = str(key).strip()
            if k and v:
                lines.append(f"{k}: {v}")
        lines.append("")
    elif isinstance(skills, str) and skills.strip():
        lines.append("Technical Skills")
        lines.append("")
        lines.append(skills.strip())
        lines.append("")

    text = "\n".join(lines).strip() + "\n"
    # Collapse accidental triple blank lines.
    return re.sub(r"\n{3,}", "\n\n", text)


def looks_like_jakes(text: str) -> bool:
    t = text.lower()
    has_edu = "education" in t
    has_exp = "experience" in t
    has_skills = "technical skills" in t or "skills" in t
    no_summary = not re.search(r"(?m)^(summary|objective)\b", text, re.I)
    return has_edu and has_exp and has_skills and no_summary


class TailorRequest(BaseModel):
    job: dict[str, Any]
    profile: dict[str, Any]
    want_resume: bool = True
    want_cover: bool = True


async def _complete(system: str, user: dict[str, Any], cfg: dict[str, str]) -> dict[str, Any]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=cfg["key"], base_url=cfg["base"])
    res = await client.chat.completions.create(
        model=cfg["model"], temperature=0.3, max_tokens=4000,
        response_format={"type": "json_object"},
        extra_body={"reasoning_effort": "low"} if "gpt-oss" in cfg["model"] else None,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)[:18000]},
        ],
    )
    return json.loads(res.choices[0].message.content or "{}")


def _facts(profile: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in profile.items() if v not in (None, "", [])}


def _fallback_jakes(profile: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    """Deterministic Jake's skeleton if the model returns unusable JSON."""
    name = f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip() or "Applicant"
    contact = [c for c in [
        profile.get("email"), profile.get("phone"),
        ", ".join(x for x in [profile.get("city"), profile.get("state_abbr") or profile.get("state")] if x),
        profile.get("linkedin"), profile.get("github"),
    ] if c]
    edu = []
    if profile.get("university"):
        degree = " ".join(x for x in [
            profile.get("degree_full") or profile.get("degree"),
            f"in {profile['major']}" if profile.get("major") else "",
        ] if x).strip()
        edu.append({
            "school": profile.get("university"),
            "location": "",
            "degree": degree,
            "dates": str(profile.get("graduation_year") or ""),
            "details": [],
        })
    experience = []
    for e in (profile.get("work_experience") or [])[:4]:
        bullets = [b.strip() for b in re.split(r"[\n•]", str(e.get("description") or "")) if b.strip()][:5]
        dates = " – ".join(x for x in [
            "/".join(x for x in [e.get("from_month"), e.get("from_year")] if x),
            "Present" if e.get("currently_work_here") else "/".join(x for x in [e.get("to_month"), e.get("to_year")] if x),
        ] if x)
        experience.append({
            "title": e.get("job_title") or "",
            "company": e.get("company") or "",
            "location": e.get("location") or "",
            "dates": dates,
            "bullets": bullets or ["Contributed to team projects and shipped production work."],
        })
    tags = profile.get("skills_tags") or []
    if isinstance(tags, list):
        skill_line = ", ".join(str(t) for t in tags[:16])
    else:
        skill_line = str(profile.get("skills_summary") or "")
    return {
        "name": name,
        "contact": contact,
        "education": edu,
        "experience": experience,
        "projects": [],
        "skills": {"Languages": skill_line} if skill_line else {},
        "highlights": [f"Formatted with Jake's Resume template for {job.get('role')} at {job.get('company')}"],
    }


@router.post("")
async def tailor(req: TailorRequest, _: str = Depends(require_user)) -> dict[str, Any]:
    cfg = chat.llm_config()
    if not cfg:
        raise HTTPException(503, "No model API key configured on the agent server.")
    company = str(req.job.get("company") or "the company")
    role = str(req.job.get("role") or "the role")
    job_block = {
        "company": company,
        "role": role,
        "location": req.job.get("location"),
        "tags": req.job.get("tags") or [],
        "description": str(req.job.get("description") or "")[:2000],
    }
    profile = _facts(req.profile)
    out: dict[str, Any] = {
        "company": company,
        "role": role,
        "template": "jakes",
        "resume_text": "",
        "cover_letter": "",
        "highlights": [],
    }
    try:
        if req.want_resume:
            data = await _complete(RESUME_PROMPT, {"JOB": job_block, "PROFILE": profile}, cfg)
            # Prefer structured → Jake's formatter. Fall back if the model returned prose only.
            if data.get("experience") or data.get("education") or data.get("name"):
                text = format_jakes(data)
                highlights = [str(h) for h in (data.get("highlights") or [])[:5]]
            elif data.get("resume_text") and looks_like_jakes(str(data["resume_text"])):
                text = str(data["resume_text"]).strip() + "\n"
                highlights = [str(h) for h in (data.get("highlights") or [])[:5]]
            else:
                fb = _fallback_jakes(profile, job_block)
                # Still try to merge any bullets the model put in resume_text highlights.
                text = format_jakes(fb)
                highlights = fb["highlights"]
            if not looks_like_jakes(text):
                fb = _fallback_jakes(profile, job_block)
                text = format_jakes(fb)
                highlights = fb["highlights"]
            out["resume_text"] = text
            out["highlights"] = highlights or [
                "Jake's Resume layout (Education → Experience → Projects → Technical Skills)"
            ]
        if req.want_cover:
            data = await _complete(COVER_PROMPT, {"JOB": job_block, "PROFILE": profile}, cfg)
            letter = str(data.get("cover_letter") or "").strip()
            if letter and not letter.lower().startswith("dear"):
                letter = f"Dear Hiring Team,\n\n{letter}"
            out["cover_letter"] = letter
    except Exception as e:
        raise HTTPException(502, f"Tailor failed: {str(e)[:200]}") from None
    if not out["resume_text"] and not out["cover_letter"]:
        raise HTTPException(502, "The model returned an empty rewrite. Try again.")
    return out


class CoverOnlyRequest(BaseModel):
    job: dict[str, Any]
    profile: dict[str, Any]


@router.post("/cover")
async def cover_only(req: CoverOnlyRequest, user: str = Depends(require_user)) -> dict[str, Any]:
    return await tailor(TailorRequest(job=req.job, profile=req.profile, want_resume=False, want_cover=True), user)
