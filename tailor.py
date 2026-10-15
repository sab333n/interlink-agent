"""
Job-specific resume rewrite and cover letter, powered by the same Groq model as chat.

POST /api/tailor  →  { resume_text, cover_letter, highlights[] }
The rewrite only uses the applicant's real experience; it never invents employers or metrics.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import chat
from auth import require_user

router = APIRouter(prefix="/api/tailor")

RESUME_PROMPT = """You rewrite one applicant's résumé for ONE specific job.

Rules:
- Use ONLY facts in PROFILE. Never invent employers, titles, degrees, dates, or metrics.
- Keep every real employer and title. You may reorder bullets and rephrase for the job's keywords.
- Prefer quantified bullets from PROFILE. If a metric isn't in PROFILE, don't add one.
- Output plain text (no markdown fences) with these sections in order:
  NAME / CONTACT
  SUMMARY (3-4 sentences aimed at this role)
  SKILLS (comma-separated, job-relevant first)
  EXPERIENCE (each role: Title — Company (dates) then 3-5 "- " bullets)
  EDUCATION
- Keep the whole résumé under 450 words.
Return JSON: {"resume_text": "...", "highlights": ["3 short reasons this rewrite fits the job"]}"""

COVER_PROMPT = """You write a cover letter for one applicant and one job.

Rules:
- Use ONLY facts in PROFILE. Never invent experience.
- 3 short paragraphs, first person, under 220 words.
- Paragraph 1: why this company/role (specific).
- Paragraph 2: one proof point from real experience (numbers only if in PROFILE).
- Paragraph 3: ask for a conversation; no fluff, no exclamation marks.
- No greeting line beyond "Dear Hiring Team," and no mailing address block.
Return JSON: {"cover_letter": "..."}"""


class TailorRequest(BaseModel):
    job: dict[str, Any]
    profile: dict[str, Any]
    want_resume: bool = True
    want_cover: bool = True


async def _complete(system: str, user: dict[str, Any], cfg: dict[str, str]) -> dict[str, Any]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=cfg["key"], base_url=cfg["base"])
    res = await client.chat.completions.create(
        model=cfg["model"], temperature=0.35, max_tokens=3500,
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
        "resume_text": "",
        "cover_letter": "",
        "highlights": [],
    }
    try:
        if req.want_resume:
            data = await _complete(RESUME_PROMPT, {"JOB": job_block, "PROFILE": profile}, cfg)
            text = str(data.get("resume_text") or "").strip()
            if text:
                out["resume_text"] = text
                out["highlights"] = [str(h) for h in (data.get("highlights") or [])[:5]]
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
