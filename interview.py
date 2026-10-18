"""
Interview coach — real Groq/OpenAI-compatible coaching, not a static page only.
Also lists the free question bank for the frontend.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth import require_user
from chat import llm_config

router = APIRouter(prefix="/api/interview")
BANK = [
    {"id": "iq-1", "company": "Google", "category": "Behavioral", "level": "Intern",
     "question": "Tell me about a time you disagreed with a teammate. What did you do?",
     "tip": "Situation → Action → Result. End with what you’d do differently."},
    {"id": "iq-2", "company": "Meta", "category": "Coding", "level": "New grad",
     "question": "Given a binary tree, return its level-order traversal.",
     "tip": "BFS with a queue. State time/space before coding."},
    {"id": "iq-3", "company": "Amazon", "category": "Behavioral", "level": "Intern",
     "question": "Describe a time you had to deliver with incomplete information.",
     "tip": "Map to Ownership / Bias for Action."},
    {"id": "iq-4", "company": "Apple", "category": "System design", "level": "Mid",
     "question": "Design a URL shortener that handles 100M redirects/day.",
     "tip": "Clarify read/write ratio, ID generation, caching, collisions."},
    {"id": "iq-5", "company": "Stripe", "category": "Domain", "level": "New grad",
     "question": "How would you design an idempotent payment API?",
     "tip": "Idempotency keys, retry semantics, exactly-once vs at-least-once."},
    {"id": "iq-6", "company": "Notion", "category": "Coding", "level": "Intern",
     "question": "Implement a LRU cache with O(1) get and put.",
     "tip": "Hash map + doubly linked list."},
    {"id": "iq-7", "company": "Lyft", "category": "Behavioral", "level": "Intern",
     "question": "Tell me about a project where you used data to change a decision.",
     "tip": "Name the metric, the insight, and the change."},
    {"id": "iq-8", "company": "Adobe", "category": "Domain", "level": "Intern",
     "question": "How would you evaluate an LLM feature before shipping?",
     "tip": "Offline metrics, human review, online A/B, failure modes."},
]


class CoachRequest(BaseModel):
    question: str
    answer: str
    company: str = ""
    category: str = ""
    profile: dict[str, Any] = Field(default_factory=dict)


@router.get("/bank")
def bank(_user_id: str = Depends(require_user)):
    return {"ok": True, "count": len(BANK), "questions": BANK}


@router.post("/coach")
async def coach(req: CoachRequest, _user_id: str = Depends(require_user)):
    cfg = llm_config()
    if not cfg:
        raise HTTPException(503, "No LLM API key configured (GROQ_API_KEY / OPENAI_API_KEY / XAI_API_KEY).")
    if not req.question.strip() or not req.answer.strip():
        raise HTTPException(400, "question and answer are required")

    name = (req.profile.get("preferredName") or req.profile.get("fullName") or "candidate").split()[0]
    system = (
        "You are a tough but fair interview coach. Score the answer 1-10, give 3 concrete "
        "improvements, and a stronger rewrite under 120 words. JSON only: "
        '{"score":n,"verdict":"...","improvements":["..."],"stronger_answer":"..."}'
    )
    user = (
        f"Candidate: {name}\nCompany context: {req.company or 'general'}\n"
        f"Category: {req.category or 'general'}\n\n"
        f"Question: {req.question}\n\nAnswer:\n{req.answer}"
    )

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=cfg["key"], base_url=cfg["base"])
        resp = await client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.4,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
    except Exception as e:
        raise HTTPException(502, f"Coach model failed: {e}") from e

    return {
        "ok": True,
        "score": data.get("score"),
        "verdict": data.get("verdict"),
        "improvements": data.get("improvements") or [],
        "stronger_answer": data.get("stronger_answer") or "",
    }
