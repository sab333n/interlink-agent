"""
Alumni / insider search via Google Programmable Search (CSE).
Runs on the agent so production (Vercel) works — Vite middleware is local-only.
"""
from __future__ import annotations

import os
import re
import urllib.parse
import urllib.request
import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from auth import require_user

router = APIRouter(prefix="/api/alumni")
STOP = {
    "of", "the", "at", "and", "for", "a", "an", "in", "on",
    "university", "college", "state",
}


def _cse_keys() -> tuple[str, str]:
    key = os.getenv("GOOGLE_SEARCH_API_KEY", "").strip()
    cx = os.getenv("GOOGLE_SEARCH_CX", "").strip()
    return key, cx


def _text_matches_university(text: str, school: str) -> bool:
    t = text.lower()
    s = school.strip().lower()
    if not s:
        return True
    if s in t:
        return True
    if re.search(r"\bulm\b", t) and re.search(r"ulm|louisiana|monroe", school, re.I):
        return True
    tokens = [w for w in re.split(r"\s+", s) if len(re.sub(r"[^a-z0-9]", "", w)) > 2 and w not in STOP]
    tokens = [re.sub(r"[^a-z0-9]", "", w) for w in tokens]
    tokens = [w for w in tokens if w and w not in STOP]
    if not tokens:
        return False
    need = 1 if len(tokens) <= 1 else min(2, len(tokens))
    return sum(1 for tok in tokens if tok in t) >= need


def _text_matches_company(text: str, company: str) -> bool:
    t = text.lower()
    c = company.strip().lower()
    if not c:
        return True
    if c in t:
        return True
    return bool(re.search(rf"\b{re.escape(c)}\b", text, re.I))


def _cse_search(query: str, key: str, cx: str, num: int = 8) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({
        "key": key,
        "cx": cx,
        "q": query,
        "num": str(min(10, max(1, num))),
    })
    url = f"https://www.googleapis.com/customsearch/v1?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "InterLink-agent/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode())
    return data.get("items") or []


def _person_from_item(item: dict[str, Any], connection: str) -> Optional[dict[str, Any]]:
    link = item.get("link") or ""
    if "linkedin.com/in/" not in link:
        return None
    title = item.get("title") or ""
    snippet = item.get("snippet") or ""
    # "Name - Role - Company | LinkedIn"
    name = re.split(r"\s[-–|]\s", title)[0].strip() or "Unknown"
    role = ""
    parts = re.split(r"\s[-–]\s", title)
    if len(parts) > 1:
        role = parts[1].replace("| LinkedIn", "").strip()
    pid = hashlib_id(link)
    return {
        "id": pid,
        "name": name,
        "role": role,
        "company": "",
        "school": "",
        "connection": connection,
        "profileUrl": link.split("?")[0],
    }


def hashlib_id(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode()).hexdigest()[:12]


@router.get("/search")
def alumni_search(
    company: str = Query(..., min_length=1),
    school: str = Query(""),
    _user_id: str = Depends(require_user),
):
    key, cx = _cse_keys()
    if not key or not cx:
        raise HTTPException(
            503,
            "Alumni search needs GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX in agent/.env "
            "(Google Programmable Search). Jobs recommend still works without this.",
        )

    company = company.strip()
    school = school.strip()
    queries: list[str] = []
    if school:
        queries.append(f'site:linkedin.com/in "{school}" "{company}"')
        queries.append(f'site:linkedin.com/in (ULM OR "{school}") "{company}"')
    queries.append(
        f'site:linkedin.com/in ("Recruiter" OR "Talent Acquisition" OR "University Recruiter") "{company}"'
    )
    queries.append(
        f'site:linkedin.com/in ("Hiring Manager" OR "Technical Recruiter") "{company}"'
    )

    people: list[dict[str, Any]] = []
    seen: set[str] = set()
    notes: list[str] = []
    for q in queries:
        try:
            items = _cse_search(q, key, cx, num=8)
        except Exception as e:
            notes.append(f"query failed: {e}")
            continue
        is_recruiter = "Recruiter" in q or "Hiring Manager" in q
        for item in items:
            title = item.get("title") or ""
            snippet = item.get("snippet") or ""
            blob = f"{title}\n{snippet}"
            if not _text_matches_company(blob, company):
                continue
            if school and not is_recruiter and not _text_matches_university(blob, school):
                continue
            conn = "referral" if is_recruiter else ("school" if school else "warm")
            person = _person_from_item(item, conn)
            if not person:
                continue
            person["company"] = company
            if school:
                person["school"] = school
            url = person["profileUrl"]
            if url in seen:
                continue
            seen.add(url)
            people.append(person)

    return {
        "ok": True,
        "source": "google-cse",
        "queries": queries,
        "people": people[:24],
        "filterNote": "; ".join(notes) if notes else None,
        "message": None if people else "No indexed LinkedIn profiles matched. Try another company spelling.",
    }
