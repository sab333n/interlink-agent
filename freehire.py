"""
freehire.me public API client (https://github.com/strelov1/freehire).

Keyless reads we lean on:
  GET /api/v1/jobs/search     — faceted full-text search
  GET /api/v1/agent/jobs/search — same + full descriptions
  GET /api/v1/jobs/facets    — facet histograms (market proxy)
  GET /api/v1/jobs/{slug}    — job detail
  GET /api/v1/jobs/{slug}/similar
  GET /api/v1/stats/jobs-activity

Authenticated freehire surfaces (market/coverage, tracking, inbox) need a
personal freehire API key — we do not proxy those; we approximate market-fit
from public facets instead.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Optional

BASE = "https://freehire.me/api/v1"
UA = "InterLink-agent/1.0 (+https://interlink.local; freehire client)"

# InterLink track → freehire category facet
TRACK_TO_CATEGORY: dict[str, str] = {
    "Software engineering": "software_engineering",
    "Product management": "product",
    "Data science & analytics": "data_science",
    "Machine learning & AI": "ml_ai",
    "Quant & finance": "finance",
    "Hardware & embedded": "embedded",
}

CATEGORY_LABEL: dict[str, str] = {
    "software_engineering": "Software engineering",
    "backend": "Software engineering",
    "frontend": "Software engineering",
    "fullstack": "Software engineering",
    "mobile": "Software engineering",
    "devops": "Software engineering",
    "sre": "Software engineering",
    "product": "Product management",
    "data_science": "Data science & AI",
    "data_analytics": "Data science & AI",
    "data_engineering": "Data science & AI",
    "ml_ai": "Data science & AI",
    "ai_engineering": "Data science & AI",
    "finance": "Quant & finance",
    "embedded": "Hardware & embedded",
    "hardware": "Hardware & embedded",
}

EMPLOYMENT_TAG: dict[str, str] = {
    "internship": "Internship",
    "fellowship": "Internship",
    "full_time": "Full-time",
    "part_time": "Part-time",
    "contract": "Contract",
}


def _get(url: str, timeout: int = 30) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def _qs(params: dict[str, Any]) -> str:
    pairs: list[tuple[str, str]] = []
    for k, v in params.items():
        if v is None or v == "" or v is False:
            continue
        if isinstance(v, bool):
            pairs.append((k, "true" if v else "false"))
        elif isinstance(v, (list, tuple)):
            vals = [str(x).strip() for x in v if str(x).strip()]
            if vals:
                pairs.append((k, ",".join(vals)))
        else:
            pairs.append((k, str(v)))
    return urllib.parse.urlencode(pairs)


def _days_ago(iso: Optional[str]) -> int:
    if not iso:
        return 14
    try:
        raw = iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0, int((datetime.now(timezone.utc) - dt).total_seconds() // 86400))
    except Exception:
        return 14


def _strip_html(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html or "")).strip()


def _salary(enrich: dict[str, Any]) -> str:
    lo = enrich.get("salary_min")
    hi = enrich.get("salary_max")
    cur = (enrich.get("salary_currency") or "USD").upper()
    period = enrich.get("salary_period") or "year"
    if lo is None and hi is None:
        return "—"
    sym = {"USD": "$", "EUR": "€", "GBP": "£"}.get(cur, f"{cur} ")
    suffix = {"year": "/yr", "month": "/mo", "day": "/day", "hour": "/hr"}.get(period, "")
    try:
        lo_i = int(lo) if lo is not None else None
        hi_i = int(hi) if hi is not None else None
    except (TypeError, ValueError):
        return "—"
    if lo_i is not None and hi_i is not None and lo_i != hi_i:
        if lo_i >= 1000 and hi_i >= 1000:
            return f"{sym}{lo_i // 1000}k–{hi_i // 1000}k{suffix}"
        return f"{sym}{lo_i}–{hi_i}{suffix}"
    n = hi_i if hi_i is not None else lo_i
    if n is None:
        return "—"
    if n >= 1000:
        return f"{sym}{n // 1000}k{suffix}"
    return f"{sym}{n}{suffix}"


def _ghost_from_reality(
    reality: Optional[dict[str, Any]],
    posted: int,
    has_url: bool,
    closed: bool,
) -> tuple[int, bool, list[dict[str, str]]]:
    """Map freehire's reality signal into InterLink ghostScore + ghostSignals."""
    signals: list[dict[str, str]] = []
    if closed:
        signals.append(
            {
                "label": "Closed",
                "description": "Posting is marked closed on freehire.",
                "strength": "major-risk",
            }
        )
        return 15, True, signals

    score = 88
    cls = (reality or {}).get("class") or "fresh"
    age = int((reality or {}).get("age_days") or posted or 0)
    repost = int((reality or {}).get("repost_count") or 0)
    mass = int((reality or {}).get("mass_posting_count") or 0)
    fake = bool((reality or {}).get("fake_freshness"))

    if cls == "fresh":
        signals.append(
            {
                "label": "Fresh listing",
                "description": f"freehire reality class: fresh ({age}d in catalogue).",
                "strength": "strong-positive",
            }
        )
    elif cls == "stale":
        score -= 22
        signals.append(
            {
                "label": "Stale posting",
                "description": f"Open ~{age}d with little movement — may be evergreen.",
                "strength": "caution",
            }
        )
    elif cls == "likely-evergreen":
        score -= 38
        signals.append(
            {
                "label": "Likely evergreen",
                "description": "Repost / always-open pattern — treat as low urgency.",
                "strength": "major-risk",
            }
        )

    if fake:
        score -= 20
        signals.append(
            {
                "label": "Fake freshness",
                "description": "Source restates posted date without a real reopen.",
                "strength": "minor-risk",
            }
        )
    if repost >= 3:
        score -= 12
        signals.append(
            {
                "label": f"Reposted {repost}×",
                "description": "Same role keeps reappearing on the board.",
                "strength": "caution",
            }
        )
    elif repost >= 1:
        signals.append(
            {
                "label": f"Reposted {repost}×",
                "description": "Seen at least one repost cycle.",
                "strength": "caution" if repost > 1 else "positive",
            }
        )
    if mass >= 5:
        score -= 10
        signals.append(
            {
                "label": "Mass posting",
                "description": f"{mass} similar roles dropped in a batch.",
                "strength": "minor-risk",
            }
        )
    if not has_url:
        score -= 25
        signals.append(
            {
                "label": "No apply URL",
                "description": "Missing original ATS link.",
                "strength": "major-risk",
            }
        )
    elif has_url:
        signals.append(
            {
                "label": "Direct ATS link",
                "description": "Apply URL points at the company board, not a recruiter repost.",
                "strength": "positive",
            }
        )

    score = max(5, min(98, score))
    is_ghost = closed or cls == "likely-evergreen" or score < 40
    return score, is_ghost, signals


def map_job(j: dict[str, Any], *, require_tech: bool = False) -> Optional[dict[str, Any]]:
    """Normalize a freehire wire job into InterLink JobPosting shape (+ extras)."""
    title = (j.get("title") or "").strip()
    company = (j.get("company") or "").strip()
    if not title or not company:
        return None
    if require_tech and not re.search(
        r"engineer|software|developer|intern|data|product|design|security|"
        r"machine learning|\bml\b|\bai\b|sre|fullstack|full.?stack|devops|quant",
        title,
        re.I,
    ):
        return None

    enrich = j.get("enrichment") if isinstance(j.get("enrichment"), dict) else {}
    url = (j.get("url") or "").strip() or None
    if url and "utm_source=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}utm_source=interlink"

    loc = (j.get("location") or "").strip() or "United States"
    work_mode = (j.get("work_mode") or "").lower()
    remote = work_mode == "remote" or bool(
        re.search(r"remote|anywhere|distributed", f"{loc} {title}", re.I)
    )
    hybrid = work_mode == "hybrid"

    emp = (enrich.get("employment_type") or "").lower()
    tags: list[str] = []
    if emp in EMPLOYMENT_TAG:
        tags.append(EMPLOYMENT_TAG[emp])
    elif re.search(r"\bintern", title, re.I):
        tags.append("Internship")
    else:
        tags.append("Full-time")
    if re.search(r"new.?grad|university|early.?career|entry.?level", title, re.I):
        tags.append("New grad")
    if remote:
        tags.append("Remote")
    elif hybrid:
        tags.append("Hybrid")

    skills = [s for s in (j.get("skills") or []) if isinstance(s, str)][:12]
    tags = list(dict.fromkeys([*tags, *skills[:4]]))

    collections = [c for c in (j.get("collections") or []) if isinstance(c, str)]
    visa = enrich.get("visa_sponsorship")
    if visa is True or "us-h1b-sponsor" in collections or "uk-skilled-worker-sponsor" in collections:
        sponsors = True
        if "us-h1b-sponsor" in collections:
            tags.append("H-1B sponsor")
    elif visa is False:
        sponsors = False
    else:
        sponsors = False

    fh_cat = (enrich.get("category") or "").lower()
    category = CATEGORY_LABEL.get(fh_cat, "Software engineering")
    if not fh_cat:
        if re.search(r"product manag|\bapm\b", title, re.I):
            category = "Product management"
        elif re.search(r"data scien|machine learning|\bml\b|\bai\b", title, re.I):
            category = "Data science & AI"

    posted = _days_ago(j.get("posted_at") or j.get("updated_at") or j.get("created_at"))
    # Prefer freehire catalogue age when reality has it
    reality = j.get("reality") if isinstance(j.get("reality"), dict) else {}
    if reality.get("age_days") is not None:
        try:
            posted = max(posted, int(reality["age_days"]))
        except (TypeError, ValueError):
            pass

    closed = bool(j.get("closed_at"))
    ghost_score, is_ghost, ghost_signals = _ghost_from_reality(reality, posted, bool(url), closed)

    desc = j.get("description") or ""
    if "<" in desc:
        desc = _strip_html(desc)
    summary = (enrich.get("summary") or "").strip()
    if summary and not desc:
        desc = summary

    reqs = enrich.get("requirements") or []
    req_texts = []
    if isinstance(reqs, list):
        for r in reqs[:12]:
            if isinstance(r, dict) and r.get("text"):
                req_texts.append(str(r["text"]))
            elif isinstance(r, str):
                req_texts.append(r)

    slug = j.get("public_slug") or ""
    # Stable id from slug when present so refresh doesn't churn ids
    if slug:
        job_id = f"fh-{slug[:48]}"
    else:
        import hashlib
        job_id = "fh-" + hashlib.sha1(f"{company}|{title}|{url or ''}".encode()).hexdigest()[:12]

    initials = "".join(w[0] for w in company.split()[:2]).upper() or "?"
    seniority = enrich.get("seniority")
    if seniority:
        tags = list(dict.fromkeys([*tags, str(seniority).replace("_", " ").title()]))

    return {
        "id": job_id,
        "company": company,
        "role": title,
        "location": loc,
        "applyUrl": url,
        "postedDaysAgo": posted,
        "sponsorsVisa": sponsors,
        "remote": remote,
        "closed": closed,
        "isGhost": is_ghost,
        "ghostScore": ghost_score,
        "ghostSignals": ghost_signals,
        "tags": tags,
        "category": category,
        "source": "freehire",
        "description": (desc or "")[:6000],
        "salary": _salary(enrich),
        "initials": initials,
        "logoColor": "#0a0a0a",
        "ilScoreRequired": None,
        "priorityApply": "us-h1b-sponsor" in collections or seniority == "intern",
        "applied": False,
        # freehire extras (safe for FE to ignore)
        "freehireSlug": slug,
        "freehireSource": j.get("source"),
        "companySlug": j.get("company_slug"),
        "workMode": work_mode or None,
        "seniority": seniority,
        "employmentType": emp or None,
        "skills": skills,
        "collections": collections,
        "domains": enrich.get("domains") or [],
        "requirements": req_texts,
        "summary": summary or None,
        "reality": reality or None,
        "experienceYearsMin": enrich.get("experience_years_min"),
        "educationLevel": enrich.get("education_level"),
        "companyType": enrich.get("company_type"),
        "companySize": enrich.get("company_size"),
    }


def search(
    *,
    q: str = "",
    limit: int = 40,
    offset: int = 0,
    agent: bool = False,
    description_format: str = "text",
    **filters: Any,
) -> dict[str, Any]:
    """Faceted search. Returns {jobs, meta, raw_count} with mapped jobs."""
    params: dict[str, Any] = {
        "limit": max(1, min(int(limit), 100)),
        "offset": max(0, int(offset)),
        **filters,
    }
    if q:
        params["q"] = q
    if agent:
        params["description_format"] = description_format
        path = f"{BASE}/agent/jobs/search"
    else:
        path = f"{BASE}/jobs/search"
    data = _get(f"{path}?{_qs(params)}")
    jobs = []
    for row in data.get("data") or []:
        mapped = map_job(row, require_tech=False)
        if mapped:
            jobs.append(mapped)
    return {
        "ok": True,
        "jobs": jobs,
        "meta": data.get("meta") or {},
        "ignored_params": (data.get("meta") or {}).get("ignored_params"),
    }


def facets(*, q: str = "", **filters: Any) -> dict[str, Any]:
    params: dict[str, Any] = {**filters}
    if q:
        params["q"] = q
    data = _get(f"{BASE}/jobs/facets?{_qs(params)}")
    return {"ok": True, "data": data.get("data") or data}


def get_job(slug: str) -> Optional[dict[str, Any]]:
    data = _get(f"{BASE}/jobs/{urllib.parse.quote(slug)}")
    row = data.get("data") or data
    return map_job(row, require_tech=False) if row else None


def similar(slug: str, *, limit: int = 8) -> list[dict[str, Any]]:
    data = _get(f"{BASE}/jobs/{urllib.parse.quote(slug)}/similar?limit={max(1, min(limit, 20))}")
    out = []
    for row in data.get("data") or []:
        mapped = map_job(row, require_tech=False)
        if mapped:
            out.append(mapped)
    return out


def jobs_activity() -> dict[str, Any]:
    data = _get(f"{BASE}/stats/jobs-activity")
    return {"ok": True, "data": data.get("data") or []}


def market_fit(
    skills: list[str],
    *,
    countries: str = "us",
    category: Optional[str] = None,
    seniority: Optional[str] = None,
) -> dict[str, Any]:
    """
    Approximate freehire's market/coverage (auth-only) using public facets.

    For each skill: how many US (etc.) openings list it, vs the filtered market total.
    """
    clean = [s.strip().lower() for s in skills if s and s.strip()][:16]
    base_filters: dict[str, Any] = {"countries": countries}
    if category:
        base_filters["category"] = category
    if seniority:
        base_filters["seniority"] = seniority

    base = facets(**base_filters)
    payload = base.get("data") or {}
    total = int(payload.get("total") or 0)
    skill_hist = (payload.get("facets") or {}).get("skills") or {}

    covered = []
    gaps = []
    for s in clean:
        # exact then fuzzy key match in histogram
        count = skill_hist.get(s)
        if count is None:
            for k, v in skill_hist.items():
                if s in k or k in s:
                    count = v
                    s = k
                    break
        count = int(count or 0)
        share = round(100.0 * count / total, 2) if total else 0.0
        row = {"skill": s, "jobs": count, "share_pct": share}
        if count > 0:
            covered.append(row)
        else:
            # probe individually — hist may truncate
            try:
                probe = facets(skills=s, **base_filters)
                ptotal = int((probe.get("data") or {}).get("total") or 0)
                share = round(100.0 * ptotal / total, 2) if total else 0.0
                row = {"skill": s, "jobs": ptotal, "share_pct": share}
                if ptotal > 0:
                    covered.append(row)
                else:
                    gaps.append(row)
            except Exception:
                gaps.append(row)

    covered.sort(key=lambda x: -x["jobs"])
    coverage_pct = round(100.0 * len(covered) / len(clean), 1) if clean else 0.0
    return {
        "ok": True,
        "market_total": total,
        "filters": base_filters,
        "coverage_pct": coverage_pct,
        "covered": covered,
        "gaps": gaps,
        "top_market_skills": sorted(
            [{"skill": k, "jobs": int(v)} for k, v in skill_hist.items()],
            key=lambda x: -x["jobs"],
        )[:25],
    }


def queries_for_profile(profile: dict[str, Any], skills: list[str]) -> list[dict[str, Any]]:
    """Build several faceted freehire searches from the InterLink profile."""
    targets = profile.get("target_roles") or profile.get("targetRoles") or []
    sponsorship = bool(profile.get("sponsorship") or profile.get("sponsorsVisa"))
    remote_pref = (profile.get("remote_preference") or profile.get("remotePreference") or "").lower()

    searches: list[dict[str, Any]] = []

    def add(q: str, **filt: Any) -> None:
        searches.append({"q": q, **filt})

    # Early-career defaults
    add("software engineer intern", seniority="intern", employment_type="internship")
    add("software engineer", seniority="junior")
    add("new grad software engineer", seniority="junior")

    for track in targets[:4]:
        cat = TRACK_TO_CATEGORY.get(track)
        if cat:
            add(track.split("&")[0].strip(), category=cat, seniority="intern")
            add(track.split("&")[0].strip(), category=cat, seniority="junior")
        else:
            add(str(track), seniority="intern")

    if skills:
        top = ",".join(s.lower() for s in skills[:5] if s)
        if top:
            add("", skills=top, seniority="intern")
            add("", skills=top, category="software_engineering")

    if sponsorship:
        add("software engineer", collections="us-h1b-sponsor", seniority="intern")
        add("software engineer", collections="us-h1b-sponsor", seniority="junior")

    if "remote" in remote_pref:
        add("software engineer intern", work_mode="remote", seniority="intern")

    # Dedup identical filter sets
    seen: set[str] = set()
    unique = []
    for s in searches:
        key = json.dumps(s, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique[:12]


def fetch_for_profile(
    profile: dict[str, Any],
    skills: list[str],
    *,
    per_query: int = 25,
    countries: str = "us",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in queries_for_profile(profile, skills):
        try:
            result = search(limit=per_query, countries=countries, **spec)
        except Exception:
            try:
                result = search(limit=per_query, **spec)  # drop country
            except Exception:
                continue
        for job in result["jobs"]:
            slug = job.get("freehireSlug") or job["id"]
            if slug in seen:
                continue
            seen.add(slug)
            out.append(job)
    return out


def fetch_default_slices(*, per_query: int = 40, countries: str = "us") -> list[dict[str, Any]]:
    """Catalogue slices used on full index refresh."""
    specs = [
        {"q": "software engineer intern", "seniority": "intern"},
        {"q": "software engineering internship", "employment_type": "internship"},
        {"q": "new grad software engineer", "seniority": "junior"},
        {"q": "machine learning intern", "seniority": "intern", "category": "ml_ai"},
        {"q": "data science intern", "seniority": "intern", "category": "data_science"},
        {"q": "frontend engineer intern", "seniority": "intern", "category": "frontend"},
        {"q": "backend engineer intern", "seniority": "intern", "category": "backend"},
        {"q": "product manager intern", "seniority": "intern", "category": "product"},
        {"q": "software engineer", "collections": "us-h1b-sponsor", "seniority": "intern"},
        {"q": "software engineer", "collections": "yc", "seniority": "intern"},
        {"q": "software engineer", "work_mode": "remote", "seniority": "intern"},
        {"q": "software engineer", "reality": "fresh", "seniority": "junior"},
    ]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        try:
            result = search(limit=per_query, countries=countries, **spec)
        except Exception:
            try:
                result = search(limit=per_query, **spec)
            except Exception:
                continue
        for job in result["jobs"]:
            slug = job.get("freehireSlug") or job["id"]
            if slug in seen:
                continue
            seen.add(slug)
            out.append(job)
    return out
