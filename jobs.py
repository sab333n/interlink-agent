"""
Job recommend / refresh / freehire faceted search API.

Sources: GitHub Simplify + public ATS boards + freehire.me catalogue.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from auth import require_user
import freehire
import jobs_ingest
import match_score

router = APIRouter(prefix="/api/jobs")


def _is_intern(role: str) -> bool:
    return bool(re.search(r"\bintern", role, re.I))


class RecommendRequest(BaseModel):
    profile: dict[str, Any] = Field(default_factory=dict)
    skills: list[str] = Field(default_factory=list)
    experience: list[dict[str, Any]] = Field(default_factory=list)
    limit: int = 40
    refresh: bool = False
    exclude_ids: list[str] = Field(default_factory=list)
    internships_only: bool = False
    sponsors_only: bool = False
    min_live: int = 0
    # freehire-style facets (applied as a live overlay on recommend)
    work_mode: Optional[str] = None
    seniority: Optional[str] = None
    collections: Optional[str] = None
    remote_only: bool = False
    fresh_only: bool = False


class FreehireSearchRequest(BaseModel):
    q: str = ""
    limit: int = 40
    offset: int = 0
    countries: str = "us"
    work_mode: Optional[str] = None
    seniority: Optional[str] = None
    category: Optional[str] = None
    employment_type: Optional[str] = None
    skills: Optional[str] = None
    collections: Optional[str] = None
    domains: Optional[str] = None
    reality: Optional[str] = None
    visa_sponsorship: Optional[bool] = None
    salary_min: Optional[int] = None
    salary_currency: Optional[str] = None
    posted_within_days: Optional[int] = None
    open_within_days: Optional[int] = None
    sort: Optional[str] = None
    order: Optional[str] = None
    agent: bool = False


class MarketFitRequest(BaseModel):
    skills: list[str] = Field(default_factory=list)
    countries: str = "us"
    category: Optional[str] = None
    seniority: Optional[str] = None


@router.get("/health")
def jobs_health():
    cached = bool(jobs_ingest._cache.get("jobs"))
    return {
        "ok": True,
        "cached": cached,
        "count": len(jobs_ingest._cache.get("jobs") or []),
        "fetched_at": jobs_ingest._cache.get("at") or 0,
        "sources": ["simplify-github", "greenhouse", "lever", "ashby", "freehire"],
        "freehire": {
            "search": "https://freehire.me/api/v1/jobs/search",
            "facets": True,
            "similar": True,
            "reality_ghost": True,
            "market_fit": "facets-proxy",
        },
    }


@router.post("/refresh")
def refresh(_user_id: str = Depends(require_user)):
    result = jobs_ingest.refresh_jobs(force=True)
    return {
        "ok": True,
        "count": result["count"],
        "sources": result.get("sources"),
        "errors": result.get("errors", []),
        "fetched_at": result["fetched_at"],
    }


def _merge_jobs(*batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for batch in batches:
        for j in batch:
            key = j.get("freehireSlug") or j.get("id") or f"{j.get('company')}|{j.get('role')}"
            if key in seen:
                continue
            seen.add(key)
            out.append(j)
    return out


@router.post("/recommend")
def recommend(req: RecommendRequest, _user_id: str = Depends(require_user)):
    if req.refresh or not jobs_ingest._cache.get("jobs"):
        jobs_ingest.refresh_jobs(force=req.refresh)
    jobs = jobs_ingest.get_cached_jobs()

    # Live freehire overlay shaped to this profile (closes the “faceted search” gap)
    try:
        live_fh = freehire.fetch_for_profile(req.profile, req.skills, per_query=20)
        jobs = _merge_jobs(live_fh, jobs)
    except Exception:
        pass

    # Optional facet overlays — hit freehire directly then merge
    facet_kwargs: dict[str, Any] = {"countries": "us", "limit": 40}
    if req.work_mode:
        facet_kwargs["work_mode"] = req.work_mode
    if req.seniority:
        facet_kwargs["seniority"] = req.seniority
    if req.collections:
        facet_kwargs["collections"] = req.collections
    if req.remote_only:
        facet_kwargs["work_mode"] = "remote"
    if req.fresh_only:
        facet_kwargs["reality"] = "fresh"
    if req.sponsors_only and not req.collections:
        facet_kwargs["collections"] = "us-h1b-sponsor"
    if any(k in facet_kwargs for k in ("work_mode", "seniority", "collections", "reality")):
        try:
            q = "software engineer"
            targets = req.profile.get("target_roles") or req.profile.get("targetRoles") or []
            if targets:
                q = str(targets[0]).split("&")[0].strip()
            extra = freehire.search(q=q, **facet_kwargs)
            jobs = _merge_jobs(extra.get("jobs") or [], jobs)
        except Exception:
            pass

    skip = set(req.exclude_ids)
    scored: list[dict[str, Any]] = []
    for job in jobs:
        if job["id"] in skip:
            continue
        if job.get("closed") or (job.get("isGhost") and job.get("ghostScore", 0) < 25):
            continue
        if req.internships_only and "Internship" not in (job.get("tags") or []):
            if not _is_intern(job.get("role") or ""):
                continue
        if req.sponsors_only and not job.get("sponsorsVisa"):
            continue
        if req.remote_only and not job.get("remote"):
            continue
        if req.fresh_only and (job.get("reality") or {}).get("class") not in (None, "fresh"):
            # keep non-freehire jobs; only filter freehire when reality present
            if job.get("source") == "freehire" and (job.get("reality") or {}).get("class") != "fresh":
                continue
        if job.get("ghostScore", 0) < req.min_live:
            continue
        m = match_score.compute_match(job, req.profile, req.skills, req.experience)
        overall = m["overall"] if m else 0
        rank = m["rank_score"] if m else job.get("ghostScore", 0) * 0.3
        # Boost freehire fresh + skill overlap lightly
        if job.get("source") == "freehire" and (job.get("reality") or {}).get("class") == "fresh":
            rank = (rank or 0) + 4
        scored.append({
            "job": job,
            "match": m,
            "match_percent": overall,
            "rank_score": rank,
        })
    scored.sort(key=lambda x: (-(x["rank_score"] or 0), x["job"].get("postedDaysAgo", 99)))
    top = scored[: max(1, min(req.limit, 100))]
    return {
        "ok": True,
        "count": len(top),
        "total_indexed": len(jobs),
        "fetched_at": jobs_ingest._cache.get("at") or 0,
        "sources": jobs_ingest._cache.get("sources") or {},
        "picks": top,
    }


@router.post("/search")
def freehire_search(req: FreehireSearchRequest, _user_id: str = Depends(require_user)):
    """Live faceted search against freehire (keyless public catalogue)."""
    filters: dict[str, Any] = {
        "countries": req.countries,
        "work_mode": req.work_mode,
        "seniority": req.seniority,
        "category": req.category,
        "employment_type": req.employment_type,
        "skills": req.skills,
        "collections": req.collections,
        "domains": req.domains,
        "reality": req.reality,
        "salary_min": req.salary_min,
        "salary_currency": req.salary_currency,
        "posted_within_days": req.posted_within_days,
        "open_within_days": req.open_within_days,
        "sort": req.sort,
        "order": req.order,
    }
    if req.visa_sponsorship is not None:
        filters["visa_sponsorship"] = req.visa_sponsorship
    try:
        return freehire.search(
            q=req.q,
            limit=req.limit,
            offset=req.offset,
            agent=req.agent,
            **filters,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"freehire search failed: {e}") from e


@router.get("/facets")
def freehire_facets(
    _user_id: str = Depends(require_user),
    q: str = "",
    countries: str = "us",
    work_mode: Optional[str] = None,
    seniority: Optional[str] = None,
    category: Optional[str] = None,
    skills: Optional[str] = None,
    collections: Optional[str] = None,
):
    try:
        return freehire.facets(
            q=q,
            countries=countries,
            work_mode=work_mode,
            seniority=seniority,
            category=category,
            skills=skills,
            collections=collections,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"freehire facets failed: {e}") from e


@router.get("/freehire/{slug}")
def freehire_job(slug: str, _user_id: str = Depends(require_user)):
    try:
        job = freehire.get_job(slug)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {"ok": True, "job": job}


@router.get("/freehire/{slug}/similar")
def freehire_similar(
    slug: str,
    _user_id: str = Depends(require_user),
    limit: int = Query(8, ge=1, le=20),
):
    try:
        return {"ok": True, "jobs": freehire.similar(slug, limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.post("/market-fit")
def market_fit(req: MarketFitRequest, _user_id: str = Depends(require_user)):
    """Skill coverage vs freehire catalogue (public facets proxy for market/coverage)."""
    if not req.skills:
        raise HTTPException(status_code=400, detail="skills required")
    try:
        return freehire.market_fit(
            req.skills,
            countries=req.countries,
            category=req.category,
            seniority=req.seniority,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.get("/activity")
def catalogue_activity(_user_id: str = Depends(require_user)):
    try:
        return freehire.jobs_activity()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
