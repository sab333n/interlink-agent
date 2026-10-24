"""
Live job ingest from public sources (no paid job API):

1. SimplifyJobs Summer internships README on GitHub
2. Public Greenhouse / Lever / Ashby board JSON APIs
3. freehire keyless catalogue API (https://github.com/strelov1/freehire)
   — faceted search, reality/ghost signal, enrichment (visa/salary/seniority)

Cached in memory with a TTL so recommend stays fast.
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Optional
from urllib.request import Request, urlopen

import services.freehire as freehire_client

SIMPLIFY_README = (
    "https://raw.githubusercontent.com/SimplifyJobs/Summer2027-Internships/dev/README.md"
)

GREENHOUSE = [
    "stripe", "openai", "anthropic", "datadog", "cloudflare", "notion", "figma",
    "airbnb", "dropbox", "robinhood", "coinbase", "plaid", "ramp", "brex",
    "doordash", "lyft", "pinterest", "reddit", "discord", "hashicorp",
    "mongodb", "snowflake", "databricks", "twilio", "okta", "asana", "grammarly",
]

LEVER = ["netflix", "spotify", "palantir", "scaleai", "rippling", "affirm", "gusto"]
ASHBY = ["linear", "vercel", "resend", "mercury", "clerk"]

CACHE_TTL = 15 * 60  # 15 minutes
_cache: dict[str, Any] = {"at": 0.0, "jobs": []}

UA = "InterLink-agent/1.0 (+https://interlink.local)"


def _get(url: str, timeout: int = 25) -> str:
    req = Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _get_json(url: str) -> Any:
    import json
    return json.loads(_get(url))


def _stable_id(prefix: str, company: str, role: str, apply: str) -> str:
    h = hashlib.sha1(f"{company}|{role}|{apply}".encode()).hexdigest()[:12]
    return f"{prefix}-{h}"


def _days_ago(iso: Optional[str]) -> int:
    if not iso:
        return 14
    try:
        # accept "2026-10-18T..." or date only
        from datetime import datetime, timezone
        raw = iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0, int((datetime.now(timezone.utc) - dt).total_seconds() // 86400))
    except Exception:
        return 14


def _ghost(posted: int, has_url: bool, closed: bool) -> int:
    if closed:
        return 15
    score = 88
    if posted > 45:
        score -= 35
    elif posted > 21:
        score -= 18
    elif posted > 7:
        score -= 8
    if not has_url:
        score -= 25
    return max(5, min(98, score))


def _job(
    *,
    prefix: str,
    company: str,
    role: str,
    location: str,
    apply_url: Optional[str],
    posted: int,
    sponsors: bool,
    remote: bool,
    closed: bool,
    tags: list[str],
    category: str,
    source: str,
    description: str = "",
) -> dict[str, Any]:
    apply = (apply_url or "").strip() or None
    initials = "".join(w[0] for w in company.split()[:2]).upper() or "?"
    return {
        "id": _stable_id(prefix, company, role, apply or role),
        "company": company,
        "role": role,
        "location": location or "United States",
        "applyUrl": apply,
        "postedDaysAgo": posted,
        "sponsorsVisa": sponsors,
        "remote": remote,
        "closed": closed,
        "isGhost": closed or posted > 60,
        "ghostScore": _ghost(posted, bool(apply), closed),
        "tags": tags,
        "category": category,
        "source": source,
        "description": (description or "")[:4000],
        "salary": "—",
        "initials": initials,
        "logoColor": "#171717",
        "ilScoreRequired": None,
        "priorityApply": False,
        "applied": False,
        "ghostSignals": [],
    }


# --- Simplify GitHub README ---

_SECTION_CAT = [
    (re.compile(r"software", re.I), "Software engineering"),
    (re.compile(r"product", re.I), "Product management"),
    (re.compile(r"data science|machine learning|\bai\b", re.I), "Data science & AI"),
    (re.compile(r"quant|finance", re.I), "Quant & finance"),
    (re.compile(r"hardware", re.I), "Hardware & embedded"),
]


def parse_simplify_readme(md: str) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    category = "Software engineering"
    # Track last company for ↳ continuation rows
    last_company = ""
    for block in re.finditer(r"<table[\s\S]*?</table>", md, re.I):
        table = block.group(0)
        # find nearest heading before this table
        head = md.rfind("\n## ", 0, block.start())
        if head >= 0:
            line = md[head: md.find("\n", head + 1)]
            for pat, label in _SECTION_CAT:
                if pat.search(line):
                    category = label
                    break
        for tr in re.finditer(r"<tr[\s\S]*?</tr>", table, re.I):
            tds = re.findall(r"<td[^>]*>([\s\S]*?)</td>", tr.group(0), re.I)
            if len(tds) < 4:
                continue
            # skip header-ish
            if re.search(r"<th", tr.group(0), re.I):
                continue
            company_html, role_html, loc_html, app_html = tds[0], tds[1], tds[2], tds[3]
            age_html = tds[4] if len(tds) > 4 else ""

            company = re.sub(r"<[^>]+>", " ", company_html)
            company = re.sub(r"&amp;", "&", company)
            company = re.sub(r"\s+", " ", company).replace("↳", "").strip()
            if not company or company.lower() == "company":
                # continuation
                company = last_company
            else:
                last_company = company
            if not company:
                continue

            role = re.sub(r"<[^>]+>", " ", role_html)
            role = re.sub(r"\s+", " ", role).strip()
            if not role or role.lower() == "role":
                continue

            location = re.sub(r"<[^>]+>", " ", loc_html)
            location = re.sub(r"\s+", " ", location).strip() or "United States"

            links = re.findall(r'href="([^"]+)"', app_html)
            apply = None
            for u in links:
                u = u.replace("&amp;", "&")
                if re.search(r"imgur|\.png(\?|$)", u, re.I):
                    continue
                apply = u
                break

            age_txt = re.sub(r"<[^>]+>", " ", age_html)
            m = re.search(r"(\d+)d", age_txt)
            posted = int(m.group(1)) if m else 7
            closed = bool(re.search(r"lock|closed", app_html + age_html, re.I))
            sponsors = not bool(re.search(r"🛂|no sponsor", app_html + role_html, re.I))
            remote = bool(re.search(r"remote", location, re.I))
            tags = ["Internship"]
            if re.search(r"new.?grad", role, re.I):
                tags.append("New grad")

            jobs.append(
                _job(
                    prefix="sij",
                    company=company,
                    role=role,
                    location=location,
                    apply_url=apply,
                    posted=posted,
                    sponsors=sponsors,
                    remote=remote,
                    closed=closed,
                    tags=tags,
                    category=category,
                    source="simplify-github",
                )
            )
    return jobs


# --- Public ATS boards ---

def _title_case(slug: str) -> str:
    return slug.replace("-", " ").title()


def fetch_greenhouse(slug: str) -> list[dict[str, Any]]:
    data = _get_json(f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true")
    out = []
    for j in data.get("jobs") or []:
        title = j.get("title") or ""
        if not re.search(r"engineer|software|developer|intern|data|product|design|security|ml |ai ", title, re.I):
            continue
        loc = (j.get("location") or {}).get("name") or "United States"
        tags = ["Full-time"]
        if re.search(r"\bintern", title, re.I):
            tags = ["Internship"]
        elif re.search(r"new.?grad|university|early.?career", title, re.I):
            tags.append("New grad")
        content = j.get("content") or ""
        # strip html lightly
        content = re.sub(r"<[^>]+>", " ", content)
        out.append(
            _job(
                prefix="gh",
                company=(j.get("company") or {}).get("name") or _title_case(slug),
                role=title,
                location=loc,
                apply_url=j.get("absolute_url"),
                posted=_days_ago(j.get("updated_at") or j.get("first_published")),
                sponsors=True,
                remote=bool(re.search(r"remote|anywhere|distributed", f"{loc} {title}", re.I)),
                closed=False,
                tags=tags,
                category="Software engineering" if re.search(r"engineer|software|developer", title, re.I) else "Other",
                source="greenhouse",
                description=content[:4000],
            )
        )
    return out


def fetch_lever(slug: str) -> list[dict[str, Any]]:
    data = _get_json(f"https://api.lever.co/v0/postings/{slug}?mode=json")
    out = []
    for j in data if isinstance(data, list) else []:
        title = j.get("text") or ""
        if not re.search(r"engineer|software|developer|intern|data|product", title, re.I):
            continue
        loc = (j.get("categories") or {}).get("location") or "United States"
        tags = ["Full-time"]
        if re.search(r"\bintern", title, re.I):
            tags = ["Internship"]
        desc = " ".join(j.get("lists") and [x.get("content", "") for x in j["lists"]] or []) or (j.get("descriptionPlain") or "")
        out.append(
            _job(
                prefix="lv",
                company=_title_case(slug),
                role=title,
                location=loc,
                apply_url=j.get("hostedUrl") or j.get("applyUrl"),
                posted=_days_ago(None),  # lever uses createdAt ms
                sponsors=True,
                remote=bool(re.search(r"remote", f"{loc} {title}", re.I)),
                closed=False,
                tags=tags,
                category="Software engineering",
                source="lever",
                description=re.sub(r"<[^>]+>", " ", desc)[:4000],
            )
        )
        created = j.get("createdAt")
        if isinstance(created, (int, float)):
            from datetime import datetime, timezone
            posted = max(0, int((time.time() * 1000 - created) / 86400000))
            out[-1]["postedDaysAgo"] = posted
            out[-1]["ghostScore"] = _ghost(posted, bool(out[-1]["applyUrl"]), False)
    return out


def fetch_ashby(slug: str) -> list[dict[str, Any]]:
    data = _get_json(f"https://api.ashbyhq.com/posting-api/job-board/{slug}")
    out = []
    for j in data.get("jobs") or []:
        title = j.get("title") or ""
        if not re.search(r"engineer|software|developer|intern|data|product", title, re.I):
            continue
        loc = j.get("location") or "United States"
        tags = ["Full-time"]
        if re.search(r"\bintern", title, re.I):
            tags = ["Internship"]
        posted = _days_ago(j.get("publishedAt") or j.get("updatedAt"))
        out.append(
            _job(
                prefix="as",
                company=_title_case(slug),
                role=title,
                location=loc if isinstance(loc, str) else "United States",
                apply_url=j.get("jobUrl") or j.get("applyUrl"),
                posted=posted,
                sponsors=True,
                remote=bool(j.get("isRemote") or re.search(r"remote", str(loc), re.I)),
                closed=False,
                tags=tags,
                category="Software engineering",
                source="ashby",
                description=(j.get("descriptionPlain") or "")[:4000],
            )
        )
    return out


def fetch_freehire(
    *,
    per_query: int = 40,
    countries: str = "us",
    profile: Optional[dict[str, Any]] = None,
    skills: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Pull faceted freehire slices (reality/ghost + enrichment mapped)."""
    if profile is not None:
        return freehire_client.fetch_for_profile(
            profile, skills or [], per_query=per_query, countries=countries
        )
    return freehire_client.fetch_default_slices(per_query=per_query, countries=countries)


def refresh_jobs(force: bool = False) -> dict[str, Any]:
    now = time.time()
    if not force and _cache["jobs"] and now - _cache["at"] < CACHE_TTL:
        return {
            "ok": True,
            "cached": True,
            "fetched_at": _cache["at"],
            "count": len(_cache["jobs"]),
            "jobs": _cache["jobs"],
            "sources": _cache.get("sources", {}),
        }

    errors: list[str] = []
    jobs: list[dict[str, Any]] = []
    sources: dict[str, int] = {}

    try:
        md = _get(SIMPLIFY_README, timeout=40)
        sj = parse_simplify_readme(md)
        jobs.extend(sj)
        sources["simplify-github"] = len(sj)
    except Exception as e:
        errors.append(f"simplify: {e}")

    for slug in GREENHOUSE:
        try:
            rows = fetch_greenhouse(slug)
            jobs.extend(rows)
            sources["greenhouse"] = sources.get("greenhouse", 0) + len(rows)
        except Exception as e:
            errors.append(f"greenhouse/{slug}: {e}")

    for slug in LEVER:
        try:
            rows = fetch_lever(slug)
            jobs.extend(rows)
            sources["lever"] = sources.get("lever", 0) + len(rows)
        except Exception as e:
            errors.append(f"lever/{slug}: {e}")

    for slug in ASHBY:
        try:
            rows = fetch_ashby(slug)
            jobs.extend(rows)
            sources["ashby"] = sources.get("ashby", 0) + len(rows)
        except Exception as e:
            errors.append(f"ashby/{slug}: {e}")

    try:
        fh = fetch_freehire(per_query=35)
        jobs.extend(fh)
        sources["freehire"] = len(fh)
    except Exception as e:
        errors.append(f"freehire: {e}")

    # Dedup by company+role
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for j in jobs:
        key = f"{j['company'].lower()}|{j['role'].lower()}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(j)

    _cache["at"] = now
    _cache["jobs"] = deduped
    _cache["sources"] = sources
    _cache["errors"] = errors[:40]
    return {
        "ok": True,
        "cached": False,
        "fetched_at": now,
        "count": len(deduped),
        "jobs": deduped,
        "sources": sources,
        "errors": errors[:40],
    }


def get_cached_jobs() -> list[dict[str, Any]]:
    if not _cache["jobs"]:
        refresh_jobs(force=True)
    elif time.time() - _cache["at"] > CACHE_TTL:
        refresh_jobs(force=False)
    return list(_cache["jobs"])
