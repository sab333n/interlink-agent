"""
Server-side job match — same signals as the frontend, so recommend is not
browser-only. Honest heuristics: we never invent skills or education.
"""
from __future__ import annotations

import re
from typing import Any, Optional

TARGET_ROLE_PATTERNS: dict[str, re.Pattern[str]] = {
    "Software engineering": re.compile(r"software|swe|full.?stack|backend|frontend|platform|infrastructure", re.I),
    "Product management": re.compile(r"product manag|\bapm\b|program manag", re.I),
    "Data science & analytics": re.compile(r"data scien|data analy|analytics|business intel", re.I),
    "Machine learning & AI": re.compile(r"machine learning|\bml\b|\bai\b|deep learning|\bllm", re.I),
    "Quant & finance": re.compile(r"quant|trading|trader|fintech", re.I),
    "Hardware & embedded": re.compile(r"hardware|embedded|firmware|fpga|asic", re.I),
}

CATEGORY_TRACKS: dict[str, list[str]] = {
    "Software engineering": ["Software engineering"],
    "Product management": ["Product management"],
    "Data science & AI": ["Data science & analytics", "Machine learning & AI"],
    "Quant & finance": ["Quant & finance"],
    "Hardware & embedded": ["Hardware & embedded"],
}

SKILL_GROUPS = [
    ("Machine learning", re.compile(r"machine learning|\bml\b|\bai\b|deep learning|computer vision|\bnlp\b|\bllm", re.I),
     re.compile(r"machine learning|pytorch|tensorflow|keras|scikit|sklearn|deep learning|\bnlp\b|\bllm|computer vision|hugging.?face|\bml\b|\bai\b", re.I)),
    ("Data", re.compile(r"\bdata\b|analytics|analyst|statistic", re.I),
     re.compile(r"\bsql\b|pandas|numpy|tableau|power.?bi|excel|statistics|\br\b|spark|data analysis|postgres|mysql|snowflake", re.I)),
    ("Frontend", re.compile(r"front.?end|\bweb\b|\bui\b|full.?stack", re.I),
     re.compile(r"react|javascript|typescript|html|css|vue|angular|next\.?js|svelte", re.I)),
    ("Backend", re.compile(r"back.?end|platform|infrastructure|full.?stack|\bapi\b|cloud|devops|\bsre\b", re.I),
     re.compile(r"node|java\b|\bgo\b|golang|django|flask|fastapi|spring|postgres|mysql|mongodb|\bsql\b|aws|gcp|azure|docker|kubernetes|rest|graphql", re.I)),
    ("Software fundamentals", re.compile(r"software|developer|\bswe\b|engineer", re.I),
     re.compile(r"python|java\b|c\+\+|\bc#|javascript|typescript|\bgo\b|golang|rust|kotlin|swift|data structures|algorithms|\bgit\b", re.I)),
]


def _clamp(n: float) -> int:
    return max(0, min(100, int(round(n))))


def _role_text(job: dict[str, Any]) -> str:
    tags = job.get("tags") or []
    return f"{job.get('role', '')} {job.get('category') or ''} {' '.join(tags)}"


def _matches_track(job: dict[str, Any], track: str) -> bool:
    cat = job.get("category") or ""
    if cat and track in CATEGORY_TRACKS.get(cat, []):
        return True
    pat = TARGET_ROLE_PATTERNS.get(track)
    return bool(pat and pat.search(job.get("role") or ""))


def compute_match(
    job: dict[str, Any],
    profile: dict[str, Any],
    skills: list[str],
    experience: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    targets = [t for t in (profile.get("target_roles") or profile.get("targetRoles") or []) if t]
    user_skills = [s for s in skills if s]
    for e in experience:
        user_skills.extend(e.get("skills") or [])
    user_skills = list(dict.fromkeys(user_skills))

    if not targets and not user_skills:
        return None

    fit_reasons: list[str] = []
    gaps: list[str] = []

    role: Optional[int] = None
    if targets:
        hit = next((t for t in targets if _matches_track(job, t)), None)
        role = 100 if hit else 15
        if hit:
            fit_reasons.append(f"Matches your target track: {hit}.")
        else:
            gaps.append(f"Outside your target roles ({', '.join(targets)}).")

    text = _role_text(job)
    groups = [g for g in SKILL_GROUPS if g[1].search(text)]
    matched: list[str] = []
    missing: list[str] = []
    skills_score: Optional[int] = None
    if user_skills and groups:
        covered = 0.0
        for _label, _trig, skill_re in groups:
            have = [s for s in user_skills if skill_re.search(s)]
            if have:
                covered += min(len(have), 3) / 3
                for s in have:
                    if s not in matched:
                        matched.append(s)
            else:
                missing.append(_label)
        skills_score = _clamp(100 * covered / len(groups))
        if matched:
            more = f" and {len(matched) - 4} more" if len(matched) > 4 else ""
            fit_reasons.append(f"Relevant skills: {', '.join(matched[:4])}{more}.")
        if missing:
            gaps.append(f"No {' / '.join(missing)} skills on your profile yet.")

    experience_score: Optional[int] = None
    if experience:
        relevant = []
        for e in experience:
            blob = f"{e.get('title', '')} {e.get('description', '')} {' '.join(e.get('skills') or [])}"
            if any(TARGET_ROLE_PATTERNS.get(t) and TARGET_ROLE_PATTERNS[t].search(blob) for t in targets) or any(
                g[1].search(blob) or g[2].search(blob) for g in groups
            ):
                relevant.append(e)
        experience_score = 100 if relevant else 55
        if relevant:
            fit_reasons.append(f"Related experience: {relevant[0].get('title')} at {relevant[0].get('company')}.")

    # Education
    school = (profile.get("school") or "").strip()
    degree = (profile.get("degree") or "").strip()
    major = (profile.get("major") or "").strip()
    education: Optional[int] = None
    if school or degree or major:
        education = 70
        is_intern = bool(re.search(r"intern|co-?op|new grad|university|campus", text, re.I))
        wants_phd = bool(re.search(r"ph\.?d|doctoral", text, re.I))
        has_phd = bool(re.search(r"ph\.?d|doctoral", degree, re.I))
        stem = bool(re.search(r"computer|software|cs\b|data|stat|electrical|math|engineer", major, re.I))
        tech = bool(re.search(r"software|engineer|data|machine|ml|ai|quant|developer", text, re.I))
        if is_intern:
            education = 92
            if school:
                fit_reasons.append(f"Education fits early-career / intern roles ({school}).")
        if wants_phd and not has_phd:
            education = min(education, 40)
            gaps.append("Listing prefers a PhD; your profile doesn’t list one.")
        elif degree:
            fit_reasons.append(f"Degree on file: {degree}" + (f" · {major}" if major else "") + ".")
        if stem and tech:
            education = min(100, education + 12)
        education = _clamp(education)

    if role is None and skills_score is None and education is None:
        return None

    parts = [(role, 0.38), (skills_score, 0.3), (experience_score, 0.17), (education, 0.15)]
    weight = sum(w for v, w in parts if v is not None)
    overall = sum((v or 0) * w for v, w in parts if v is not None) / weight if weight else 0

    sponsorship = profile.get("sponsorship") or profile.get("needs_sponsorship")
    if sponsorship in ("Yes", True, "yes") and not job.get("sponsorsVisa", job.get("sponsors_visa", True)):
        overall *= 0.4
        gaps.append("Listing says it does not sponsor visas, and you need sponsorship.")

    remote_pref = profile.get("remotePreference") or profile.get("remote_preference")
    if remote_pref == "Remote":
        if job.get("remote"):
            fit_reasons.append("Remote, which matches your preference.")
        else:
            gaps.append("Not remote, and you prefer remote work.")

    if job.get("isGhost") or job.get("closed"):
        gaps.append("Listing may be closed — verify before applying.")

    ghost = int(job.get("ghostScore") or job.get("ghost_score") or 70)
    live = max(0, min(100, ghost)) / 100
    rank = overall * (0.55 + 0.45 * live)

    return {
        "overall": _clamp(overall),
        "role": role,
        "skills": skills_score,
        "experience": experience_score,
        "education": education,
        "matched_skills": matched,
        "missing_skills": missing,
        "fit_reasons": fit_reasons,
        "gaps": gaps,
        "rank_score": round(rank, 2),
        "label": (
            "Strong match" if overall >= 85 else
            "Good match" if overall >= 70 else
            "Fair match" if overall >= 55 else
            "Stretch"
        ),
    }
