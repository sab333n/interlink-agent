"""
Free work-email finder: identity → company domain → email format → confidence.

Uses only public, unauthenticated sources and never logs into or scrapes LinkedIn:
  • DNS over HTTPS (dns.google) for MX records
  • the company's own website (homepage title, contact/about pages)
  • public commits in the company's GitHub organization (name ↔ work-email pairs)
  • Gravatar (an avatar registered to the exact address is strong evidence it exists)

Mailbox-level SMTP verification isn't attempted: residential ISPs block port 25.
Set GITHUB_TOKEN in agent/.env to raise GitHub's 60 requests/hour anonymous limit.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import require_user  # noqa: E402

router = APIRouter(prefix="/api/email", dependencies=[Depends(require_user)])

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) InterLink-EmailFinder/1.0"
CACHE_TTL = 24 * 3600
_domain_cache: dict[str, tuple[float, dict[str, Any]]] = {}

PATTERNS: dict[str, str] = {
    "first.last": "{f}.{l}",
    "first": "{f}",
    "flast": "{fi}{l}",
    "firstlast": "{f}{l}",
    "first_last": "{f}_{l}",
    "firstl": "{f}{li}",
    "f.last": "{fi}.{l}",
    "last": "{l}",
    "last.first": "{l}.{f}",
    "lastf": "{l}{fi}",
    "first-last": "{f}-{l}",
}
# Rough share of companies using each format, used when no evidence is found.
PRIOR = {
    "first.last": 0.42, "first": 0.16, "flast": 0.14, "firstlast": 0.07,
    "firstl": 0.04, "f.last": 0.04, "first_last": 0.03, "last": 0.02,
    "last.first": 0.02, "lastf": 0.02, "first-last": 0.01,
}
ROLE_ACCOUNTS = re.compile(
    r"^(info|hello|hi|contact|support|help|sales|press|media|news|jobs|careers|recruiting|"
    r"talent|hr|people|team|admin|office|privacy|legal|security|abuse|billing|accounts?|"
    r"marketing|partners?|investors?|ir|noreply|no-reply|donotreply|webmaster|postmaster|"
    r"feedback|community|events|social|devrel|engineering|developers?|api|status|ops|"
    r"compliance|dpo|gdpr|ethics|accessibility|inquiries|enquiries|general|mail|email)$",
    re.I,
)
ATS_HOST = re.compile(
    r"(greenhouse\.io|lever\.co|ashbyhq\.com|myworkday(jobs|site)\.com|workday\.com|"
    r"smartrecruiters\.com|icims\.com|taleo\.net|oraclecloud\.com|eightfold\.ai|jobvite\.com|"
    r"workable\.com|successfactors\.(com|eu)|simplify\.jobs|bamboohr\.com|recruitee\.com|"
    r"paylocity\.com|ultipro\.com|ukg\.net|adp\.com|dayforcehcm\.com|brassring\.com|"
    r"avature\.net|phenompeople\.com|rippling\.com|gem\.com|dover\.com|breezy\.hr|"
    r"applytojob\.com|linkedin\.com|indeed\.com|github\.com|notion\.site|wellfound\.com|"
    r"ycombinator\.com|teamtailor\.com|pinpointhq\.com|csod\.com)$",
    re.I,
)
NAME_NOISE = re.compile(
    r"\b(the|inc|llc|corp|corporation|company|co|ltd|limited|plc|group|holdings|"
    r"incorporated|usa|us|na)\b\.?",
    re.I,
)
DESCRIPTOR = re.compile(
    r"\b(technologies|technology|semiconductors?|industries|systems?|solutions|labs|"
    r"software|international|global|enterprises|energy|financial|health|healthcare|ai)\b",
    re.I,
)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


# ── HTTP helpers ────────────────────────────────────────────────────────────

def _get(url: str, *, headers: Optional[dict[str, str]] = None, timeout: float = 8,
         limit: int = 600_000) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(limit).decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:
        return 0, ""


async def aget(url: str, **kw: Any) -> tuple[int, str]:
    return await asyncio.to_thread(_get, url, **kw)


def _github_headers() -> dict[str, str]:
    h = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


class RateLimited(Exception):
    pass


async def gh(path: str) -> Any:
    status, body = await aget(f"https://api.github.com{path}", headers=_github_headers())
    if status in (403, 429):
        raise RateLimited
    if status != 200:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return None


# ── Identity ────────────────────────────────────────────────────────────────

def ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def name_from_linkedin(url: str) -> Optional[str]:
    m = re.search(r"linkedin\.com/in/([^/?#]+)", url, re.I)
    if not m:
        return None
    slug = urllib.parse.unquote(m.group(1))
    parts = [p for p in slug.split("-") if p and not re.search(r"\d", p) and len(p) > 1]
    if len(parts) < 2:
        return None
    return " ".join(p.capitalize() for p in parts[:3])


def split_name(full: str) -> Optional[tuple[str, str]]:
    cleaned = re.sub(r"\(.*?\)|,.*$", "", ascii_fold(full))
    cleaned = re.sub(r"\b(dr|mr|mrs|ms|prof|jr|sr|ii|iii|phd|mba)\b\.?", "", cleaned, flags=re.I)
    parts = [re.sub(r"[^a-z]", "", p.lower()) for p in cleaned.split()]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return None
    return parts[0], parts[-1]


def render(pattern: str, first: str, last: str) -> str:
    return PATTERNS[pattern].format(f=first, l=last, fi=first[0], li=last[0])


def pattern_of(local: str, full_name: str) -> Optional[str]:
    names = split_name(full_name)
    if not names:
        return None
    local = local.lower().split("+")[0]
    for p in PATTERNS:
        if render(p, *names) == local:
            return p
    return None


# ── Company domain ──────────────────────────────────────────────────────────

def registrable(host: str) -> str:
    parts = host.lower().removeprefix("www.").split(".")
    if len(parts) <= 2:
        return ".".join(parts)
    keep = 3 if parts[-2] in {"co", "com", "org", "net", "ac", "gov"} else 2
    return ".".join(parts[-keep:])


def company_slug(company: str, drop_descriptors: bool = False) -> str:
    s = re.sub(r"\(.*?\)", "", company)
    s = NAME_NOISE.sub("", s)
    if drop_descriptors:
        s = DESCRIPTOR.sub("", s)
    return re.sub(r"[^a-z0-9]", "", ascii_fold(s).lower().replace("&", "and"))


def company_tokens(company: str) -> list[str]:
    s = NAME_NOISE.sub("", ascii_fold(company).lower())
    return [t for t in re.findall(r"[a-z0-9]+", s) if len(t) >= 3] or [company_slug(company)]


def domain_candidates(company: str, apply_url: Optional[str], hint: Optional[str]) -> list[str]:
    out: list[str] = []
    if hint:
        out.append(registrable(re.sub(r"^https?://", "", hint).split("/")[0]))
    if apply_url:
        try:
            host = urllib.parse.urlparse(apply_url).hostname or ""
            if host and not ATS_HOST.search(host):
                out.append(registrable(host))
        except ValueError:
            pass
    for slug in (company_slug(company), company_slug(company, True)):
        if len(slug) >= 2:
            out += [f"{slug}.com", f"{slug}.ai", f"{slug}.io", f"{slug}.co", f"{slug}.org"]
    return list(dict.fromkeys(out))


async def mx_records(domain: str) -> list[str]:
    status, body = await aget(f"https://dns.google/resolve?name={domain}&type=MX", timeout=6)
    if status != 200:
        return []
    try:
        answers = json.loads(body).get("Answer") or []
    except ValueError:
        return []
    recs = []
    for a in answers:
        if a.get("type") != 15:
            continue
        bits = str(a.get("data", "")).split()
        if len(bits) == 2 and bits[1] not in {".", ""}:
            recs.append((int(bits[0]), bits[1].rstrip(".").lower()))
    return [h for _, h in sorted(recs)]


def mail_provider(mx: list[str]) -> str:
    joined = " ".join(mx)
    if "google.com" in joined or "googlemail.com" in joined:
        return "Google Workspace"
    if "outlook.com" in joined or "microsoft" in joined:
        return "Microsoft 365"
    if "pphosted" in joined or "ppe-hosted" in joined:
        return "Proofpoint"
    if "mimecast" in joined:
        return "Mimecast"
    return mx[0] if mx else "unknown"


async def homepage(domain: str) -> tuple[bool, str]:
    """(company name appears on the site, raw html)."""
    status, html = await aget(f"https://{domain}", timeout=8)
    if status != 200 or not html:
        status, html = await aget(f"https://www.{domain}", timeout=8)
    return status == 200, html


async def resolve_domain(company: str, apply_url: Optional[str], hint: Optional[str]) -> Optional[dict[str, Any]]:
    tokens = company_tokens(company)
    best: Optional[dict[str, Any]] = None
    for domain in domain_candidates(company, apply_url, hint)[:8]:
        mx, (up, html) = await asyncio.gather(mx_records(domain), homepage(domain))
        if not mx:
            continue
        head = html[:20000].lower()
        title = " ".join(
            " ".join(m) for m in re.findall(
                r"<title[^>]*>(.*?)</title>|og:site_name\"?\s+content=\"([^\"]+)", head, re.S)
        )
        matches = sum(t in title or t in head[:5000] for t in tokens)
        confirmed = up and matches >= max(1, len(tokens) // 2)
        info = {"domain": domain, "mx": mx, "provider": mail_provider(mx),
                "confirmed": confirmed, "html": html}
        if confirmed or domain == (hint or ""):
            return info
        best = best or info
    return best


# ── Format evidence ─────────────────────────────────────────────────────────

async def site_evidence(domain: str, html: str) -> list[dict[str, str]]:
    pages = [html]
    paths = ["/contact", "/contact-us", "/about", "/team", "/about-us", "/press", "/company"]
    results = await asyncio.gather(*(aget(f"https://{domain}{p}", timeout=6) for p in paths))
    pages += [body for status, body in results if status == 200]
    found: dict[str, dict[str, str]] = {}
    for page in pages:
        for email in EMAIL_RE.findall(page):
            email = email.lower().rstrip(".")
            local, _, host = email.partition("@")
            if registrable(host) != domain or ROLE_ACCOUNTS.match(local) or len(local) > 40:
                continue
            found[email] = {"email": email, "name": "", "source": "company website"}
    return list(found.values())[:25]


async def github_evidence(company: str, domain: str) -> list[dict[str, str]]:
    stem = domain.split(".")[0]
    orgs: list[str] = []
    for login in dict.fromkeys([stem, company_slug(company), company_slug(company, True)]):
        if not login:
            continue
        org = await gh(f"/orgs/{login}")
        if not org:
            continue
        blog = str(org.get("blog") or "") + " " + str(org.get("email") or "")
        if domain in blog.lower():
            orgs.append(login)
            break
    if not orgs:
        return []
    repos = await gh(f"/orgs/{orgs[0]}/repos?sort=pushed&per_page=6&type=public") or []
    repos = [r["name"] for r in repos if not r.get("fork")][:3]
    pages = await asyncio.gather(*(gh(f"/repos/{orgs[0]}/{r}/commits?per_page=100") for r in repos))
    pairs: dict[str, dict[str, str]] = {}
    for commits in pages:
        for c in commits or []:
            author = (c.get("commit") or {}).get("author") or {}
            email = str(author.get("email") or "").lower()
            name = str(author.get("name") or "")
            local, _, host = email.partition("@")
            if registrable(host) != domain or ROLE_ACCOUNTS.match(local) or "bot" in name.lower():
                continue
            pairs.setdefault(email, {"email": email, "name": name, "source": f"github.com/{orgs[0]} commits"})
    return list(pairs.values())[:60]


def infer_format(samples: list[dict[str, str]]) -> tuple[Optional[str], int, int]:
    """(pattern, supporting samples, samples with a known name)."""
    votes: Counter[str] = Counter()
    named = 0
    for s in samples:
        if not s["name"]:
            continue
        p = pattern_of(s["email"].split("@")[0], s["name"])
        if p is None:
            continue
        named += 1
        votes[p] += 1
    if votes:
        p, n = votes.most_common(1)[0]
        return p, n, named
    # Unnamed website addresses still reveal shape: "jane.doe" → first.last.
    shapes: Counter[str] = Counter()
    for s in samples:
        local = s["email"].split("@")[0]
        if re.fullmatch(r"[a-z]{2,}\.[a-z]{2,}", local):
            shapes["first.last"] += 1
        elif re.fullmatch(r"[a-z]{2,}_[a-z]{2,}", local):
            shapes["first_last"] += 1
        elif re.fullmatch(r"[a-z]\.[a-z]{2,}", local):
            shapes["f.last"] += 1
    if shapes:
        p, n = shapes.most_common(1)[0]
        return p, n, 0
    return None, 0, 0


async def gravatar_exists(email: str) -> bool:
    h = hashlib.sha256(email.strip().lower().encode()).hexdigest()
    status, _ = await aget(f"https://gravatar.com/avatar/{h}?d=404&s=8", timeout=5, limit=4096)
    return status == 200


async def domain_profile(company: str, apply_url: Optional[str], hint: Optional[str]) -> Optional[dict[str, Any]]:
    key = f"{company.lower()}|{hint or ''}|{apply_url or ''}"
    hit = _domain_cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    info = await resolve_domain(company, apply_url, hint)
    if info is None:
        return None
    async def github_or_limited() -> Optional[list[dict[str, str]]]:
        try:
            return await github_evidence(company, info["domain"])
        except RateLimited:
            return None

    site, github_found = await asyncio.gather(
        site_evidence(info["domain"], info.pop("html")), github_or_limited())
    limited = github_found is None
    github = github_found or []
    samples = github + [s for s in site if s["email"] not in {g["email"] for g in github}]
    pattern, support, named = infer_format(samples)
    info.update(pattern=pattern, support=support, named=named, samples=samples[:8],
                github_limited=limited)
    if not limited:
        _domain_cache[key] = (time.time(), info)
    return info


# ── Scoring ─────────────────────────────────────────────────────────────────

def score_candidates(first: str, last: str, prof: dict[str, Any]) -> list[dict[str, Any]]:
    domain = prof["domain"]
    pattern, support, named = prof["pattern"], prof["support"], prof["named"]
    out = []
    for p, prior in PRIOR.items():
        if pattern:
            if p == pattern:
                ratio = support / max(named, support)
                base = 55 + min(support, 6) * 5 * ratio if named else 50 + min(support, 4) * 4
            else:
                base = prior * 30
        else:
            base = prior * 100
        if not prof["confirmed"]:
            base *= 0.75
        out.append({"email": f"{render(p, first, last)}@{domain}", "pattern": p,
                    "confidence": round(min(base, 90)), "signals": []})
    out.sort(key=lambda c: -c["confidence"])
    return out


# ── LinkedIn identity ───────────────────────────────────────────────────────

BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_profile_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def canonical_linkedin(url: str) -> Optional[tuple[str, str]]:
    """(slug, https://www.linkedin.com/in/<slug>) or None."""
    m = re.search(r"linkedin\.com/in/([^/?#\s]+)", url.strip(), re.I)
    if not m:
        return None
    slug = urllib.parse.unquote(m.group(1)).strip().lower()
    return slug, f"https://www.linkedin.com/in/{urllib.parse.quote(slug)}"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", html_unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def html_unescape(text: str) -> str:
    import html as _html
    return _html.unescape(text)


ROLE_WORDS = re.compile(
    r"\b(co-?founder|founder|ceo|cto|coo|cfo|cpo|vp|svp|evp|president|chair(man|woman)?|director|"
    r"head|lead|principal|senior|sr\.?|staff|manager|engineer(ing)?|developer|designer|recruiter|"
    r"talent|partner|intern|of|product|software|technical|university|campus)\b",
    re.I,
)


def parse_profile_text(title: str, description: str) -> dict[str, str]:
    """Pull name / headline / company out of LinkedIn's public title + description.

    Titles look like "Jane Doe - Acme | LinkedIn" or "Jane Doe - Software Engineer - Acme | LinkedIn";
    descriptions contain "Experience: Acme · Education: … · Location: …".
    """
    out: dict[str, str] = {}
    title = re.sub(r"\s*[|–-]\s*LinkedIn\s*$", "", title).strip()
    parts = [p.strip() for p in re.split(r"\s+[-–]\s+", title) if p.strip()]
    if parts:
        out["name"] = parts[0]
    if len(parts) >= 3:
        out["headline"] = parts[1]
        out["company"] = parts[-1]
    elif len(parts) == 2:
        out["headline"] = parts[1]
    exp = re.search(r"Experience:\s*([^·|\n]+?)\s*(?:·|\||$)", description)
    if exp:
        out["company"] = exp.group(1).strip()
    elif "company" not in out and len(parts) == 2:
        at = re.search(r"\b(?:at|@)\s+(.+)$", parts[1])
        company = at.group(1).strip() if at else ROLE_WORDS.sub("", parts[1]).strip(" ,&/")
        if company:
            out["company"] = company
    loc = re.search(r"Location:\s*([^·|\n]+)", description)
    if loc:
        out["location"] = loc.group(1).strip()
    return out


async def profile_from_linkedin_page(url: str) -> Optional[dict[str, str]]:
    status, body = await aget(url, headers={"User-Agent": BROWSER_UA, "Accept-Language": "en-US,en;q=0.9"},
                              timeout=10)
    if status != 200 or "authwall" in body[:4000]:
        return None
    found: dict[str, str] = {}
    for raw in re.findall(r'<script type="application/ld\+json">(.*?)</script>', body, re.S):
        try:
            data = json.loads(raw)
        except ValueError:
            continue
        for node in data.get("@graph", [data]) if isinstance(data, dict) else []:
            if node.get("@type") != "Person":
                continue
            found["name"] = str(node.get("name") or "")
            titles = node.get("jobTitle")
            titles = [str(t) for t in (titles if isinstance(titles, list) else [titles]) if t]
            readable = [t for t in titles if not re.fullmatch(r"[*\s]+", t)]
            if readable:
                found["headline"] = readable[0]
            works = node.get("worksFor") or []
            if isinstance(works, dict):
                works = [works]
            if works and isinstance(works[0], dict) and works[0].get("name"):
                found["company"] = str(works[0]["name"])
                if works[0].get("url"):
                    found["company_url"] = str(works[0]["url"])
    title = _clean((re.search(r"<title[^>]*>(.*?)</title>", body, re.S) or [None, ""])[1])
    desc_m = re.search(r'<meta[^>]+name="description"[^>]+content="([^"]*)"', body)
    parsed = parse_profile_text(title, html_unescape(desc_m.group(1)) if desc_m else "")
    for k, v in parsed.items():
        found.setdefault(k, v)
    if not found.get("name"):
        return None
    found["source"] = "public LinkedIn profile"
    return found


async def profile_from_search(slug: str, url: str) -> Optional[dict[str, str]]:
    engines = []
    for query in (url, f'"linkedin.com/in/{slug}"', f"site:linkedin.com/in/{slug}"):
        q = urllib.parse.quote(query)
        engines += [
            ("DuckDuckGo", f"https://html.duckduckgo.com/html/?q={q}",
             r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>'),
            ("Bing", f"https://www.bing.com/search?q={q}",
             r'<h2[^>]*><a[^>]+href="([^"]+)"[^>]*>(.*?)</a></h2>.*?<p[^>]*>(.*?)</p>'),
        ]
    for engine, search_url, result_re in engines:
        status, body = await aget(search_url, headers={"User-Agent": BROWSER_UA, "Accept-Language": "en-US"},
                                  timeout=10)
        if status != 200:
            continue
        for href, title, snippet in re.findall(result_re, body, re.S):
            target = urllib.parse.unquote(html_unescape(href)).lower()
            if not re.search(rf"linkedin\.com/in/{re.escape(slug)}(?:[/?&#]|$)", target):
                continue
            parsed = parse_profile_text(_clean(title), _clean(snippet))
            if parsed.get("name"):
                parsed["source"] = f"{engine} listing of the LinkedIn profile"
                return parsed
    return None


async def company_website(company_url: str) -> Optional[str]:
    """The website listed on a public LinkedIn company page."""
    if "linkedin.com/company/" not in company_url:
        return None
    status, body = await aget(company_url.split("?")[0],
                              headers={"User-Agent": BROWSER_UA, "Accept-Language": "en-US"}, timeout=10)
    if status != 200:
        return None
    for raw in re.findall(r'<script type="application/ld\+json">(.*?)</script>', body, re.S):
        try:
            data = json.loads(raw)
        except ValueError:
            continue
        for node in data.get("@graph", [data]) if isinstance(data, dict) else []:
            if node.get("@type") == "Organization":
                for key in ("sameAs", "url"):
                    site = node.get(key)
                    if isinstance(site, str) and "linkedin.com" not in site and site.startswith("http"):
                        return urllib.parse.urlparse(site).hostname
    m = re.search(r'about_website[^>]*href="([^"]+)"', body)
    if m:
        target = urllib.parse.unquote(html_unescape(m.group(1)))
        inner = re.search(r"url=(https?://[^&]+)", target)
        return urllib.parse.urlparse(inner.group(1) if inner else target).hostname
    return None


async def resolve_linkedin(url: str) -> Optional[dict[str, str]]:
    canon = canonical_linkedin(url)
    if not canon:
        return None
    slug, clean_url = canon
    hit = _profile_cache.get(slug)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    found = await profile_from_linkedin_page(clean_url)
    if not found or not found.get("company"):
        searched = await profile_from_search(slug, clean_url)
        if searched:
            found = {**searched, **{k: v for k, v in (found or {}).items() if v}}
    if not found:
        fallback = name_from_linkedin(clean_url)
        found = {"name": fallback, "source": "LinkedIn URL"} if fallback else {"source": "none"}
    found["url"] = clean_url
    if found.get("company"):
        _profile_cache[slug] = (time.time(), found)
    return found


@router.post("/profile")
async def linkedin_profile(req: dict[str, str]) -> dict[str, Any]:
    found = await resolve_linkedin(req.get("linkedin_url", ""))
    if not found:
        raise HTTPException(400, "That doesn't look like a LinkedIn profile URL (linkedin.com/in/…).")
    return found


class FindRequest(BaseModel):
    linkedin_url: Optional[str] = None
    full_name: Optional[str] = None
    company: Optional[str] = None
    title: Optional[str] = None
    domain: Optional[str] = None
    apply_url: Optional[str] = None


@router.post("/find")
async def find_email(req: FindRequest) -> dict[str, Any]:
    identity: dict[str, str] = {}
    if req.linkedin_url:
        identity = await resolve_linkedin(req.linkedin_url) or {}
        if not identity:
            raise HTTPException(400, "That doesn't look like a LinkedIn profile URL (linkedin.com/in/…).")
    full_name = (req.full_name or "").strip() or identity.get("name", "")
    company = (req.company or "").strip() or identity.get("company", "")
    names = split_name(full_name)
    if not names:
        return {"ok": False, "identity": identity, "need": "name",
                "message": "LinkedIn didn't share this profile publicly right now and search engines haven't "
                           "indexed it. Enter the person's name and company to continue."}
    if not company:
        return {"ok": False, "identity": identity, "need": "company",
                "message": f"Found {full_name}, but their current company isn't public on LinkedIn. Enter it to continue."}

    domain_hint = req.domain
    if not domain_hint and identity.get("company_url") and not req.company:
        domain_hint = await company_website(identity["company_url"])
    prof = await domain_profile(company, req.apply_url, domain_hint)
    if prof is None:
        return {"ok": False, "identity": identity, "need": "domain",
                "message": f"Couldn't find a website for {company} that accepts email. Enter the company's domain to continue."}

    candidates = score_candidates(*names, prof)
    top = candidates[:4]
    hits = await asyncio.gather(*(gravatar_exists(c["email"]) for c in top))
    for c, hit in zip(top, hits):
        if hit:
            c["confidence"] = 97
            c["signals"].append("Gravatar profile registered to this exact address")
    candidates.sort(key=lambda c: -c["confidence"])

    signals = []
    if identity.get("company") and not req.company:
        signals.append(f"Current company ({company}) read from the {identity.get('source', 'LinkedIn profile')}")
    signals.append(f"{prof['domain']} accepts email ({prof['provider']})")
    signals.append("Domain confirmed on the company's website" if prof["confirmed"]
                   else "Domain guessed from the company name — not confirmed on its website")
    if prof["pattern"] and prof["named"]:
        signals.append(f"{prof['support']} of {prof['named']} public employee addresses use the {prof['pattern']} format")
    elif prof["pattern"]:
        signals.append(f"Addresses on the company website look like {prof['pattern']}")
    else:
        signals.append("No public employee addresses found — format is the most common industry default")
    if prof.get("github_limited"):
        signals.append("GitHub's hourly lookup limit was hit, so employee commits weren't checked — "
                       "try again later or add GITHUB_TOKEN to agent/.env")
    signals.append("Mailbox not verified (requires SMTP access, blocked on most home networks)")

    best = candidates[0]
    best["signals"] = best["signals"] + signals
    level = "high" if best["confidence"] >= 75 else "medium" if best["confidence"] >= 50 else "low"
    return {
        "ok": True,
        "name": full_name,
        "company": company,
        "identity": identity,
        "domain": prof["domain"],
        "best": best,
        "level": level,
        "alternatives": [c for c in candidates[1:5]],
        "evidence": [{"email": s["email"], "name": s["name"], "source": s["source"]} for s in prof["samples"][:5]],
    }
