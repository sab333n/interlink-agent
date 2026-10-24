"""
Bridge to the local `api-anything` CLI
(https://github.com/goodnight000/api-anything).

Turns LinkedIn (and other taught sites) into JSON your agent can call — after
`api-anything login linkedin` on this machine.

Hosted Render/Fly cannot use this (needs the laptop's session). Local agent can.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from services.auth import require_user

router = APIRouter(prefix="/api/anything")
BIN = os.getenv("API_ANYTHING_BIN", "").strip() or shutil.which("api-anything") or "api-anything"


def _run(args: list[str], timeout: int = 60) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [BIN, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ},
        )
    except FileNotFoundError as e:
        raise HTTPException(
            503,
            "api-anything CLI not found. Install from "
            "https://github.com/goodnight000/api-anything (see INSTALL.md).",
        ) from e
    except subprocess.TimeoutExpired as e:
        raise HTTPException(504, f"api-anything timed out: {' '.join(args)}") from e

    raw = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    try:
        data = json.loads(raw) if raw.startswith("{") else {"ok": False, "raw": raw, "stderr": proc.stderr}
    except json.JSONDecodeError:
        data = {"ok": False, "raw": raw[:2000], "returncode": proc.returncode}
    if proc.returncode != 0 and data.get("ok") is not False:
        data.setdefault("ok", False)
        data.setdefault("returncode", proc.returncode)
    return data


@router.get("/health")
def health():
    path = shutil.which(BIN) or (BIN if os.path.isfile(BIN) else None)
    return {
        "ok": bool(path),
        "bin": path or BIN,
        "hint": None if path else "Install api-anything CLI (GitHub goodnight000/api-anything).",
        "linkedin_login": "api-anything login linkedin",
        "repo": "https://github.com/goodnight000/api-anything",
    }


@router.get("/sites")
def sites(_user_id: str = Depends(require_user)):
    # `api-anything sites` prints text; ops is per-site. Probe known bundled sites.
    known = ["linkedin", "hacker-news", "goodreads", "google-flights", "youtube", "amazon"]
    out = []
    for site in known:
        data = _run(["ops", site], timeout=20)
        if isinstance(data, dict) and data.get("operations"):
            out.append({"site": site, "operations": [o.get("name") for o in data["operations"]]})
        elif data.get("site"):
            out.append({"site": data.get("site"), "operations": [o.get("name") for o in data.get("operations") or []]})
    return {"ok": True, "sites": out}


class CallRequest(BaseModel):
    site: str
    op: str
    params: dict[str, str] = Field(default_factory=dict)


@router.post("/call")
def call_op(req: CallRequest, _user_id: str = Depends(require_user)):
    args = ["call", req.site.strip(), req.op.strip()]
    for k, v in req.params.items():
        if v is None:
            continue
        args.append(f"{k}={v}")
    data = _run(args, timeout=90)
    if data.get("class") == "auth":
        raise HTTPException(
            401,
            data.get("next")
            or f"Run locally: api-anything login {req.site}",
        )
    if not data.get("ok"):
        raise HTTPException(502, data.get("reason") or data.get("raw") or "api-anything call failed")
    return data


class PeopleSearchRequest(BaseModel):
    keywords: str
    company: str = ""


@router.post("/linkedin/people")
def linkedin_people(req: PeopleSearchRequest, _user_id: str = Depends(require_user)):
    """Insider-style people search via api-anything LinkedIn searchPeople."""
    keywords = req.keywords.strip()
    if req.company.strip() and req.company.strip().lower() not in keywords.lower():
        keywords = f"{keywords} {req.company.strip()}".strip()
    if not keywords:
        raise HTTPException(400, "keywords required")
    data = _run(["call", "linkedin", "searchPeople", f"keywords={keywords}"], timeout=90)
    if data.get("class") == "auth":
        raise HTTPException(401, "Run: api-anything login linkedin  (imports your Chrome LinkedIn session)")
    if not data.get("ok"):
        raise HTTPException(502, data.get("reason") or "LinkedIn search failed")

    # Normalize into AlumniSearchPerson-shaped rows when possible
    rows = data.get("data")
    people: list[dict[str, Any]] = []
    if isinstance(rows, list):
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            public_id = row.get("publicId") or row.get("publicIdentifier") or ""
            name = (
                f"{row.get('firstName', '')} {row.get('lastName', '')}".strip()
                or row.get("name")
                or "Unknown"
            )
            people.append({
                "id": f"aa-{public_id or i}",
                "name": name,
                "role": row.get("headline") or row.get("occupation") or row.get("title") or "",
                "company": req.company or row.get("company") or "",
                "school": "",
                "connection": "warm",
                "profileUrl": f"https://www.linkedin.com/in/{public_id}/" if public_id else None,
                "raw": row,
            })
    return {"ok": True, "source": "api-anything/linkedin", "people": people, "ms": data.get("ms")}
