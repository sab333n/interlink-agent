"""
Model-backed texting agent: browser → this route → model API → streamed back.

The API key lives only in the server environment. Any OpenAI-compatible provider works:
  LLM_API_KEY      required (falls back to GROQ_API_KEY / OPENAI_API_KEY / XAI_API_KEY)
  LLM_BASE_URL     defaults to the provider implied by whichever key is set
  LLM_MODEL        the writing model (drafts, real replies)
  LLM_FAST_MODEL   cheap model for short command-style turns

The response is newline-delimited JSON events:
  {"type":"text","delta":"..."}   streamed reply text
  {"type":"picks","picks":[...]}  job cards to render
  {"type":"draft","draft":{...}}  an email draft card
  {"type":"apply","job_id":"..."} open in-site apply for a job
  {"type":"person","person":{...}} who the conversation is about now
  {"type":"error","message":"..."}
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from auth import require_user
import email_finder

router = APIRouter(prefix="/api/chat")

PROVIDERS = [
    ("LLM_API_KEY", None, None, None),
    ("GROQ_API_KEY", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "openai/gpt-oss-20b"),
    ("OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4o", "gpt-4o-mini"),
    ("XAI_API_KEY", "https://api.x.ai/v1", "grok-3", "grok-3-mini"),
]


def llm_config() -> Optional[dict[str, str]]:
    for env_key, base, model, fast in PROVIDERS:
        key = os.getenv(env_key, "").strip()
        if not key:
            continue
        if env_key == "LLM_API_KEY" and not os.getenv("LLM_BASE_URL"):
            base, model, fast = PROVIDERS[1][1:]
        base = os.getenv("LLM_BASE_URL") or base
        model = os.getenv("LLM_MODEL") or model
        return {"key": key, "base": base, "model": model, "fast": os.getenv("LLM_FAST_MODEL") or fast or model}
    return None


SYSTEM_PROMPT = """You talk like a smart friend texting, not a support bot or a careers site.
You're InterLink: you help a student or early-career applicant find roles, reach the right people, and apply.

VOICE
- lowercase by default. short sentences. plain words
- lead with the answer, then the why
- have a take. pick a lane, don't hedge both sides
- when it's good news, be hyped. when they're wrong, correct gently
- 1-3 sentences is normal. saying almost nothing is often the best reply
- it's fine to be funny. it's fine to not fill space

NEVER
- "I hope this helps", "as an AI", "it's important to note"
- exclamation spam, bullet lists for casual talk, hedging
- walls of text when two lines would do
- inventing jobs, emails, companies, numbers, or experience. only use CONTEXT and tool results.
  if you don't know, say so in one line

TOOLS (you act through these, you never pretend)
- show_picks: show their best-matching open roles from CONTEXT, by pick number
- find_work_email: look up someone's work email from a LinkedIn profile URL. be honest if it's a guess
- draft_email: a short outreach email, under 90 words: a specific hook about the company or role,
  ONE concrete proof point from their real experience (numbers if they have them), one small ask
  (15 minutes), signed with their name. emails are normal sentence case, not lowercase
- start_application: open the in-app application for a pick. they review everything before it's sent
nothing goes out without their say-so.

EXAMPLES (match the rhythm, not the words)
good news: "wait that's huge - how'd it go?"
correction: "ah got you - july, not june."
practical: "yeah, that role fits better. want the draft?"
new fact: "a hackathon win? founders love that. was it rules or ML?"
picks: (call show_picks) "these three. cloudflare's the one - your python + react lines up."
best one: "cloudflare. 88% match, posted 2 days ago, and they sponsor. want me to start it?"
linkedin url: (call find_work_email) "found her. jane.doe@stripe.com - likely, not verified. want me to write to her?"
yes: (call draft_email) "here. short on purpose, recruiters skim."
nerves: "normal. worst case is silence, and silence costs you nothing."
spray request: "nah, spraying hurts your odds. let's do the 3 best fits properly."
unknown: "google's not in your list, so i can't score it honestly."
thanks: "anytime. go get it."
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "show_picks",
        "description": "Show the user job picks from CONTEXT as cards.",
        "parameters": {"type": "object", "properties": {
            "numbers": {"type": "array", "items": {"type": "integer"},
                        "description": "Pick numbers from CONTEXT, best first, usually 3."}},
            "required": ["numbers"]}}},
    {"type": "function", "function": {
        "name": "find_work_email",
        "description": "Find a person's work email from their LinkedIn profile URL.",
        "parameters": {"type": "object", "properties": {
            "linkedin_url": {"type": "string"}}, "required": ["linkedin_url"]}}},
    {"type": "function", "function": {
        "name": "draft_email",
        "description": "Show an outreach email draft card the user can edit and send from Gmail.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string", "description": "Recipient email"},
            "to_name": {"type": "string"},
            "company": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string", "description": "Full email body including greeting and sign-off"}},
            "required": ["to", "to_name", "company", "subject", "body"]}}},
    {"type": "function", "function": {
        "name": "start_application",
        "description": "Open the in-app application for a pick number from CONTEXT.",
        "parameters": {"type": "object", "properties": {
            "number": {"type": "integer"}}, "required": ["number"]}}},
]


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatTurn]
    context: dict[str, Any] = {}


def context_block(ctx: dict[str, Any]) -> str:
    return "CONTEXT (the only facts you may use about the user, their jobs and progress):\n" + json.dumps(
        ctx, ensure_ascii=False, indent=1)[:9000]


def is_quick(text: str) -> bool:
    """Short command-ish turns go to the cheap model; writing and real questions go to the good one."""
    t = text.strip().lower()
    if re.search(r"draft|write|email|shorter|rewrite|why|how should|help me", t):
        return False
    return len(t) < 40


def event(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode()


async def run_tool(name: str, args: dict[str, Any], ctx: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Returns (result for the model, events for the browser)."""
    picks = ctx.get("picks") or []
    if name == "show_picks":
        chosen = [picks[n - 1] for n in args.get("numbers", []) if 0 < n <= len(picks)]
        if not chosen:
            return "No matching picks in CONTEXT.", []
        return json.dumps(chosen), [{"type": "picks", "picks": chosen}]
    if name == "start_application":
        n = int(args.get("number", 0))
        if not 0 < n <= len(picks):
            return "That pick number doesn't exist.", []
        pick = picks[n - 1]
        if not pick.get("can_apply_in_app"):
            return f"{pick['company']} has no online form InterLink can fill.", []
        return f"Opened the application for {pick['company']}.", [{"type": "apply", "job_id": pick["job_id"]}]
    if name == "draft_email":
        draft = {k: str(args.get(k, "")) for k in ("to", "to_name", "company", "subject", "body")}
        return "Draft shown to the user.", [{"type": "draft", "draft": draft}]
    if name == "find_work_email":
        try:
            r = await email_finder.find_email(email_finder.FindRequest(linkedin_url=args.get("linkedin_url", "")))
        except HTTPException as e:
            return f"Lookup failed: {e.detail}", []
        if not r.get("ok"):
            return f"Lookup incomplete: {r.get('message')}", []
        person = {"name": r["name"], "company": r["company"], "headline": r.get("identity", {}).get("headline", ""),
                  "email": r["best"]["email"], "confidence": r["best"]["confidence"], "level": r["level"]}
        return json.dumps(person), [{"type": "person", "person": person}]
    return f"Unknown tool {name}.", []


async def stream_reply(req: ChatRequest, cfg: dict[str, str]) -> AsyncIterator[bytes]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=cfg["key"], base_url=cfg["base"])
    history = [{"role": t.role if t.role in ("user", "assistant") else "user", "content": t.content}
               for t in req.messages[-24:] if t.content.strip()]
    last = history[-1]["content"] if history else ""
    model = cfg["fast"] if is_quick(last) and not re.search(r"linkedin\.com/in/", last) else cfg["model"]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": context_block(req.context)},
        *history,
    ]
    try:
        for _ in range(4):
            stream = await client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS, stream=True, temperature=0.6,
                max_tokens=1500 if "gpt-oss" in model else 500,
                extra_body={"reasoning_effort": "low"} if "gpt-oss" in model else None)
            text, calls = "", {}
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    text += delta.content
                    yield event({"type": "text", "delta": delta.content})
                for tc in delta.tool_calls or []:
                    slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                    slot["id"] = tc.id or slot["id"]
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["args"] += tc.function.arguments
            if not calls:
                return
            messages.append({"role": "assistant", "content": text or None, "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                for c in calls.values()]})
            for c in calls.values():
                try:
                    args = json.loads(c["args"] or "{}")
                except ValueError:
                    args = {}
                result, events = await run_tool(c["name"], args, req.context)
                for ev in events:
                    yield event(ev)
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
            # Tool results need judgment to summarize, so the follow-up uses the writing model.
            model = cfg["model"]
            if text:
                yield event({"type": "text", "delta": "\n"})
    except Exception as e:  # provider errors shouldn't crash the stream mid-way
        yield event({"type": "error", "message": f"model error: {str(e)[:200]}"})


@router.get("/config")
async def chat_config(_: str = Depends(require_user)) -> dict[str, Any]:
    cfg = llm_config()
    return {"enabled": cfg is not None, "model": cfg["model"] if cfg else None}


@router.post("")
async def chat(req: ChatRequest, _: str = Depends(require_user)):
    cfg = llm_config()
    if cfg is None:
        raise HTTPException(503, "No model API key configured on the agent server.")
    return StreamingResponse(stream_reply(req, cfg), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
