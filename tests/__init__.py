"""
Deterministic unit tests for the agent's pure logic: no LLM, no network, no Supabase.

Run from agent/:  .venv/bin/python -m unittest discover -s tests -t .
(`npm test` at the repo root runs these too.)
"""
import os
import socket
import tempfile

# Blank every key before any agent module loads, so agent/.env can't switch on
# real providers (load_dotenv never overrides variables that are already set).
for _key in ("LLM_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY",
             "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_ANON_KEY",
             "POSTHOG_PROJECT_TOKEN", "AGENT_REQUIRE_AUTH"):
    os.environ[_key] = ""
# Start from an empty profile instead of whoever's profile.json is on disk.
os.environ["PROFILE_PATH"] = os.path.join(tempfile.gettempdir(), "interlink-test-no-profile.json")


def _no_network(*_args, **_kwargs):
    raise RuntimeError("Network access is disabled in unit tests")


socket.socket.connect = _no_network
socket.socket.connect_ex = _no_network
socket.create_connection = _no_network
socket.getaddrinfo = _no_network
