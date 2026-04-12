"""
SQL Query Debugger — Local Baseline Script
==========================================
Runs an LLM agent against sql_env for quick local testing.
Not used for submission — see inference.py for the hackathon format.

Environment variables:
    HF_TOKEN          — API key (also checked as API_KEY)
    API_BASE_URL      — OpenAI-compatible base URL
                        (default: https://api.groq.com/openai/v1)
    MODEL_NAME        — Model to use
                        (default: llama-3.3-70b-versatile)
    SQL_ENV_URL       — WebSocket URL of the sql_env server
                        (default: ws://localhost:8000)

Usage:
    export HF_TOKEN=<your-api-key>
    PYTHONPATH=src:envs uv run python envs/sql_env/baseline.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "envs"))

from envs.sql_env.client import SqlEnv
from envs.sql_env.models import SqlAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
SQL_ENV_URL = os.getenv("SQL_ENV_URL", "ws://localhost:8000")

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
SEED = 42

SYSTEM_PROMPT = """\
You are a SQL debugging expert. You will be given a broken SQL query and the
database schema. Your job is to fix the query so it produces the correct result
described by the user.

Reply with ONLY the corrected SQL query — no explanation, no markdown fences,
no extra text. Just the raw SQL statement.\
"""


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def ask_llm(client: OpenAI, broken_query: str, schema: str, expected_description: str) -> str:
    user_message = (
        f"Database schema:\n{schema}\n\n"
        f"Expected result: {expected_description}\n\n"
        f"Broken SQL query to fix:\n{broken_query}"
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_episode(client: OpenAI, difficulty: str) -> float:
    print(f"\n{'='*60}")
    print(f"  Difficulty: {difficulty.upper()}")
    print(f"{'='*60}")

    async with SqlEnv(base_url=SQL_ENV_URL) as env:
        reset_result = await env.reset(seed=SEED, difficulty=difficulty)
        obs = reset_result.observation

        print(f"Task:         {obs.task_id}")
        print(f"Broken query: {obs.broken_query}")
        print(f"Expected:     {obs.expected_description}")

        fixed_query = ask_llm(
            client,
            broken_query=obs.broken_query,
            schema=obs.schema_description,
            expected_description=obs.expected_description,
        )
        print(f"\nFixed query:  {fixed_query}")

        step_result = await env.step(SqlAction(fixed_query=fixed_query))
        obs = step_result.observation

        print(f"\nReward:   {obs.reward:.2f}")
        print(f"Feedback: {obs.feedback}")
        if obs.query_result:
            print(f"Output:\n{obs.query_result}")

        return float(obs.reward) if obs.reward is not None else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    if not HF_TOKEN:
        print("ERROR: No API key found. Set HF_TOKEN or API_KEY and retry.", file=sys.stderr)
        sys.exit(1)

    print(f"Model:   {MODEL_NAME}")
    print(f"Server:  {SQL_ENV_URL}")

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    scores: list[float] = []
    for difficulty in DIFFICULTY_LEVELS:
        reward = await run_episode(client, difficulty)
        scores.append(reward)

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for difficulty, score in zip(DIFFICULTY_LEVELS, scores):
        status = "PASS" if score >= 0.5 else "FAIL"
        print(f"  {difficulty:10s}  reward={score:.2f}  {status}")
    print(f"\n  Overall: {sum(scores) / len(scores):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
