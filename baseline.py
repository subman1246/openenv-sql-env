"""
SQL Query Debugger — Baseline LLM Agent (Groq)
===============================================
Connects to a running sql_env server, runs an LLM via Groq's
OpenAI-compatible API on every task at each difficulty level,
and prints a score report.

Get a free Groq API key at: https://console.groq.com
  1. Sign up / log in
  2. Go to "API Keys" in the left sidebar
  3. Click "Create API Key" and copy it
  4. Export it: export GROQ_API_KEY=<your-key>

Environment variables:
    GROQ_API_KEY  — Groq API key (required)
    MODEL_NAME    — Model to use (default: llama-3.3-70b-versatile)

Usage:
    python baseline.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from dataclasses import dataclass
from typing import List

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Required: get a free key at https://console.groq.com
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Groq's OpenAI-compatible endpoint
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")

# sql_env WebSocket server (started via: python -m envs.sql_env.server)
SQL_ENV_URL = "ws://localhost:8000"

# One task per difficulty, all with seed=42 for reproducibility
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

async def ask_llm(client: AsyncOpenAI, broken_query: str, schema: str, expected_description: str) -> str:
    """Send broken query + schema to Groq LLM and return the fixed SQL."""
    user_message = (
        f"Database schema:\n{schema}\n\n"
        f"Expected result: {expected_description}\n\n"
        f"Broken SQL query to fix:\n{broken_query}"
    )
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if the model wrapped the query anyway
    raw = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    difficulty: str
    task_id: str
    reward: float
    success: bool
    feedback: str
    fixed_query: str


async def run_episode(client: AsyncOpenAI, difficulty: str) -> EpisodeResult:
    """Run one episode at the given difficulty using seed=42."""
    from envs.sql_env.client import SqlEnv
    from envs.sql_env.models import SqlAction

    async with SqlEnv(base_url=SQL_ENV_URL) as env:
        # Reset with seed=42 and difficulty to get a deterministic task
        reset_result = await env.reset(seed=SEED, difficulty=difficulty)
        obs = reset_result.observation

        print(f"\n[{difficulty.upper()}] Task: {obs.task_id}")
        print(f"  Broken query : {obs.broken_query}")

        # Ask the LLM (via Groq) to fix the query
        fixed_query = await ask_llm(
            client,
            broken_query=obs.broken_query,
            schema=obs.schema_description,
            expected_description=obs.expected_description,
        )
        print(f"  Fixed query  : {fixed_query}")

        # Submit the fixed query to the environment
        step_result = await env.step(SqlAction(fixed_query=fixed_query))
        obs = step_result.observation

        print(f"  Reward: {obs.reward:.1f}  Success: {obs.success}")
        print(f"  Feedback: {obs.feedback}")

        return EpisodeResult(
            difficulty=difficulty,
            task_id=obs.task_id,
            reward=obs.reward,
            success=obs.success,
            feedback=obs.feedback,
            fixed_query=fixed_query,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    if not GROQ_API_KEY:
        print(
            "ERROR: GROQ_API_KEY is not set.\n"
            "Get a free key at https://console.groq.com and run:\n"
            "  export GROQ_API_KEY=<your-key>",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build one AsyncOpenAI client pointed at Groq's OpenAI-compatible endpoint
    groq_client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    print(f"Model  : {MODEL_NAME}")
    print(f"Server : {SQL_ENV_URL}")
    print(f"Seed   : {SEED}")
    print("=" * 60)

    results: List[EpisodeResult] = []
    for difficulty in DIFFICULTY_LEVELS:
        try:
            result = await run_episode(groq_client, difficulty)
            results.append(result)
        except Exception as exc:
            print(f"\n[{difficulty.upper()}] ERROR: {exc}")
            results.append(
                EpisodeResult(
                    difficulty=difficulty,
                    task_id="unknown",
                    reward=0.0,
                    success=False,
                    feedback=str(exc),
                    fixed_query="",
                )
            )

    # Score report
    print("\n" + "=" * 60)
    print("SCORE REPORT")
    print("=" * 60)
    total_reward = 0.0
    for r in results:
        status = "PASS" if r.success else "FAIL"
        print(f"  {r.difficulty:8s}  task={r.task_id:12s}  reward={r.reward:.1f}  [{status}]")
        total_reward += r.reward

    overall = total_reward / len(results) if results else 0.0
    print("-" * 60)
    print(f"  Overall score: {overall:.3f}  ({total_reward:.1f} / {len(results):.0f} tasks)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
