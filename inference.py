"""
SQL Query Debugger — Baseline LLM Agent
========================================
Connects to a running sql_env server, runs an LLM via any OpenAI-compatible
API on every task at each difficulty level, and prints a score report.

Environment variables:
    OPENAI_API_KEY  — API key (required)
    API_BASE_URL    — Base URL for the OpenAI-compatible API
                      (default: https://api.groq.com/openai/v1)
    MODEL_NAME      — Model to use (default: llama-3.3-70b-versatile)
    SQL_ENV_URL     — WebSocket URL of the sql_env server
                      (default: ws://localhost:8000)

Usage:
    export OPENAI_API_KEY=<your-key>
    python inference.py
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

API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
SQL_ENV_URL = os.environ.get("SQL_ENV_URL", "ws://localhost:8000")

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
    """Send broken query + schema to the LLM and return the fixed SQL."""
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
        reset_result = await env.reset(seed=SEED, difficulty=difficulty)
        obs = reset_result.observation

        print(f"\n[{difficulty.upper()}] Task: {obs.task_id}")
        print(f"  Broken query : {obs.broken_query}")

        fixed_query = await ask_llm(
            client,
            broken_query=obs.broken_query,
            schema=obs.schema_description,
            expected_description=obs.expected_description,
        )
        print(f"  Fixed query  : {fixed_query}")

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
    if not API_KEY:
        print(
            "ERROR: OPENAI_API_KEY is not set.\n"
            "Set your API key and run:\n"
            "  export OPENAI_API_KEY=<your-key>",
            file=sys.stderr,
        )
        sys.exit(1)

    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    print(f"Model  : {MODEL_NAME}")
    print(f"API    : {API_BASE_URL}")
    print(f"Server : {SQL_ENV_URL}")
    print(f"Seed   : {SEED}")
    print("=" * 60)

    results: List[EpisodeResult] = []
    for difficulty in DIFFICULTY_LEVELS:
        try:
            result = await run_episode(client, difficulty)
            results.append(result)
        except Exception as exc:
            print(f"\n[{difficulty.upper()}] ERROR: {exc}", file=sys.stderr)
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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
