"""
SQL Query Debugger — Hackathon Inference Script
================================================
Runs an LLM agent against sql_env for three difficulty levels and
prints structured output that the hackathon judges parse.

Environment variables:
    API_BASE_URL      — OpenAI-compatible base URL
                        (default: https://api.groq.com/openai/v1)
    MODEL_NAME        — Model to use
                        (default: llama-3.3-70b-versatile)
    HF_TOKEN          — API key (also checked as API_KEY)
    LOCAL_IMAGE_NAME  — Docker image name for the environment
                        (default: sql-env)

Usage:
    export HF_TOKEN=<your-api-key>
    python inference.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from typing import List

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "sql-env")

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
SEED = 42
MAX_STEPS = 5

SYSTEM_PROMPT = """\
You are a SQL debugging expert. You will be given a broken SQL query and the
database schema. Your job is to fix the query so it produces the correct result
described by the user.

Reply with ONLY the corrected SQL query — no explanation, no markdown fences,
no extra text. Just the raw SQL statement.\
"""


# ---------------------------------------------------------------------------
# Structured logging (judges parse this output)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_str = error if error is not None else "null"
    done_str = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM helper (sync OpenAI client)
# ---------------------------------------------------------------------------

def ask_llm(client: OpenAI, broken_query: str, schema: str, expected_description: str) -> str:
    """Send broken query + schema to the LLM and return the fixed SQL."""
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
    # Strip markdown fences if the model wrapped the query anyway
    raw = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


# ---------------------------------------------------------------------------
# Episode runner (async env, sync LLM)
# ---------------------------------------------------------------------------

async def run_episode(client: OpenAI, difficulty: str) -> None:
    """Run one episode at the given difficulty and print structured logs."""
    from envs.sql_env.client import SqlEnv
    from envs.sql_env.models import SqlAction

    log_start(task=difficulty, env="sql_env", model=MODEL_NAME)

    rewards: List[float] = []
    steps = 0
    score = 0.0
    success = False

    try:
        async with SqlEnv.from_docker_image(LOCAL_IMAGE_NAME) as env:
            reset_result = await env.reset(seed=SEED, difficulty=difficulty)
            obs = reset_result.observation
            done = False

            while steps < MAX_STEPS and not done:
                steps += 1
                error_msg = None
                try:
                    fixed_query = ask_llm(
                        client,
                        broken_query=obs.broken_query,
                        schema=obs.schema_description,
                        expected_description=obs.expected_description,
                    )
                    step_result = await env.step(SqlAction(fixed_query=fixed_query))
                    obs = step_result.observation
                    reward = float(obs.reward)
                    done = bool(obs.success) or step_result.done
                except Exception as exc:
                    fixed_query = ""
                    reward = 0.0
                    done = True
                    error_msg = str(exc)

                rewards.append(reward)
                log_step(step=steps, action=fixed_query, reward=reward, done=done, error=error_msg)

        score = rewards[-1] if rewards else 0.0
        success = score >= 0.5

    except Exception as exc:
        error_msg = str(exc)
        if steps == 0:
            rewards.append(0.0)
            log_step(step=1, action="", reward=0.0, done=True, error=error_msg)
            steps = 1
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    if not HF_TOKEN:
        print(
            "ERROR: No API key found. Set HF_TOKEN or API_KEY and retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    for difficulty in DIFFICULTY_LEVELS:
        await run_episode(client, difficulty)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
