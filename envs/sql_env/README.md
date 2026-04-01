---
title: SQL Query Debugger
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - sql
  - reinforcement-learning
---

# SQL Query Debugger — OpenEnv Environment

An RL environment where AI agents learn to fix broken SQL queries. The agent receives a broken query and a database schema, submits a corrected query, and receives partial-credit reward based on how closely the output matches the expected result.

---

## Environment Description

SQL debugging is a high-value real-world skill: developers spend significant time diagnosing and fixing broken queries across production databases. It requires understanding schema structure, join semantics, aggregation logic, and filtering conditions — a rich combination of reasoning tasks.

This makes SQL debugging well-suited for RL training:

- **Objective is unambiguous** — correctness is determined by exact comparison against a known expected output, not a subjective judge.
- **Partial credit is natural** — a query that returns the right columns but wrong rows is better than one that crashes, enabling a smooth reward gradient.
- **Difficulty is tunable** — tasks range from simple syntax fixes to multi-table aggregation bugs, making it easy to curriculum-train agents.
- **Execution is cheap and safe** — all queries run against isolated in-memory SQLite databases with no external infrastructure required.

---

## Action Space

The agent submits a single action per episode: a corrected SQL query.

| Field | Type | Description |
|-------|------|-------------|
| `fixed_query` | `str` | The agent's corrected SQL query. Must be valid SQLite syntax. |

```python
from envs.sql_env.models import SqlAction

action = SqlAction(fixed_query="SELECT name FROM employees WHERE salary > 50000")
```

---

## Observation Space

Observations are returned by both `reset()` (initial state) and `step()` (after the agent acts).

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Unique identifier for the current task (e.g. `easy_001`). |
| `difficulty` | `str` | Difficulty level: `easy`, `medium`, or `hard`. |
| `broken_query` | `str` | The broken SQL query the agent must fix. |
| `schema_description` | `str` | Human-readable description of the database schema, including tables, columns, and types. |
| `expected_description` | `str` | Plain English description of what the correct query should return. |
| `query_result` | `str` | Formatted string showing the result of executing the agent's query. Empty string on `reset()`. |
| `reward` | `float \| None` | Reward for the submitted query (see Reward Function below). `None` on `reset()`. |
| `done` | `bool` | Whether the episode has ended. Always `True` after `step()`. |
| `success` | `bool` | `True` if the agent achieved a perfect match (`reward == 1.0`). |
| `feedback` | `str` | Human-readable explanation of the score, e.g. `"Perfect match!"` or `"Right columns but wrong number of rows."` |

---

## Reward Function

The environment uses a partial-credit scoring system. Reward is determined by how closely the agent's query output matches the expected output:

| Reward | Condition |
|--------|-----------|
| `0.0` | Query fails to execute — syntax error or runtime error. |
| `0.2` | Query executes but returns wrong column names. |
| `0.4` | Right columns, but wrong number of rows. |
| `0.7` | Right columns and row count, but some cell values differ. |
| `1.0` | Perfect match — output is identical to expected result. |

Comparison is case-insensitive on column names and order-insensitive on rows. Floating-point values are compared with 6 decimal places of precision.

---

## Tasks

Tasks are defined in `data/tasks.json`. Each task includes a schema, seed data, a broken query, and the expected correct query. Tasks are grouped into three difficulty levels.

### Easy — Syntax Errors

Simple `SELECT` queries on a single table with surface-level bugs: dangling `AND` clauses, wrong column names, or mismatched aliases. No joins required.

**Example bug:** The query references a column that does not exist in the schema.

### Medium — Wrong JOIN Conditions

Multi-table queries with incorrect join predicates or mismatched foreign keys. The overall query structure is correct; the agent must identify which join condition is wrong.

**Example bug:** Two tables are joined on the wrong column, producing a cartesian product or an empty result set.

### Hard — Aggregation and GROUP BY Bugs

Queries involving `GROUP BY`, `HAVING`, subqueries, or aggregation functions (`SUM`, `AVG`, `COUNT`) with incorrect logic — such as grouping on the wrong column, applying an aggregation to the wrong field, or a missing `HAVING` clause.

**Example bug:** Revenue is aggregated by the wrong dimension, returning totals that are numerically plausible but factually incorrect.

---

## Baseline Scores

Evaluated using `llama-3.3-70b-versatile` via [Groq](https://console.groq.com) with seed `42`:

| Difficulty | Task ID | Reward | Result |
|------------|---------|--------|--------|
| easy | `easy_001` | 1.0 | PASS |
| medium | `medium_001` | 1.0 | PASS |
| hard | `hard_001` | 1.0 | PASS |

**Overall score: 1.000** (3.0 / 3 tasks)

---

## Setup and Usage

### Prerequisites

```bash
# From the OpenEnv repo root
uv sync --all-extras
```

### Run the Server Locally

```bash
# From the OpenEnv repo root
PYTHONPATH=src:envs uv run python -m envs.sql_env.server
# Server starts at http://localhost:8000
# WebSocket endpoint: ws://localhost:8000
# Web UI: http://localhost:8000/web
```

### Run the Baseline Agent

Requires a free [Groq API key](https://console.groq.com).

```bash
export GROQ_API_KEY=<your-key>
PYTHONPATH=src:envs uv run python baseline.py
```

Override the model with:

```bash
MODEL_NAME=llama-3.1-8b-instant GROQ_API_KEY=<your-key> uv run python baseline.py
```

### Run with Docker

```bash
# Build the image
docker build -t sql-env:latest -f envs/sql_env/server/Dockerfile .

# Run the container
docker run -p 8000:8000 sql-env:latest
```

### Hugging Face Space

The environment is live at:

**[https://huggingface.co/spaces/Subman1246/sql_env](https://huggingface.co/spaces/Subman1246/sql_env)**

Connect directly to the Space:

```python
env = SqlEnv(base_url="wss://subman1246-sql-env.hf.space")
```

---

## Quick Start

```python
import asyncio
from envs.sql_env.client import SqlEnv
from envs.sql_env.models import SqlAction

async def main():
    async with SqlEnv(base_url="ws://localhost:8000") as env:
        # Reset to receive a task — filter by difficulty and seed for reproducibility
        result = await env.reset(seed=42, difficulty="medium")
        obs = result.observation

        print(f"Task:         {obs.task_id} ({obs.difficulty})")
        print(f"Schema:       {obs.schema_description}")
        print(f"Expected:     {obs.expected_description}")
        print(f"Broken query: {obs.broken_query}")

        # Submit a corrected query
        result = await env.step(SqlAction(fixed_query="SELECT ..."))
        obs = result.observation

        print(f"Reward:   {obs.reward}")
        print(f"Success:  {obs.success}")
        print(f"Feedback: {obs.feedback}")
        print(f"Output:\n{obs.query_result}")

asyncio.run(main())
```

### Sync API

```python
from envs.sql_env.client import SqlEnv
from envs.sql_env.models import SqlAction

env = SqlEnv(base_url="ws://localhost:8000").sync()
with env:
    result = env.reset(seed=42, difficulty="easy")
    result = env.step(SqlAction(fixed_query="SELECT name FROM employees WHERE salary > 50000"))
    print(result.observation.feedback)
```

---

## File Structure

```
envs/sql_env/
├── client.py                  # SqlEnv client (WebSocket)
├── models.py                  # SqlAction, SqlObservation, SqlState
├── openenv.yaml               # OpenEnv manifest
├── pyproject.toml             # Project metadata and dependencies
├── data/
│   └── tasks.json             # Task definitions (schema, broken query, expected query)
└── server/
    ├── app.py                 # FastAPI + WebSocket server entry point
    ├── sql_environment.py     # Core environment logic and grader
    └── Dockerfile             # Container image
```
