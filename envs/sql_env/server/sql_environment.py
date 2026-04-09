# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SQL Query Debugger Environment.

Presents agents with a broken SQL query and a database schema. The agent must
fix the query so it produces the correct output. Uses SQLite in-memory databases
and partial-credit scoring.

Difficulty levels:
  - easy:   syntax errors only (dangling AND, wrong column name)
  - medium: wrong JOIN conditions or WHERE clauses
  - hard:   wrong aggregations, GROUP BY, or subqueries

Reward tiers:
  0.0  — query fails to execute (syntax/runtime error)
  0.2  — executes but wrong number of columns
  0.4  — right columns, wrong number of rows
  0.7  — right shape but wrong cell values
  1.0  — perfect match with expected output
"""

import json
import random
import sqlite3
from pathlib import Path
from typing import Any, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from models import SqlAction, SqlObservation, SqlState
except ModuleNotFoundError:
    from envs.sql_env.models import SqlAction, SqlObservation, SqlState


# Path to task definitions relative to this file
_TASKS_PATH = Path(__file__).parent.parent / "data" / "tasks.json"


def _load_tasks() -> List[dict]:
    with open(_TASKS_PATH) as f:
        return json.load(f)


def _run_query(conn: sqlite3.Connection, query: str) -> Tuple[Optional[List[Tuple]], Optional[List[str]], Optional[str]]:
    """
    Execute a query and return (rows, column_names, error).
    On success error is None; on failure rows and column_names are None.
    """
    try:
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        col_names = [d[0] for d in cursor.description] if cursor.description else []
        return rows, col_names, None
    except sqlite3.Error as exc:
        return None, None, str(exc)


def _rows_to_str(rows: List[Tuple], col_names: List[str]) -> str:
    """Format query results as a readable string."""
    if not rows:
        return "(empty result)"
    header = " | ".join(col_names)
    separator = "-" * len(header)
    body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
    return f"{header}\n{separator}\n{body}"


def _grade(
    agent_rows: List[Tuple],
    agent_cols: List[str],
    expected_rows: List[Tuple],
    expected_cols: List[str],
) -> Tuple[float, str]:
    """
    Compute partial-credit score and feedback.

    Returns (score, feedback_string).
    """
    # Normalise column names to lowercase for comparison
    agent_cols_norm = [c.lower() for c in agent_cols]
    expected_cols_norm = [c.lower() for c in expected_cols]

    if set(agent_cols_norm) != set(expected_cols_norm):
        return 0.2, (
            f"Wrong columns returned. Expected {expected_cols_norm}, "
            f"got {agent_cols_norm}."
        )

    # Re-order agent rows to match expected column order for fair comparison
    if agent_cols_norm != expected_cols_norm:
        col_map = [agent_cols_norm.index(c) for c in expected_cols_norm]
        agent_rows = [tuple(row[i] for i in col_map) for row in agent_rows]

    if len(agent_rows) != len(expected_rows):
        return 0.4, (
            f"Right columns but wrong number of rows. "
            f"Expected {len(expected_rows)}, got {len(agent_rows)}."
        )

    # Compare row sets (order-insensitive)
    def normalise_row(row: Tuple) -> Tuple:
        return tuple(round(v, 6) if isinstance(v, float) else v for v in row)

    agent_set = sorted(normalise_row(r) for r in agent_rows)
    expected_set = sorted(normalise_row(r) for r in expected_rows)

    if agent_set == expected_set:
        return 1.0, "Perfect match! Query output matches expected result exactly."

    return 0.7, (
        "Right shape (columns and row count) but some values differ. "
        "Check your filter conditions, JOIN predicates, or aggregation logic."
    )


# ---------------------------------------------------------------------------
# Grader dispatch (one per difficulty so they can be tuned independently)
# ---------------------------------------------------------------------------

def grade_easy(agent_rows, agent_cols, expected_rows, expected_cols) -> Tuple[float, str]:
    return _grade(agent_rows, agent_cols, expected_rows, expected_cols)


def grade_medium(agent_rows, agent_cols, expected_rows, expected_cols) -> Tuple[float, str]:
    return _grade(agent_rows, agent_cols, expected_rows, expected_cols)


def grade_hard(agent_rows, agent_cols, expected_rows, expected_cols) -> Tuple[float, str]:
    return _grade(agent_rows, agent_cols, expected_rows, expected_cols)


_GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SqlEnvironment(Environment):
    """
    SQL Query Debugger RL environment.

    Each episode presents one broken SQL query. The agent submits a fixed
    query as its action; the environment executes it against an in-memory
    SQLite database and returns a partial-credit reward.

    Example:
        >>> env = SqlEnvironment()
        >>> obs = env.reset(seed=0)
        >>> print(obs.broken_query)
        >>> action = SqlAction(fixed_query="SELECT name FROM employees WHERE salary > 50000")
        >>> obs = env.step(action)
        >>> print(obs.reward, obs.feedback)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    MAX_STEPS: int = 5

    def __init__(self):
        super().__init__()
        self._tasks = _load_tasks()
        self._state = SqlState(episode_id=str(uuid4()), step_count=0)
        self._current_task: Optional[dict] = None
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_task(self, seed: Optional[int], difficulty: Optional[str] = None) -> dict:
        tasks = [t for t in self._tasks if t["difficulty"] == difficulty] if difficulty else self._tasks
        if not tasks:
            tasks = self._tasks
        if seed is not None:
            rng = random.Random(seed)
            return rng.choice(tasks)
        return random.choice(tasks)

    def _setup_db(self, task: dict) -> sqlite3.Connection:
        """Create an in-memory SQLite DB and populate it with task data."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(task["schema_sql"])
        conn.executescript(task["seed_data_sql"])
        conn.commit()
        return conn

    def _close_db(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        **kwargs: Any,
    ) -> SqlObservation:
        """
        Reset the environment: pick a task, set up the DB, return initial obs.

        Args:
            seed: Optional random seed for reproducible task selection.
            episode_id: Optional custom episode identifier.
            difficulty: Optional difficulty filter (easy, medium, hard).
        """
        self._close_db()
        task = self._pick_task(seed, difficulty)
        self._current_task = task
        self._conn = self._setup_db(task)

        self._state = SqlState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            current_task_id=task["task_id"],
            current_difficulty=task["difficulty"],
            total_reward=0.0,
            max_steps=self.MAX_STEPS,
        )

        return SqlObservation(
            task_id=task["task_id"],
            difficulty=task["difficulty"],
            broken_query=task["broken_query"],
            schema_description=task["schema_description"],
            expected_description=task["expected_description"],
            query_result="",
            done=False,
            reward=0.0,
            success=False,
            feedback="Environment ready. Submit your fixed SQL query.",
            attempts_remaining=self.MAX_STEPS,
        )

    def step(
        self,
        action: SqlAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SqlObservation:
        """
        Execute the agent's fixed query and grade it.

        Args:
            action: SqlAction containing the agent's corrected SQL query.

        Returns:
            SqlObservation with reward, feedback, and query results.
        """
        if self._current_task is None or self._conn is None:
            return SqlObservation(
                done=True,
                reward=0.0,
                success=False,
                feedback="Environment not initialised. Call reset() first.",
                attempts_remaining=0,
            )

        self._state.step_count += 1
        attempts_remaining = max(0, self.MAX_STEPS - self._state.step_count)
        task = self._current_task

        # Execute agent query
        agent_rows, agent_cols, agent_error = _run_query(self._conn, action.fixed_query)

        if agent_error is not None:
            _reward = max(0.01, min(0.99, 0.0))
            self._state.total_reward += _reward
            return SqlObservation(
                task_id=task["task_id"],
                difficulty=task["difficulty"],
                broken_query=task["broken_query"],
                schema_description=task["schema_description"],
                expected_description=task["expected_description"],
                query_result=f"ERROR: {agent_error}",
                done=True,
                reward=_reward,
                success=False,
                feedback=f"Query failed to execute: {agent_error}",
                attempts_remaining=attempts_remaining,
            )

        # Execute expected query to get ground truth
        expected_rows, expected_cols, expected_error = _run_query(self._conn, task["expected_query"])

        if expected_error is not None:
            _reward = max(0.01, min(0.99, 0.0))
            return SqlObservation(
                task_id=task["task_id"],
                difficulty=task["difficulty"],
                broken_query=task["broken_query"],
                schema_description=task["schema_description"],
                expected_description=task["expected_description"],
                query_result=_rows_to_str(agent_rows, agent_cols),
                done=True,
                reward=_reward,
                success=False,
                feedback=f"Internal error: expected query failed ({expected_error})",
                attempts_remaining=attempts_remaining,
            )

        grader = _GRADERS.get(task["difficulty"], _grade)
        score, feedback = grader(agent_rows, agent_cols, expected_rows, expected_cols)
        score = max(0.01, min(0.99, score))

        self._state.total_reward += score

        # End episode on perfect score or when max steps reached
        done = (score >= 0.99) or (self._state.step_count >= self.MAX_STEPS)
        if self._state.step_count >= self.MAX_STEPS and score < 0.99:
            feedback = f"Maximum attempts reached. Episode ended. {feedback}"

        return SqlObservation(
            task_id=task["task_id"],
            difficulty=task["difficulty"],
            broken_query=task["broken_query"],
            schema_description=task["schema_description"],
            expected_description=task["expected_description"],
            query_result=_rows_to_str(agent_rows, agent_cols),
            done=done,
            reward=score,
            success=(score >= 0.99),
            feedback=feedback,
            attempts_remaining=attempts_remaining,
        )

    @property
    def state(self) -> SqlState:
        return self._state
