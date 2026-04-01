# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SQL Query Debugger Environment Client.

Example:
    >>> from envs.sql_env.client import SqlEnv
    >>>
    >>> async with SqlEnv(base_url="http://localhost:8000") as env:
    ...     result = await env.reset(seed=42)
    ...     print(result.observation.broken_query)
    ...     from envs.sql_env.models import SqlAction
    ...     result = await env.step(SqlAction(fixed_query="SELECT name FROM employees WHERE salary > 50000"))
    ...     print(result.observation.reward, result.observation.feedback)

Sync usage:
    >>> env = SqlEnv(base_url="http://localhost:8000").sync()
    >>> with env:
    ...     result = env.reset(seed=42)
    ...     result = env.step(SqlAction(fixed_query="SELECT name FROM employees WHERE salary > 50000"))
    ...     print(result.observation.feedback)
"""

from typing import Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .models import SqlAction, SqlObservation, SqlState


class SqlEnv(EnvClient[SqlAction, SqlObservation, SqlState]):
    """
    Client for the SQL Query Debugger environment.

    Connects via WebSocket for persistent, low-latency sessions.
    Each instance maintains its own server-side environment session.
    """

    def _step_payload(self, action: SqlAction) -> Dict:
        return {"fixed_query": action.fixed_query}

    def _parse_result(self, payload: Dict) -> StepResult[SqlObservation]:
        obs_data = payload.get("observation", {})
        reward = payload.get("reward", 0.0)
        done = payload.get("done", False)
        observation = SqlObservation(
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", ""),
            broken_query=obs_data.get("broken_query", ""),
            schema_description=obs_data.get("schema_description", ""),
            expected_description=obs_data.get("expected_description", ""),
            query_result=obs_data.get("query_result", ""),
            done=done,
            reward=reward,
            success=obs_data.get("success", False),
            feedback=obs_data.get("feedback", ""),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict) -> SqlState:
        return SqlState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_task_id=payload.get("current_task_id", ""),
            current_difficulty=payload.get("current_difficulty", ""),
            total_reward=payload.get("total_reward", 0.0),
        )
