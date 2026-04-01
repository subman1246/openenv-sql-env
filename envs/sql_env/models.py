# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the SQL Query Debugger environment.

The sql_env environment presents agents with broken SQL queries and grades
their fixes using partial-credit scoring against expected output.
"""

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class SqlAction(Action):
    """Action for the SQL environment — the agent's corrected SQL query."""

    fixed_query: str = Field(..., description="The agent's corrected SQL query")


class SqlObservation(Observation):
    """Observation from the SQL environment."""

    task_id: str = Field(default="", description="Unique identifier for the current task")
    difficulty: str = Field(default="", description="Task difficulty: easy, medium, or hard")
    broken_query: str = Field(default="", description="The broken SQL query the agent must fix")
    schema_description: str = Field(default="", description="Human-readable description of the database schema")
    expected_description: str = Field(default="", description="Plain English description of the expected result")
    query_result: str = Field(default="", description="The result of running the agent's query (empty on reset)")
    success: bool = Field(default=False, description="Whether the agent's query matched the expected output")
    feedback: str = Field(default="", description="Human-readable feedback on the agent's query")
    attempts_remaining: int = Field(default=5, description="Number of attempts remaining in this episode")


class SqlState(State):
    """State for the SQL environment."""

    current_task_id: str = Field(default="", description="ID of the current task")
    current_difficulty: str = Field(default="", description="Difficulty of the current task")
    total_reward: float = Field(default=0.0, description="Cumulative reward for the current episode")
    max_steps: int = Field(default=5, description="Maximum number of steps allowed per episode")
