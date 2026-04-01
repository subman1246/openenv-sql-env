"""Manual test script for SqlEnv — connects to ws://localhost:8000."""

import asyncio
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "envs")

from sql_env.client import SqlEnv
from sql_env.models import SqlAction


async def main() -> None:
    async with SqlEnv(base_url="http://localhost:8000") as env:
        # 1. Reset
        reset_result = await env.reset(seed=42)
        obs = reset_result.observation
        print("=== reset() ===")
        print(f"  task_id:            {obs.task_id}")
        print(f"  difficulty:         {obs.difficulty}")
        print(f"  broken_query:       {obs.broken_query}")
        print(f"  schema_description:\n{obs.schema_description}")

        # 2. Step with the corrected query for easy_001
        action = SqlAction(fixed_query="SELECT name FROM employees WHERE salary > 50000")
        step_result = await env.step(action)
        sobs = step_result.observation
        print("\n=== step() ===")
        print(f"  reward:   {step_result.reward}")
        print(f"  done:     {step_result.done}")
        print(f"  success:  {sobs.success}")
        print(f"  feedback: {sobs.feedback}")

        # 3. State
        state = await env.state()
        print("\n=== state() ===")
        print(f"  step_count: {state.step_count}")
        print(f"  episode_id: {state.episode_id}")
        print(f"  total_reward: {state.total_reward}")


if __name__ == "__main__":
    asyncio.run(main())
