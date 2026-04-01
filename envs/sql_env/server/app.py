# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the SQL Query Debugger environment.

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Or via uv:
    uv run --project . server
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    ) from e

try:
    from models import SqlAction, SqlObservation
except ModuleNotFoundError:
    from envs.sql_env.models import SqlAction, SqlObservation

try:
    from server.sql_environment import SqlEnvironment
except ModuleNotFoundError:
    from envs.sql_env.server.sql_environment import SqlEnvironment


app = create_app(
    SqlEnvironment,
    SqlAction,
    SqlObservation,
    env_name="sql_env",
    max_concurrent_envs=10,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
