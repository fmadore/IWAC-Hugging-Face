"""Startup configuration shared by scripts and installed entry points.

Load the working-directory .env (or the workspace's .env) before constructing
defaults. Explicit process environment variables always win. Repository
getters remain lazy so a caller can intentionally change a target at runtime.
"""

import os

from dotenv import find_dotenv, load_dotenv
from .paths import workspace_root


def initialize_environment() -> None:
    candidate = os.getenv("IWAC_ENV_FILE") or find_dotenv(usecwd=True)
    if not candidate:
        candidate = str(workspace_root() / ".env")
    load_dotenv(candidate, override=False)


initialize_environment()


def get_private_repo_id() -> str:
    return os.getenv("IWAC_HF_PRIVATE_REPO", "fmadore/islam-west-africa-collection-full")


def get_public_repo_id() -> str:
    return os.getenv("IWAC_HF_PUBLIC_REPO", "fmadore/islam-west-africa-collection")
