"""Offline checks that the reader-facing docs still name real commands and flags.

These tests read code spans (fenced blocks and inline backticks) only, so prose
can change freely. They catch the drift that a refactor leaves behind: a
renamed stage, an update mode that no longer exists, a CI pin that no longer
matches the lock.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from iwac_pipeline.cli import ANALYSIS_MODULES, PROCESS_MODULES

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CLAUDE = ROOT / "CLAUDE.md"
DOCS = [README, CLAUDE, ROOT / "CONTRIBUTING.md", *sorted((ROOT / "docs").glob("*.md"))]

_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_INLINE = re.compile(r"`([^`\n]+)`")


def code_spans(path: Path) -> list[str]:
    """Contents of every fenced block and inline code span in a Markdown file."""
    text = path.read_text(encoding="utf-8")
    spans = _FENCE.findall(text)
    spans += _INLINE.findall(_FENCE.sub("", text))
    return spans


def _named(spans: list[str], command: str) -> set[str]:
    pattern = re.compile(rf"\b{re.escape(command)}\s+([a-z][a-z0-9-]*)")
    return {name for span in spans for name in pattern.findall(span)}


def _update_modes() -> tuple[str, ...]:
    """``UPDATE_MODES`` read from source, without importing the stage helpers."""
    tree = ast.parse((ROOT / "post-processing" / "_common.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "UPDATE_MODES" for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("UPDATE_MODES not found in post-processing/_common.py")


@pytest.mark.parametrize(
    ("command", "registry"),
    [("iwac-process", PROCESS_MODULES), ("iwac-analyze", ANALYSIS_MODULES)],
)
def test_readme_commands_exist(command, registry):
    named = _named(code_spans(README), command)
    unknown = named - set(registry)
    assert not unknown, f"README names {command} subcommands the CLI lacks: {sorted(unknown)}"


@pytest.mark.parametrize(
    ("command", "registry"),
    [("iwac-process", PROCESS_MODULES), ("iwac-analyze", ANALYSIS_MODULES)],
)
@pytest.mark.parametrize("doc", [README, CLAUDE], ids=lambda p: p.name)
def test_every_cli_name_is_documented(command, registry, doc):
    spans = code_spans(doc)
    mentioned = _named(spans, command) | {span.strip() for span in spans}
    missing = set(registry) - mentioned
    assert not missing, f"{doc.name} does not list {command} names: {sorted(missing)}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_update_mode_examples_use_valid_choices(doc):
    valid = set(_update_modes())
    used = {
        value
        for span in code_spans(doc)
        for value in re.findall(r"--update-mode[ =]([A-Za-z_]+)", span)
    }
    assert used <= valid, f"{doc.name} uses invalid --update-mode values: {sorted(used - valid)}"


def test_ci_ruff_pin_matches_lock():
    lock = re.search(r"^ruff==([0-9][^\s\\;]*)", (ROOT / "requirements-lock.txt").read_text(encoding="utf-8"), re.M)
    ci = re.findall(r"\bruff==([0-9][^\s\"']*)", (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    assert lock, "requirements-lock.txt does not pin ruff"
    assert ci, ".github/workflows/ci.yml does not pin ruff"
    assert set(ci) == {lock.group(1)}, f"CI installs ruff {ci}, the lock pins {lock.group(1)}"
