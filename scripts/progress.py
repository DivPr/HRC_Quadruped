#!/usr/bin/env python3
"""Show which onboarding stages are done -- the thing you paste into SUBMISSION.md.

    uv run python scripts/progress.py            # CPU stages, skips slow training
    uv run python scripts/progress.py --slow     # also runs the Stage 4 PPO smoke test

Runs the test suite, then prints per-stage status plus the TODO(student) blocks
you have not filled in yet. Stage 5 is not covered here: it runs inside the ROS 2
container (`tests/test_05_ros2_smoke.sh`).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROGRESS_PATH = REPO_ROOT / ".pup_progress.json"
TODO_START = re.compile(r"#\s*=====\s*TODO\(student\):\s*(.+?)\s*=====")
TODO_END = re.compile(r"#\s*=====\s*end TODO\s*=====")
STAGE_OF_PATH = (
    ("pup/sim/", "test_01"),
    ("pup/jaxlab/", "test_02"),
    ("pup/envs/", "test_03"),
    ("pup/train/", "test_04"),
    ("pup/policy/", "test_04"),
    ("ros2_ws/", "test_05"),
)
LABELS = {
    "test_01": "Stage 1  MuJoCo + PD controller",
    "test_02": "Stage 2  JAX for robotics",
    "test_03": "Stage 3  MJX environment",
    "test_04": "Stage 4  Brax PPO + export",
    "test_05": "Stage 5  ROS 2 sim2sim",
}


def find_todos() -> dict[str, list[tuple[str, int, str]]]:
    """Return every *unfinished* ``TODO(student)`` block, grouped by stage key.

    A block counts as unfinished while its body still raises
    ``NotImplementedError``; filling it in makes it disappear from this list, so
    the count is a real progress bar rather than a census of the markers.
    """
    found: dict[str, list[tuple[str, int, str]]] = {key: [] for key in LABELS}
    skip_parts = {".venv", "__pycache__", "build", "install", "log", "tests", "scripts"}
    for path in sorted(REPO_ROOT.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if skip_parts & set(relative.split("/")[:-1]) or relative.startswith("."):
            continue
        stage = next((key for prefix, key in STAGE_OF_PATH if relative.startswith(prefix)), None)
        if stage is None:
            continue
        lines = path.read_text().splitlines()
        for number, line in enumerate(lines, start=1):
            match = TODO_START.search(line)
            if match is None:
                continue
            body: list[str] = []
            for following in lines[number:]:
                if TODO_END.search(following):
                    break
                body.append(following)
            if any("NotImplementedError" in text for text in body):
                found[stage].append((relative, number, match.group(1)))
    return found


def run_tests(include_slow: bool) -> None:
    """Run pytest so that conftest.py refreshes .pup_progress.json."""
    marks = "not gpu and not ros" if include_slow else "not gpu and not ros and not slow"
    command = [sys.executable, "-m", "pytest", "-q", "--tb=no", "-m", marks]
    print("$ " + " ".join(command) + "\n", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=False)


def main() -> int:
    """Run the suite and print the stage checklist; returns 0 unless a stage failed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slow", action="store_true", help="include the Stage 4 PPO smoke test")
    parser.add_argument("--no-run", action="store_true", help="reuse the last pytest results")
    arguments = parser.parse_args()
    if not arguments.no_run:
        run_tests(arguments.slow)
    if not PROGRESS_PATH.exists():
        print("no test results found; run pytest first")
        return 1

    data = json.loads(PROGRESS_PATH.read_text())
    stages = data["stages"]
    todos = find_todos()
    failed_any = False

    print("\n" + "=" * 78)
    print("PUP ONBOARDING PROGRESS")
    print("=" * 78)
    for key, label in LABELS.items():
        counts = Counter(stages.get(key, {}))
        passed, failed = counts["passed"], counts["failed"]
        not_started, remaining = counts["not_started"], len(todos[key])
        if key == "test_05":
            status = "◻ CONTAINER"
            detail = "run tests/test_05_ros2_smoke.sh inside docker/ros2"
        elif not (passed or failed or not_started):
            status, detail = "◻ NO TESTS", "-"
            continue
        elif failed:
            status, detail = "✗ FAILED", f"{passed} passing, {failed} failing"
            failed_any = True
        elif not_started and not passed:
            status, detail = "▷ NOT STARTED", f"{not_started} tests waiting"
        elif not_started:
            status, detail = "◑ IN PROGRESS", f"{passed} passing, {not_started} not started"
        else:
            status, detail = "✓ PASSED", f"{passed} passing"
        print(f"{status:<16} {label:<34} {detail}")
        if remaining:
            print(f"{'':<16} {remaining} TODO(student) block(s) still unwritten:")
            for relative, number, description in todos[key]:
                print(f"{'':<18} {relative}:{number}  {description}")
    print("=" * 78)
    total = sum(len(items) for items in todos.values())
    if total:
        print(f"{total} TODO(student) block(s) still unwritten.")
    else:
        print("Every TODO(student) block is written. Nice.")
    return 1 if failed_any else 0


if __name__ == "__main__":
    raise SystemExit(main())
