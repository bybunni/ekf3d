from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_integration_example_script_runs_and_emits_estimates() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "integration_example.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--steps",
            "4",
            "--dt-s",
            "0.5",
            "--seed",
            "1",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    output_lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert output_lines[0].startswith("timestamp_s, est_x_m, est_y_m")
    assert len(output_lines) == 1 + 4
