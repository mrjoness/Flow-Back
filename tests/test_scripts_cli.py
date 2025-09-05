import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = [
    ("train.py", ["torch"]),
    ("pre_train.py", ["torch", "pytorch_lightning"]),
    ("post_train.py", ["torch", "pytorch_lightning", "mdtraj"]),
    ("eval.py", ["torch", "mdtraj", "sidechainnet"]),
]

@pytest.mark.parametrize("script,deps", SCRIPTS)
def test_script_help(script, deps):
    for mod in deps:
        pytest.importorskip(mod)
    script_path = Path(__file__).resolve().parents[1] / "src" / "scripts" / script
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    output = result.stdout.lower() + result.stderr.lower()
    assert "usage" in output
