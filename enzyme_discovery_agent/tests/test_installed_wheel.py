"""Build and smoke-test a wheel outside the checkout, including runtime assets."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import pytest


def test_clean_wheel_contains_and_loads_runtime_knowledge(tmp_path):
    pytest.importorskip("setuptools")
    try:
        import tomllib
    except ImportError:
        pytest.skip("wheel smoke-test configuration reader requires Python 3.11+")
    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    stage = tmp_path / "source"
    stage.mkdir()
    shutil.copy2(root / "pyproject.toml", stage / "pyproject.toml")
    for name in config["tool"]["setuptools"]["packages"]:
        relative = Path(*name.split("."))
        destination = stage / relative
        destination.mkdir(parents=True, exist_ok=True)
        for source in (root / relative).glob("*.py"):
            shutil.copy2(source, destination / source.name)
    # Copy source assets independently of the configured globs. An omitted glob
    # must cause the resulting wheel test to fail rather than hide the omission.
    for name in config["tool"]["setuptools"]["package-dir"]:
        for folder, dirs, files in os.walk(root / name):
            dirs[:] = [d for d in dirs if d not in {"runs", "tests", "campaigns", "cofold", "archive", "results", "figures", "analysis", "integrated_campaigns", "__pycache__"} and not d.startswith(".")]
            for filename in files:
                source = Path(folder) / filename
                if source.suffix == ".json" or source.name == "README.md":
                    target = stage / source.relative_to(root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    result = subprocess.run([sys.executable, "-c", "from setuptools.build_meta import build_wheel; build_wheel('dist')"], cwd=stage, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    wheel = next((stage / "dist").glob("*.whl"))
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)
    smoke = """
import sys, json
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from structural_modeling_agent.developability.evidence import LiabilityKnowledge
from structural_modeling_agent.modeling_reasoning.reaction import ReactionCoordinates
assert isinstance(LiabilityKnowledge.load().records, list)
assert ReactionCoordinates.load().mechanisms
from structural_modeling_agent import __file__ as location
root = Path(location).parent
assert json.loads((root/'validation_contract.json').read_text())
assert json.loads((root/'knowledge/attack_angle_metric.json').read_text())['metric_id'] == 'inverse_distance_attack_angle_v1'
assert json.loads((root/'knowledge/mlmm_engine_options.json').read_text())['schema_version'] == 'mlmm-engine-options-1.0'
from structural_modeling_agent.bioemu2_adapter import prepare
from structural_modeling_agent.workflow_policy import TOOL_PURPOSES
assert TOOL_PURPOSES['dynamics']['tool'] == 'bioemu2'
assert TOOL_PURPOSES['de_novo_design']['recipient'] == 'enzyme_design_agent'
"""
    result = subprocess.run([sys.executable, "-I", "-c", smoke, str(installed)], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
