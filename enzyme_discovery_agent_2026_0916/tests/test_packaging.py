"""Packaging config must stay in sync with the source tree.

The wheel renames the dated source dir to the stable import name via
``package-dir``, and lists its subpackages EXPLICITLY.  An explicit list is a
trap: a newly added subpackage is silently omitted from the wheel and only fails
at import time for an installed user, never in a source checkout.  These tests
close that gap, plus the equivalent gap for non-.py package data.

Offline and stdlib-only: this reads the declared configuration and compares it to
the tree, rather than shelling out a real build.
"""

from __future__ import annotations

import unittest
from pathlib import Path

try:  # tomllib is stdlib on 3.11+; requires-python allows 3.10
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_IMPORT_NAME = "enzyme_discovery_agent"


def _config() -> dict:
    with _PYPROJECT.open("rb") as f:
        return tomllib.load(f)


@unittest.skipIf(tomllib is None, "tomllib requires Python 3.11+")
class PackagingConfigTest(unittest.TestCase):
    def setUp(self):
        if not _PYPROJECT.is_file():
            self.skipTest(f"no pyproject.toml at {_PYPROJECT}")
        self.cfg = _config()
        self.setuptools = self.cfg.get("tool", {}).get("setuptools", {})
        src_map = self.setuptools.get("package-dir", {})
        self.src_dir = _REPO_ROOT / src_map.get(_IMPORT_NAME, "")
        if not self.src_dir.is_dir():
            self.skipTest(f"source dir {self.src_dir} not found")

    def _declared_packages(self) -> set[str]:
        return set(self.setuptools.get("packages", []))

    def _tree_packages(self) -> set[str]:
        """Import paths for every directory in the source tree holding __init__.py."""
        found = set()
        for init in self.src_dir.rglob("__init__.py"):
            rel = init.parent.relative_to(self.src_dir)
            parts = [p for p in rel.parts if p]
            if any(p in {"runs", "archive", "__pycache__", "tests"} for p in parts):
                continue
            found.add(".".join([_IMPORT_NAME, *parts]) if parts else _IMPORT_NAME)
        return found

    def test_build_backend_is_declared(self):
        """Without [build-system], the build falls back to legacy behaviour."""
        build = self.cfg.get("build-system", {})
        self.assertTrue(build.get("requires"), "pyproject must declare build-system.requires")
        self.assertTrue(build.get("build-backend"), "pyproject must declare a build-backend")

    def test_every_subpackage_in_the_tree_ships(self):
        missing = self._tree_packages() - self._declared_packages()
        self.assertFalse(
            missing,
            "these packages exist in the tree but are NOT in "
            f"[tool.setuptools].packages, so they would be missing from the wheel: "
            f"{sorted(missing)}",
        )

    def test_no_declared_package_is_stale(self):
        stale = self._declared_packages() - self._tree_packages()
        self.assertFalse(stale, f"declared but absent from the tree: {sorted(stale)}")

    def test_declared_package_data_exists(self):
        """Every package-data glob must match at least one real file."""
        patterns = (self.setuptools.get("package-data", {}) or {}).get(_IMPORT_NAME, [])
        self.assertTrue(patterns, "expected package-data for the agent package")
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                self.assertTrue(
                    list(self.src_dir.glob(pattern)),
                    f"package-data pattern {pattern!r} matches no file under {self.src_dir}",
                )

    def test_console_script_target_is_importable(self):
        """The CLI entry point must name a real module attribute."""
        scripts = self.cfg.get("project", {}).get("scripts", {})
        target = scripts.get("enzyme-discovery-agent")
        self.assertTrue(target, "expected an enzyme-discovery-agent console script")
        module_path, _, attr = target.partition(":")
        self.assertTrue(attr, f"entry point {target!r} names no attribute")

        module = __import__(module_path, fromlist=[attr])
        self.assertTrue(
            callable(getattr(module, attr, None)),
            f"entry point {target!r} does not resolve to a callable",
        )

    def test_readme_is_declared_and_present(self):
        readme = self.cfg.get("project", {}).get("readme")
        self.assertTrue(readme, "project.readme should be declared so the wheel carries a description")
        self.assertTrue((_REPO_ROOT / readme).is_file(), f"declared readme {readme!r} is missing")

    def test_pytest_norecursedirs_keeps_the_defaults_worth_keeping(self):
        """Overriding norecursedirs drops pytest's defaults; virtualenvs must stay excluded."""
        norecurse = self.cfg.get("tool", {}).get("pytest", {}).get("ini_options", {}).get(
            "norecursedirs", []
        )
        for required in (".*", "*.egg", "node_modules"):
            self.assertIn(
                required, norecurse,
                f"{required!r} is a pytest default that this override would otherwise drop",
            )


if __name__ == "__main__":
    unittest.main()
