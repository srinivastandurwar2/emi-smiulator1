"""Run the test-suite. Uses pytest if installed, otherwise a small built-in runner.

    python scripts/run_tests.py
"""
import inspect
import itertools
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import pytest
    sys.exit(pytest.main(["-q", os.path.join(ROOT, "tests")]))
except ImportError:
    pass


class _Approx:
    """Minimal stand-in for pytest.approx."""

    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.e, self.rel, self.abs_tol = expected, rel, abs

    def __eq__(self, other):
        return abs(other - self.e) <= max(self.rel * abs(self.e), self.abs_tol)


class _Mark:
    @staticmethod
    def parametrize(names, values, ids=None):
        def deco(f):
            f.__dict__.setdefault("_grids", []).append(([n.strip() for n in names.split(",")], list(values)))
            return f
        return deco


class _Pytest:
    mark = _Mark()
    approx = _Approx


sys.modules["pytest"] = _Pytest()
import tests.test_physics as T  # noqa: E402

passed = failed = 0
for name, f in inspect.getmembers(T, inspect.isfunction):
    if not name.startswith("test_"):
        continue
    grids = getattr(f, "_grids", [])
    layers = []
    for names, values in grids:
        layers.append([dict(zip(names, v if len(names) > 1 else (v,))) for v in values])
    for combo in itertools.product(*layers) if layers else [()]:
        kw = {k: v for d in combo for k, v in d.items()}
        try:
            f(**kw)
            passed += 1
        except Exception:
            failed += 1
            print(f"FAIL {name} {kw}")
            traceback.print_exc()
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
