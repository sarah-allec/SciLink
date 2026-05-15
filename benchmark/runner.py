"""Central CLI:  python -m benchmark.runner <test_name> [args]

Discovery + dispatch only; each test_*.py owns its own argparse so
the wire-up here stays tiny.

Examples:
    python -m benchmark.runner list                  # show available tests
    python -m benchmark.runner test_router --limit 5
    python -m benchmark.runner test_dft  --system licoo2
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path


def _available_tests() -> list[str]:
    here = Path(__file__).parent
    return sorted(
        m.name for m in pkgutil.iter_modules([str(here)])
        if m.name.startswith("test_")
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        print("\nAvailable tests:")
        for t in _available_tests():
            print(f"  {t}")
        return 0
    if argv[0] == "list":
        for t in _available_tests():
            print(t)
        return 0

    name = argv[0]
    if not name.startswith("test_"):
        name = "test_" + name
    if name not in _available_tests():
        print(f"unknown test: {name!r}", file=sys.stderr)
        print(f"available: {', '.join(_available_tests())}", file=sys.stderr)
        return 2

    mod = importlib.import_module(f"benchmark.{name}")
    if not hasattr(mod, "main"):
        print(f"{name} has no main(argv) entry point", file=sys.stderr)
        return 2
    return int(mod.main(argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
