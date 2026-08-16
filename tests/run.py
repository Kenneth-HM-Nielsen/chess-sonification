"""Test runner that reports skips instead of hiding them.

    python -m tests.run
    python -m tests.run --strict     # a skipped test is a failure

A suite that reports green on a fresh clone because its fixtures are absent is
worse than one that has no tests, so every skip is named on the way out and
`--strict` turns them into a non-zero exit for CI.
"""

from __future__ import annotations

import sys
import unittest


def main(argv: list[str]) -> int:
    strict = "--strict" in argv
    verbosity = 2 if "-v" in argv else 1

    suite = unittest.TestLoader().discover("tests", top_level_dir=".")
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)

    if result.skipped:
        print(f"\n{len(result.skipped)} test(s) skipped:")
        for test, reason in result.skipped:
            print(f"  - {test.id()}\n      {reason}")
        if strict:
            print("\n--strict: skipped tests count as failures.")
    elif strict:
        print("\n--strict: no tests were skipped.")

    passed = result.wasSuccessful() and not (strict and result.skipped)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
