"""Dependency direction and layer ownership.

`ingest` reads, `features` measures, `tension` interprets, `render` sounds. A
module deciding something that belongs to another layer is a defect even when the
output is correct, so the boundaries are asserted rather than described.

Checks run over the parsed module rather than its text, so that a comment or
docstring explaining a rule does not count as breaking it, and so that a
different spelling of the same import does not slip through.
"""

from __future__ import annotations

import ast
import pathlib
import unittest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
MODULES = ("__init__", "ingest", "features", "tension", "render", "cli")

# ingest -> features -> tension -> render. Anything to the right is downstream.
ORDER = ("ingest", "features", "tension", "render")


def tree(name: str) -> ast.Module:
    return ast.parse((SRC / f"{name}.py").read_text(encoding="utf-8"))


def imported_names(name: str) -> set[str]:
    """Every module or symbol name the module imports, however it is spelled."""
    found: set[str] = set()
    for node in ast.walk(tree(name)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.update(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.update(node.module.split("."))
            found.update(alias.name for alias in node.names)
    return found


def _docstrings(module: ast.Module) -> set[int]:
    """ids of the string nodes that are docstrings, so they can be ignored."""
    ids = set()
    for node in ast.walk(module):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                ids.add(id(first.value))
    return ids


def referenced_tokens(name: str) -> set[str]:
    """Identifiers and string literals the code actually uses, lowercased.

    Docstrings are excluded: a module must be free to explain the rule it keeps.
    """
    module = tree(name)
    skip = _docstrings(module)
    tokens: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Name):
            tokens.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            tokens.add(node.attr.lower())
        elif isinstance(node, ast.arg):
            tokens.add(node.arg.lower())
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in skip:
                tokens.add(node.value.lower())
    return tokens


class LayerBoundaries(unittest.TestCase):
    def test_tension_reads_pressure_not_budget(self):
        """`budget` is an intermediate owned by features.

        Pressure being None and budget being None are not the same signal -- a
        known clock under an unknown time control yields a real budget and no
        pressure -- so tension takes the derived value and never the
        intermediate. A source scan cannot catch a computed key; the binding
        check is the behavioural one in Phase 3, which deletes the budget fields
        and requires the track to come out identical.
        """
        offenders = [t for t in referenced_tokens("tension") if "budget" in t]
        self.assertEqual(offenders, [])

    def test_a_docstring_may_still_explain_the_rule(self):
        """Guards the check itself: prose about budget must not fail the build."""
        module = ast.parse('"""We deliberately never read budget here."""\nx = 1\n')
        skip = _docstrings(module)
        strings = [n.value for n in ast.walk(module)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)
                   and id(n) not in skip]
        self.assertEqual(strings, [])

    def test_tension_knows_nothing_about_audio(self):
        tokens = referenced_tokens("tension") | imported_names("tension")
        for term in ("soundfile", "numpy", "scipy", "wave", "samplerate",
                     "sample_rate", "waveform", "audio", "hz"):
            with self.subTest(term=term):
                self.assertNotIn(term, {t.lower() for t in tokens})

    def test_no_module_imports_a_downstream_one(self):
        for index, name in enumerate(ORDER):
            imported = imported_names(name)
            for downstream in ORDER[index + 1:]:
                with self.subTest(module=name, downstream=downstream):
                    self.assertNotIn(downstream, imported)

    def test_no_engine_anywhere(self):
        for name in MODULES:
            with self.subTest(module=name):
                self.assertNotIn("engine", imported_names(name))
                self.assertNotIn("stockfish",
                                 {t.lower() for t in referenced_tokens(name)})


if __name__ == "__main__":
    unittest.main()
