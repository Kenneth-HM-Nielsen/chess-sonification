"""Dependency direction and layer ownership.

`ingest` reads, `features` measures, `tension` interprets, `render` sounds. A
module deciding something that belongs to another layer is a defect even when the
output is correct, so the boundaries are asserted rather than described.
"""

from __future__ import annotations

import pathlib
import unittest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"


def source(name: str) -> str:
    return (SRC / f"{name}.py").read_text(encoding="utf-8")


class LayerBoundaries(unittest.TestCase):
    def test_tension_reads_pressure_not_budget(self):
        """`budget` is an intermediate owned by features.

        Pressure being None and budget being None are not the same signal -- a
        known clock under an unknown time control yields a real budget and no
        pressure -- so tension takes the derived value and never the intermediate.
        """
        self.assertNotIn("budget", source("tension"))

    def test_tension_knows_nothing_about_audio(self):
        text = source("tension")
        for term in ("soundfile", "numpy", "scipy", "wave", "sample_rate",
                     "SAMPLE_RATE", "waveform"):
            with self.subTest(term=term):
                self.assertNotIn(term, text)

    def test_analysis_layers_do_not_import_downstream_ones(self):
        self.assertNotIn("from .tension", source("features"))
        self.assertNotIn("from .render", source("features"))
        self.assertNotIn("from .features", source("ingest"))
        self.assertNotIn("from .tension", source("ingest"))
        self.assertNotIn("from .render", source("tension"))

    def test_no_engine_anywhere(self):
        for name in ("ingest", "features", "tension", "render", "cli"):
            with self.subTest(module=name):
                self.assertNotIn("chess.engine", source(name))
                self.assertNotIn("stockfish", source(name).lower())


if __name__ == "__main__":
    unittest.main()
