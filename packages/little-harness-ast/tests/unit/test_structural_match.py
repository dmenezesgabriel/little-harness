from __future__ import annotations

from little_harness_ast.structural_match import StructuralMatch


class TestStructuralMatch:
    def test_locates_a_single_line_match(self) -> None:
        assert StructuralMatch(5, 5, 10, 20, "print(x)").location() == "line 5"

    def test_locates_a_multi_line_match(self) -> None:
        assert StructuralMatch(3, 7, 0, 40, "def f():\n    ...").location() == (
            "lines 3-7"
        )
