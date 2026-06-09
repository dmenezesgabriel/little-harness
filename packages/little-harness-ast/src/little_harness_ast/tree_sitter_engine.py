"""Adapter mapping py-tree-sitter onto the `SyntaxEngine` port.

This is the only module that imports tree-sitter (the runtime and the grammar
packages). Queries must capture their target as `@match`; those captured nodes
are what the grep and edit tools act on. A bad query or unknown language is
turned into a clean `ValueError` so callers never see a vendor-specific type.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import tree_sitter_javascript
import tree_sitter_python
from tree_sitter import Language, Node, Parser, Query, QueryCursor, QueryError

from little_harness_ast.structural_match import StructuralMatch

TARGET_CAPTURE = "match"

# Each language maps to its grammar's `language()` factory. Adding a language is
# a new dependency plus one entry here; the rest of the engine is generic.
LANGUAGE_FACTORIES: dict[str, Callable[[], object]] = {
    "python": tree_sitter_python.language,
    "javascript": tree_sitter_javascript.language,
}


class TreeSitterEngine:
    """Finds `@match` captures with tree-sitter, as vendor-free value objects.

    Example:
        TreeSitterEngine().find_matches("print(1)", "python", "(call) @match")

    """

    def find_matches(
        self, source: str, language: str, query: str
    ) -> Sequence[StructuralMatch]:
        """Return `@match` captures from running `query` on `source`."""
        ts_language = load_language(language)
        source_bytes = source.encode()
        tree = Parser(ts_language).parse(source_bytes)
        nodes = run_query(ts_language, query, tree.root_node)
        return [to_match(node, source_bytes) for node in nodes]


def load_language(language: str) -> Language:
    """Load a tree-sitter ``Language`` by name, raising ``ValueError`` if unknown."""
    factory = LANGUAGE_FACTORIES.get(language)

    if factory is None:
        raise ValueError(
            f"Unsupported language: {language!r}. "
            f"Expected one of {sorted(LANGUAGE_FACTORIES)}."
        )

    return Language(factory())


def run_query(ts_language: Language, query: str, root: Node) -> Sequence[Node]:
    """Compile and execute a tree-sitter query, raising ``ValueError`` on bad syntax."""
    try:
        compiled = Query(ts_language, query)
    except QueryError as error:
        raise ValueError(
            f"Invalid query: {query!r}. Expected a valid tree-sitter query ({error})."
        ) from error

    captures = QueryCursor(compiled).captures(root)
    return captures.get(TARGET_CAPTURE, [])


def to_match(node: Node, source_bytes: bytes) -> StructuralMatch:
    """Convert a tree-sitter ``Node`` into a vendor-free ``StructuralMatch``."""
    text = source_bytes[node.start_byte : node.end_byte].decode()
    # tree-sitter rows are 0-based; tools and humans count lines from 1.
    return StructuralMatch(
        node.start_point.row + 1,
        node.end_point.row + 1,
        node.start_byte,
        node.end_byte,
        text,
    )
