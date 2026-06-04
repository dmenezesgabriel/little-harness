"""Entry-point builders for the tree-sitter AST tools.

Registered under the `little_harness.tools` group as `ast_grep` and `ast_edit`.
Both share a single tree-sitter engine implementation.

Example:
    tool = build_ast_grep()
"""

from __future__ import annotations

from little_harness.application.ports.agent_tool import AgentTool

from little_harness_ast.ast_edit_tool import AstEditTool
from little_harness_ast.ast_grep_tool import AstGrepTool
from little_harness_ast.tree_sitter_engine import TreeSitterEngine


def build_ast_grep() -> AgentTool:
    return AstGrepTool(TreeSitterEngine())


def build_ast_edit() -> AgentTool:
    return AstEditTool(TreeSitterEngine())
