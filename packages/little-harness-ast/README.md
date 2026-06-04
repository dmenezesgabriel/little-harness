# little-harness-ast

Syntax-tree (AST) search and **edit** tool plugins for
[little-harness](https://github.com/dmenezesgabriel/little-harness), powered by
[py-tree-sitter](https://github.com/tree-sitter/py-tree-sitter).

| Tool | Input | Approval |
| --- | --- | --- |
| `ast_grep` | `{"path", "language", "query"}` | no (read-only) |
| `ast_edit` | `{"path", "language", "query", "replacement"}` | yes (writes) |

Both use [tree-sitter queries](https://tree-sitter.github.io/tree-sitter/using-parsers#query-syntax):
the node to act on is the one captured as **`@match`**. `ast_grep` lists every
`@match`; `ast_edit` replaces the *unique* `@match` node's exact bytes with the
replacement text — a structure-aware edit, not a textual one.

The tree-sitter runtime and grammar packages are wrapped behind a
`SyntaxEngine` port; the vendor import lives only in `TreeSitterEngine`.

## Languages

`python` and `javascript` are supported out of the box. Adding a language is a
new grammar dependency plus one entry in `LANGUAGE_FACTORIES`.

## Install

```bash
uv pip install "little-harness[ast]"
```

## Examples

Search for every call expression:

```json
{"path": "app.py", "language": "python", "query": "(call) @match"}
```

Replace a specific call:

```json
{"path": "app.py", "language": "python",
 "query": "(call function: (identifier) @_f (#eq? @_f \"print\")) @match",
 "replacement": "log()"}
```
