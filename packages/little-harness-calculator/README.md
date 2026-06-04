# little-harness-calculator

A [little-harness](https://github.com/dmenezesgabriel/little-harness) tool plugin
that evaluates arithmetic expressions safely (no `eval`: it parses to an AST and
evaluates a small, explicit set of operators). It implements the core
`AgentTool` port and registers a `calculator` tool under the
`little_harness.tools` entry-point group, so the agent can call it to compute
results.

```
uv pip install "little-harness[calculator]"
```

Installed tools are discovered automatically into the agent's tool registry.
