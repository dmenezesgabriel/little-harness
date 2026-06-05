# little-harness-json-policy

A [little-harness](https://github.com/dmenezesgabriel/little-harness) agent
**policy** plugin: the strict-JSON reasoning protocol that drives the model
through the reason–act loop. It implements the core `AgentPolicy` port and
registers a `json` policy under the `little_harness.agent_policies` entry-point
group, so the protocol can be swapped without editing core.

```
uv pip install "little-harness[json-policy]"
little-harness --policy json -p "What is 144 / 12?"
```

`--policy` defaults to the sole installed policy, so with only this plugin
installed it is selected automatically.
