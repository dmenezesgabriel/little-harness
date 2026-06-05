# little-harness-logging

A [little-harness](https://github.com/dmenezesgabriel/little-harness)
**observer** plugin: it emits one structured JSON log record per agent
lifecycle event (run start/finish, model completion, tool invocation, repair),
each carrying the run's correlation id. It implements the core `AgentObserver`
port and registers a `logging` observer under the `little_harness.observers`
entry-point group.

```
uv pip install "little-harness[logging]"
little-harness --observer logging -p "What is 144 / 12?"   # or the --log shorthand
```

Records go to stderr, keeping stdout reserved for the plain-text answer.
