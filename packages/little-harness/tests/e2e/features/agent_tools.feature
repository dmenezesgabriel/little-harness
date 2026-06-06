Feature: A real provider drives every installed tool through the agent core
  Each scenario enables exactly one tool and asks a real model to use it, so the
  whole stack runs end to end: composition root, streaming chat model, JSON policy,
  approval hook, and the tool plugin. The same scenarios run under both providers
  (local llama.cpp and remote Gemini via litellm).

  Scenario: read_file surfaces a file's contents
    Given a workspace file "note.txt" containing "the secret word is plum"
    When the agent is asked to read "note.txt"
    Then the answer contains "plum"

  Scenario: write_file creates a file on disk
    When the agent is asked to write "written by the agent" into "out.txt"
    Then the workspace file "out.txt" contains "written by the agent"

  Scenario: edit_file replaces text in a file
    Given a workspace file "config.txt" containing "level = low"
    When the agent is asked to change "low" to "high" in "config.txt"
    Then the workspace file "config.txt" contains "level = high"

  Scenario: bash runs a shell command
    When the agent is asked to run a shell command printing "hello-from-bash"
    Then the answer contains "hello-from-bash"

  Scenario: calculator evaluates arithmetic
    When the agent is asked the arithmetic question "What is 144 divided by 12?"
    Then the answer contains "12"

  Scenario: ripgrep finds a match in the workspace
    Given a workspace file "haystack.txt" containing "find the needle in here"
    When the agent is asked to search the workspace for "needle"
    Then the answer contains "needle"

  Scenario: ast_grep finds a structural match
    Given a workspace file "sample.py" containing "print('hi')"
    When the agent is asked to find print calls in the Python file "sample.py"
    Then the answer contains "print"

  Scenario: ast_edit rewrites a node on disk
    Given a workspace file "greet.py" containing "def greet(): return 1"
    When the agent is asked to rename the Python function "greet" to "salute" in "greet.py"
    Then the workspace file "greet.py" contains "salute"
