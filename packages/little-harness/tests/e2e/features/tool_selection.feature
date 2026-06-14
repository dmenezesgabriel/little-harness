Feature: A real provider selects the correct tool from the full set
  Every installed tool is available (no --tools filter). The model must choose
  the right one for the job rather than using a general-purpose alternative
  such as bash. Each scenario mirrors the corresponding scenario in
  agent_tools.feature but without restricting the tool set.

  Scenario: read_file chosen when all tools are available
    Given a workspace file "note.txt" containing "the secret word is plum"
    When the agent with all tools is asked to read "note.txt"
    Then the answer contains "plum"

  Scenario: write_file chosen when all tools are available
    When the agent with all tools is asked to write "written by the agent" into "out.txt"
    Then the workspace file "out.txt" contains "written by the agent"

  Scenario: edit_file chosen when all tools are available
    Given a workspace file "config.txt" containing "level = low"
    When the agent with all tools is asked to change "low" to "high" in "config.txt"
    Then the workspace file "config.txt" contains "level = high"

  Scenario: bash chosen when all tools are available
    When the agent with all tools is asked to run a shell command printing "hello-from-bash"
    Then the answer contains "hello-from-bash"

  Scenario: calculator chosen when all tools are available
    When the agent with all tools is asked the arithmetic question "What is 144 divided by 12?"
    Then the answer contains "12"

  Scenario: ripgrep chosen when all tools are available
    Given a workspace file "haystack.txt" containing "find the needle in here"
    When the agent with all tools is asked to search the workspace for "needle"
    Then the answer contains "needle"

  Scenario: ripgrep finds a hidden file when all tools are available
    Given a workspace file ".hidden_haystack.txt" containing "find the hidden needle"
    When the agent with all tools is asked to search the workspace including hidden files for "hidden needle"
    Then the answer contains "hidden needle"

  Scenario: ast_grep chosen when all tools are available
    Given a workspace file "sample.py" containing "print('hi')"
    When the agent with all tools is asked to find print calls in the Python file "sample.py"
    Then the answer contains "print"

  Scenario: ast_edit chosen when all tools are available
    Given a workspace file "greet.py" containing "def greet(): return 1"
    When the agent with all tools is asked to rename the Python function "greet" to "salute" in "greet.py"
    Then the workspace file "greet.py" contains "salute"
