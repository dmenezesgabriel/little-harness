Feature: Filesystem tools run through the agent core
  Each tool is discovered via its entry point and executed by the real agent
  loop, so a scripted model's tool call produces the tool's real effect.

  Scenario: reading a file
    Given a file "note.txt" containing "hello from disk"
    When the agent uses read_file on "note.txt"
    Then the run output contains "hello from disk"
    And the run output shows action "read_file"

  Scenario: writing a file
    When the agent uses write_file to write "written by agent" to "out.txt"
    Then the file "out.txt" contains "written by agent"
    And the run output shows action "write_file"

  Scenario: editing a file
    Given a file "app.py" containing "x = 1"
    When the agent uses edit_file to replace "x = 1" with "x = 2" in "app.py"
    Then the file "app.py" contains "x = 2"
    And the run output shows action "edit_file"

  Scenario: running a shell command
    When the agent uses bash to run "echo integration-ok"
    Then the run output contains "integration-ok"
    And the run output shows action "bash"
