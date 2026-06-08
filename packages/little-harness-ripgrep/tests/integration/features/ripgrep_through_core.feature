Feature: The grep tool runs through the agent core
  The tool is discovered via its entry point and executed by the real agent
  loop, so a scripted model's tool call drives a real Python grep search.
  No external binary is required.

  Scenario: searching a directory returns the ripgrep action name
    When the agent uses grep with arguments "needle ."
    Then the run output shows action "ripgrep"

  Scenario: finding a real match returns succeeded output
    Given a file "hello.py" containing "needle_token_xyz"
    When the agent searches for "needle_token_xyz" in that file
    Then the search succeeds
    And the output contains "needle_token_xyz"

  Scenario: searching for an absent pattern reports no matches
    Given a file "empty.py" containing "nothing relevant here"
    When the agent searches for "MISSING_PATTERN_ABC_789" in that file
    Then the search succeeds
    And the output contains "No matches found."

  Scenario: an invalid regex fails gracefully
    Given a file "any.py" containing "some content"
    When the agent searches with invalid regex "(unclosed_paren" in that file
    Then the search fails
