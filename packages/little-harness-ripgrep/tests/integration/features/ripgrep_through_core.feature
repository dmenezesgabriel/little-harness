Feature: The ripgrep tool runs through the agent core
  The tool is discovered via its entry point and executed by the real agent
  loop, so a scripted model's tool call drives a real ripgrep search.

  Scenario: searching through the agent
    When the agent uses ripgrep with arguments "needle ."
    Then the run output shows action "ripgrep"
