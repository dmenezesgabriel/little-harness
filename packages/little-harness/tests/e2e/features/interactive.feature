Feature: Interactive REPL mode drives the full agent stack
  The same build_application and agent loop as one-shot, but via
  InteractiveConsole with stdin/stdout I/O. The REPL path activates
  when --prompt is absent.

  Scenario: calculator via REPL
    When I run the repl with prompts
      """
      What is 144 divided by 12?
      """
    Then the repl output contains "12"

  Scenario: read_file via REPL
    Given a workspace file "note.txt" containing "the secret word is plum"
    When I run the repl with prompts
      """
      Call exactly the read_file tool once with input note.txt. Then answer with the file contents.
      """
    Then the repl output contains "plum"

  Scenario: /help slash command
    When I run the repl with prompts
      """
      /help
      """
    Then the repl output contains "/exit"
    Then the repl output contains "/clear"
    Then the repl output contains "/skill"

  Scenario: /skill lists loaded skills
    Given a workspace file ".agents/skills/python/SKILL.md" with text
      """
      ---
      name: python
      description: Python expertise
      ---

      # Python

      Content.
      """
    When I run the repl with prompts
      """
      /skill
      """
    Then the repl output contains "python"

  Scenario: /skill reload re-reads skills from disk
    Given a workspace file ".agents/skills/python/SKILL.md" with text
      """
      ---
      name: python
      description: Python expertise
      ---

      # Python

      Content.
      """
    When I run the repl with prompts
      """
      /skill
      /skill reload
      /skill
      """
    Then the repl output contains "python"
