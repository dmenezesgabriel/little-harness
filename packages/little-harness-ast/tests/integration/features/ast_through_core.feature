Feature: AST tools run through the agent core
  Both tools are discovered via their entry points and executed by the real
  agent loop, so a scripted model's tool call drives a real tree-sitter query.

  Scenario: searching the syntax tree
    Given a python file "app.py" containing "print('found me')"
    When the agent uses ast_grep on "app.py" with query "(call) @match"
    Then the run output contains "print('found me')"
    And the run output shows action "ast_grep"

  Scenario: editing the syntax tree
    Given a python file "app.py" containing "print('old')"
    When the agent uses ast_edit on "app.py" with query "(call) @match" and replacement "log('new')"
    Then the file "app.py" contains "log('new')"
    And the run output shows action "ast_edit"
