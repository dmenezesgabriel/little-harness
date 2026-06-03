Feature: Local agent answers arithmetic questions
  The agent runs the real local llama.cpp model end to end, exercising the full
  stack (composition root, streaming chat model, JSON policy, calculator tool).

  Scenario: Agent answers a division question
    Given a local agent
    When I ask "What is 144 divided by 12? Reply with the number."
    Then the answer contains "12"

  Scenario: Agent answers an addition question
    Given a local agent
    When I ask "What is 2 plus 2? Reply with the number."
    Then the answer contains "4"
