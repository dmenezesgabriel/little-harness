from __future__ import annotations

import pytest

from local_llm.domain.values.role import ASSISTANT, SYSTEM, USER, Role


class TestRole:
    def test_singletons_match_their_names(self) -> None:
        # Act / Assert
        assert SYSTEM.name == "system"
        assert USER.name == "user"
        assert ASSISTANT.name == "assistant"

    def test_equal_roles_share_identity_as_dict_keys(self) -> None:
        # Arrange: roles key per-role strategies, so equality + hashing matter.
        table = {SYSTEM: "s", USER: "u", ASSISTANT: "a"}

        # Act / Assert
        assert table[Role("user")] == "u"

    def test_rejects_an_unknown_role(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="Invalid role: tool"):
            Role("tool")
