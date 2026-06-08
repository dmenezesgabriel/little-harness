# pyright: reportPrivateUsage=false
from __future__ import annotations

from little_harness.application.ports.permission_requester import PermissionRequester
from little_harness.presentation.cli.interactive_console import InteractiveRunner
from little_harness_rich.console import RichInteractiveConsole
from little_harness_rich.permission import RichPermissionRequester
from little_harness_rich.provider import build, build_permission_requester


class FakeApplication:
    def build_system_message(self) -> str:
        return "system"

    def run_turn(self, prompt: object, messages: object) -> tuple[object, object]:
        return "result", "updated"


class FakeCommandRegistry:
    pass


class TestBuild:
    def test_returns_a_runner_conforming_to_the_port(self) -> None:
        app = FakeApplication()
        registry = FakeCommandRegistry()

        # Act: build the runner
        # Under PEP 544, type checking with InteractiveRunner
        # asserts structural conformance.
        runner: InteractiveRunner = build(app, registry)  # type: ignore[arg-type]

        # Assert: it has a start method
        assert hasattr(runner, "start")
        assert callable(runner.start)

        # Verify that dependencies are correctly passed and stored
        assert isinstance(runner, RichInteractiveConsole)
        assert runner._app is app
        assert runner._registry is registry


class TestBuildPermissionRequester:
    def test_returns_a_rich_permission_requester(self) -> None:

        # Act
        requester: PermissionRequester = build_permission_requester()

        # Assert
        assert hasattr(requester, "request_approval")
        assert callable(requester.request_approval)
        assert isinstance(requester, RichPermissionRequester)
