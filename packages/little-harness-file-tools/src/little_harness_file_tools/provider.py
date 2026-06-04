"""Entry-point builders for the filesystem and shell tools.

Each builder is registered under the `little_harness.tools` group and returns one
ready `AgentTool`. The composition root calls the builder for every selected tool
and registers the result in the `ToolRegistry`.

Example:
    tool = build_read_file()
"""

from __future__ import annotations

from little_harness.application.ports.agent_tool import AgentTool

from little_harness_file_tools.bash_tool import BashTool
from little_harness_file_tools.command_guardrail import DangerousCommandGuardrail
from little_harness_file_tools.edit_file_tool import EditFileTool
from little_harness_file_tools.read_file_tool import ReadFileTool
from little_harness_file_tools.shell_command_runner import SubprocessShellRunner
from little_harness_file_tools.write_file_tool import WriteFileTool


def build_read_file() -> AgentTool:
    return ReadFileTool()


def build_write_file() -> AgentTool:
    return WriteFileTool()


def build_edit_file() -> AgentTool:
    return EditFileTool()


def build_bash() -> AgentTool:
    return BashTool(SubprocessShellRunner(), DangerousCommandGuardrail())
