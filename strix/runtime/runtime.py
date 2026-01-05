from abc import ABC, abstractmethod
from typing import TypedDict


class SandboxInfo(TypedDict):
    workspace_id: str
    api_url: str
    auth_token: str | None
    tool_server_port: int
    agent_id: str


class AbstractRuntime(ABC):
    @abstractmethod
    async def create_sandbox(
        self,
        agent_id: str,
        existing_token: str | None = None,
        local_sources: list[dict[str, str]] | None = None,
    ) -> SandboxInfo:
        raise NotImplementedError

    @abstractmethod
    async def get_sandbox_url(self, container_id: str, port: int) -> str:
        raise NotImplementedError

    @abstractmethod
    async def destroy_sandbox(self, container_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def restart_tool_server(self, container_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def check_tool_server_health(self, container_id: str, port: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def cleanup_resources(self, run_id: str | None = None) -> None:
        """
        Cleanup resources associated with the runtime.
        If run_id is provided, cleanup resources for that specific run.
        Otherwise, perform a general cleanup of stale resources.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_strix_containers(self, exclude_run_id: str | None = None) -> list[str]:
        """
        Get a list of active Strix container IDs/names.
        """
        raise NotImplementedError
