"""Shared base interfaces for the multi-agent system."""

from myTypes import AgentRequest
from myTypes import AgentResponse


class BaseAgent:
    """Base interface for all agents."""

    name: str

    def run(self, request: AgentRequest) -> AgentResponse:
        raise NotImplementedError
