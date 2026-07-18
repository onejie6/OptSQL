"""Self-reflection module interface."""

from myTypes import ReflectionMemory
from myTypes import RuntimeState


class SelfReflectionModule:
    """Reflect on failed or divergent runs and suggest recovery actions."""

    def should_reflect(self, state: RuntimeState) -> bool:
        raise NotImplementedError

    def diagnose_failure(self, state: RuntimeState) -> dict:
        raise NotImplementedError

    def propose_recovery(self, state: RuntimeState, diagnosis: dict) -> dict:
        raise NotImplementedError

    def update_memory(
        self,
        memory: ReflectionMemory,
        diagnosis: dict,
        recovery_plan: dict,
    ) -> ReflectionMemory:
        raise NotImplementedError
