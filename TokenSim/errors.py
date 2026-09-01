class TokenSimError(Exception):
    """Base class for TokenSim-specific failures."""


class ConfigurationError(TokenSimError):
    """Invalid configuration or CLI-derived configuration."""


class WorkloadValidationError(TokenSimError):
    """Invalid workload dataset, record, or workload option."""


class SimulationStateError(TokenSimError):
    """Internal simulator state is inconsistent."""


class OutOfBlocksError(SimulationStateError):
    """A KV cache tier has no available physical blocks."""


class TransferError(TokenSimError):
    """Swap or remote KV transfer cannot be completed."""
