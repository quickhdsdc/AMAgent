
class SandboxError(Exception):
    """Base exception for sandbox-related errors."""


class SandboxTimeoutError(SandboxError):
    """Exception raised when a sandbox operation times out."""


class SandboxResourceError(SandboxError):
    """Exception raised for resource-related errors."""
