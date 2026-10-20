"""Authenticated package deployment interfaces."""

from generalforecast.deployments.contracts import ServiceError

__all__ = [
    "Client",
    "ServiceError",
]


def __getattr__(name: str):
    """Delegates the obsolete SDK import to the authenticated versioned client."""
    if name == "Client":
        from generalforecast.client import Client

        return Client
    raise AttributeError(name)
