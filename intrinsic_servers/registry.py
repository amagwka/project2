from __future__ import annotations
from typing import Callable, Dict, Type

from .base import BaseUDPIntrinsicServer

# Registry mapping names to server classes
INTRINSIC_SERVER_REGISTRY: Dict[str, Type[BaseUDPIntrinsicServer]] = {}


def register(name: str) -> Callable[[Type[BaseUDPIntrinsicServer]], Type[BaseUDPIntrinsicServer]]:
    def wrapper(cls: Type[BaseUDPIntrinsicServer]) -> Type[BaseUDPIntrinsicServer]:
        INTRINSIC_SERVER_REGISTRY[name] = cls
        return cls
    return wrapper


def get_server(name: str) -> Type[BaseUDPIntrinsicServer]:
    if name not in INTRINSIC_SERVER_REGISTRY:
        raise KeyError(f"Unknown intrinsic server: {name}")
    return INTRINSIC_SERVER_REGISTRY[name]
