"""Factory cacheada de `QdrantClient`.

Compartilhar uma instância por URL evita proliferação de pools HTTP/gRPC quando
múltiplos serviços (provider, repository, frontend) abrem clientes em paralelo.
"""

from __future__ import annotations

from qdrant_client import QdrantClient


class QdrantClientFactory:
    _instances: dict[str, QdrantClient] = {}

    @classmethod
    def get(cls, url: str) -> QdrantClient:
        if url not in cls._instances:
            cls._instances[url] = QdrantClient(url=url)
        return cls._instances[url]

    @classmethod
    def reset(cls) -> None:
        for client in cls._instances.values():
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
        cls._instances.clear()
