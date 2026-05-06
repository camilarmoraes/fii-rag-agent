"""Wrapper sobre `QdrantClient` com operações de alto nível em PointStruct.

Substitui a camada `langchain-qdrant` (`QdrantVectorStore`). Não tem nenhuma
dependência LangChain — vetores são `list[float]` e payloads são dicts (vindos
do `ChunkPayload.model_dump()` no caller).
"""

from __future__ import annotations

from typing import Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.models import (
    CollectionInfo,
    Distance,
    Filter,
    PointStruct,
    Record,
    ScoredPoint,
)

from server.src.fii_rag.store.schema import (
    DENSE_VECTOR_NAME,
    CollectionSchemaBuilder,
)


class QdrantRepository:
    def __init__(self, client: QdrantClient):
        self.client = client

    # ------------------------------------------------------------------
    # Collection-level
    # ------------------------------------------------------------------

    def collection_exists(self, name: str) -> bool:
        try:
            self.client.get_collection(name)
            return True
        except Exception:  # noqa: BLE001
            return False

    def get_collection_info(self, name: str) -> CollectionInfo:
        return self.client.get_collection(name)

    def list_collection_names(self) -> list[str]:
        return [c.name for c in self.client.get_collections().collections]

    def ensure_collection(
        self,
        name: str,
        dim: int,
        distance: Distance = Distance.COSINE,
        hybrid: bool = False,
    ) -> None:
        """Cria a coleção se não existir; no-op caso contrário."""
        if self.collection_exists(name):
            return
        kwargs: dict = {
            "collection_name": name,
            "vectors_config": CollectionSchemaBuilder.build_vectors_config(
                dim=dim, distance=distance, hybrid=hybrid
            ),
        }
        sparse = CollectionSchemaBuilder.build_sparse_vectors_config(hybrid=hybrid)
        if sparse is not None:
            kwargs["sparse_vectors_config"] = sparse
        self.client.create_collection(**kwargs)

    def delete_collection(self, name: str) -> None:
        self.client.delete_collection(name)

    def detect_dense_dim(self, name: str) -> Optional[int]:
        """Retorna a dimensão do vetor `dense` (named) ou do único vetor unnamed.

        Retorna `None` se a coleção não tem vetor identificável (ex: só sparse).
        """
        info = self.get_collection_info(name)
        cfg = info.config.params.vectors
        if isinstance(cfg, dict):
            if DENSE_VECTOR_NAME in cfg:
                return cfg[DENSE_VECTOR_NAME].size
            return next(iter(cfg.values())).size
        return getattr(cfg, "size", None)

    def is_named_vectors(self, name: str) -> bool:
        info = self.get_collection_info(name)
        return isinstance(info.config.params.vectors, dict)

    # ------------------------------------------------------------------
    # Point-level
    # ------------------------------------------------------------------

    def upsert_points(
        self, name: str, points: list[PointStruct], batch_size: int = 256
    ) -> None:
        if not points:
            return
        for i in range(0, len(points), batch_size):
            self.client.upsert(collection_name=name, points=points[i : i + batch_size])

    def query_dense(
        self,
        name: str,
        vector: list[float],
        limit: int = 10,
        using: Optional[str] = None,
        query_filter: Optional[Filter] = None,
        with_payload: bool = True,
    ) -> list[ScoredPoint]:
        """Busca densa simples.

        `using=None` quando a coleção tem vetor unnamed (legado).
        `using="dense"` quando a coleção tem named vectors (PR 3 com hybrid).
        """
        result = self.client.query_points(
            collection_name=name,
            query=vector,
            using=using,
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
        )
        return result.points

    def scroll(
        self,
        name: str,
        scroll_filter: Optional[Filter] = None,
        limit: int = 100,
        with_payload: bool = True,
        offset: Optional[Union[int, str]] = None,
    ) -> tuple[list[Record], Optional[Union[int, str]]]:
        records, next_offset = self.client.scroll(
            collection_name=name,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=with_payload,
            offset=offset,
        )
        return records, next_offset

    def count(self, name: str, count_filter: Optional[Filter] = None) -> int:
        return self.client.count(
            collection_name=name, count_filter=count_filter, exact=True
        ).count
