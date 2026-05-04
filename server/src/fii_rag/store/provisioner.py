"""Provisiona uma 'coleção lógica' como N físicas no Qdrant.

Naming: `<logical>__<strategy>` (separador `__`). A configuração da lógica
(strategies ativas, hybrid on/off, dim, distance, created_at) é gravada num
**ponto especial** dentro de `<logical>__summary` com:
    - id determinístico (uuid5 do nome lógico)
    - vector zero (compatível com o schema)
    - payload com `_kind = "config"` para ser excluído de queries normais
      via `Filter(must_not=[FieldCondition(key='_kind', match=MatchValue('config'))])`

Payload indexes criados em todas as físicas para suportar filtros eficientes
(doc_id, ticker, report_year, report_quarter, report_type, num_dy, num_pvp,
num_vacancia, num_patrimonio).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import NAMESPACE_DNS, uuid5

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from server.src.fii_rag.store.naming import CollectionNaming
from server.src.fii_rag.store.repository import QdrantRepository
from server.src.fii_rag.store.schema import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

LOGICAL_CONFIG_NAMESPACE = uuid5(NAMESPACE_DNS, "fii-rag-agent.logical-config")
LOGICAL_CONFIG_KIND = "config"
PAYLOAD_KIND_KEY = "_kind"

PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "doc_id": PayloadSchemaType.KEYWORD,
    "strategy": PayloadSchemaType.KEYWORD,
    "ticker": PayloadSchemaType.KEYWORD,
    "report_year": PayloadSchemaType.INTEGER,
    "report_quarter": PayloadSchemaType.INTEGER,
    "report_type": PayloadSchemaType.KEYWORD,
    "num_dy": PayloadSchemaType.FLOAT,
    "num_pvp": PayloadSchemaType.FLOAT,
    "num_vacancia": PayloadSchemaType.FLOAT,
    "num_patrimonio": PayloadSchemaType.FLOAT,
    PAYLOAD_KIND_KEY: PayloadSchemaType.KEYWORD,
}


@dataclass
class LogicalCollectionInfo:
    logical: str
    strategies: list[str]
    has_summary: bool


def exclude_config_filter() -> Filter:
    """Filter padrão que esconde o ponto `_kind=config` de queries comuns."""
    return Filter(
        must_not=[
            FieldCondition(
                key=PAYLOAD_KIND_KEY,
                match=MatchValue(value=LOGICAL_CONFIG_KIND),
            )
        ]
    )


class LogicalCollectionProvisioner:
    """Cria/gerencia coleções lógicas multi-estratégia."""

    def __init__(self, repository: QdrantRepository):
        self.repo = repository

    # ------------------------------------------------------------------
    # Provision / delete
    # ------------------------------------------------------------------

    def provision(
        self,
        logical: str,
        strategies: list[str],
        hybrid: bool = True,
        dense_dim: int = 768,
        distance: Distance = Distance.COSINE,
    ) -> dict:
        if not CollectionNaming.is_valid_logical(logical):
            raise ValueError(
                f"Nome lógico inválido: {logical!r}. "
                "Use apenas letras, dígitos e _, máx 40 chars."
            )

        # `summary` é obrigatório
        strategies = list(strategies)
        if "summary" not in strategies:
            strategies.insert(0, "summary")

        physical_names: list[str] = []
        for strategy in strategies:
            name = CollectionNaming.to_physical(logical, strategy)
            self._create_physical(name, dense_dim, distance, hybrid)
            self._create_payload_indexes(name)
            physical_names.append(name)

        config_data = {
            PAYLOAD_KIND_KEY: LOGICAL_CONFIG_KIND,
            "logical": logical,
            "strategies": list(strategies),
            "hybrid": hybrid,
            "dense_dim": dense_dim,
            "distance": str(distance),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_logical_config(logical, dense_dim, hybrid, config_data)
        return {"physical_names": physical_names, "config": config_data}

    def delete_logical(self, logical: str) -> int:
        """Apaga todas as físicas com prefixo `<logical>__`. Retorna a contagem."""
        count = 0
        for name in self.repo.list_collection_names():
            parsed = CollectionNaming.to_logical(name)
            if parsed and parsed[0] == logical:
                try:
                    self.repo.delete_collection(name)
                    count += 1
                except Exception as e:  # noqa: BLE001
                    print(f"[Provisioner] Aviso ao deletar {name!r}: {e}")
        return count

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_logical_config(self, logical: str) -> Optional[dict]:
        summary_coll = CollectionNaming.to_physical(logical, "summary")
        if not self.repo.collection_exists(summary_coll):
            return None
        records, _ = self.repo.scroll(
            summary_coll,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key=PAYLOAD_KIND_KEY,
                        match=MatchValue(value=LOGICAL_CONFIG_KIND),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not records:
            return None
        return dict(records[0].payload or {})

    def list_logical(self) -> list[LogicalCollectionInfo]:
        """Agrupa físicas por nome lógico via convenção `<logical>__<strategy>`."""
        groups: dict[str, list[str]] = {}
        for name in self.repo.list_collection_names():
            parsed = CollectionNaming.to_logical(name)
            if parsed is None:
                continue
            logical, strategy = parsed
            groups.setdefault(logical, []).append(strategy)

        return [
            LogicalCollectionInfo(
                logical=logical,
                strategies=sorted(strategies),
                has_summary="summary" in strategies,
            )
            for logical, strategies in sorted(groups.items())
        ]

    def list_legacy(self) -> list[str]:
        """Retorna colls que NÃO seguem a convenção `<logical>__<strategy>`."""
        return [
            name
            for name in self.repo.list_collection_names()
            if not CollectionNaming.is_physical_subcollection(name)
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _create_physical(
        self, name: str, dense_dim: int, distance: Distance, hybrid: bool
    ) -> None:
        if self.repo.collection_exists(name):
            return
        kwargs: dict = {
            "collection_name": name,
            "vectors_config": {
                DENSE_VECTOR_NAME: VectorParams(size=dense_dim, distance=distance)
            },
        }
        if hybrid:
            kwargs["sparse_vectors_config"] = {SPARSE_VECTOR_NAME: SparseVectorParams()}
        self.repo.client.create_collection(**kwargs)

    def _create_payload_indexes(self, collection_name: str) -> None:
        for field, schema in PAYLOAD_INDEXES.items():
            try:
                self.repo.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:  # noqa: BLE001
                # idempotente: já existe ou conflito → ignora
                pass

    def _write_logical_config(
        self, logical: str, dense_dim: int, hybrid: bool, payload: dict
    ) -> None:
        summary_coll = CollectionNaming.to_physical(logical, "summary")
        config_id = str(uuid5(LOGICAL_CONFIG_NAMESPACE, logical))
        zero_dense = [0.0] * dense_dim
        vector: dict
        if hybrid:
            vector = {
                DENSE_VECTOR_NAME: zero_dense,
                SPARSE_VECTOR_NAME: SparseVector(indices=[0], values=[0.0]),
            }
        else:
            vector = {DENSE_VECTOR_NAME: zero_dense}
        self.repo.client.upsert(
            collection_name=summary_coll,
            points=[PointStruct(id=config_id, vector=vector, payload=payload)],
        )
