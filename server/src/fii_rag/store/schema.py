"""Construção do `vectors_config` (e `sparse_vectors_config`) das collections.

PR 2 cria collections single-vector unnamed (compat com colls legadas criadas
pelo `QdrantVectorStore`). PR 3 ativa hybrid: vetor named `dense` + sparse
named `sparse`.
"""

from __future__ import annotations

from typing import Optional, Union

from qdrant_client.models import Distance, SparseVectorParams, VectorParams

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class CollectionSchemaBuilder:
    @staticmethod
    def build_vectors_config(
        dim: int, distance: Distance = Distance.COSINE, hybrid: bool = False
    ) -> Union[VectorParams, dict[str, VectorParams]]:
        params = VectorParams(size=dim, distance=distance)
        return {DENSE_VECTOR_NAME: params} if hybrid else params

    @staticmethod
    def build_sparse_vectors_config(
        hybrid: bool = False,
    ) -> Optional[dict[str, SparseVectorParams]]:
        if not hybrid:
            return None
        return {SPARSE_VECTOR_NAME: SparseVectorParams()}
