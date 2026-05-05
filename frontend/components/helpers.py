"""Helpers de classificação e selectors compartilhados pelas páginas."""

from __future__ import annotations

from qdrant_client import QdrantClient

from frontend.components.di import get_provisioner

LOGICAL_PREFIX = "[L] "
LEGACY_PREFIX = "[Legacy] "


def classify_collections(client: QdrantClient) -> tuple[list[dict], list[str]]:
    """Separa colls em 2 grupos: lógicas (com info de strategies) e legadas.

    Cada `logical_dict` contém: `logical`, `strategies`, `has_summary`, `config`
    (lido de `_LOGICAL_CONFIG_` quando disponível).
    """
    prov = get_provisioner(client)
    logicals = prov.list_logical()
    legacy = prov.list_legacy()
    logical_dicts: list[dict] = []
    for li in logicals:
        cfg = prov.read_logical_config(li.logical) or {}
        logical_dicts.append(
            {
                "logical": li.logical,
                "strategies": li.strategies,
                "has_summary": li.has_summary,
                "config": cfg,
            }
        )
    return logical_dicts, legacy


def build_selector_options(logicals: list[dict], legacies: list[str]) -> list[str]:
    return [LOGICAL_PREFIX + l["logical"] for l in logicals] + [
        LEGACY_PREFIX + name for name in legacies
    ]


def parse_selector(label: str) -> tuple[str, str]:
    """Devolve `(kind, name)` onde `kind` ∈ {"logical", "legacy"}."""
    if label.startswith(LOGICAL_PREFIX):
        return "logical", label[len(LOGICAL_PREFIX) :]
    if label.startswith(LEGACY_PREFIX):
        return "legacy", label[len(LEGACY_PREFIX) :]
    return "legacy", label  # fallback
