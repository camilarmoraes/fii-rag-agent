"""Convenção de naming `<logical>__<strategy>` para sub-collections.

PR 2 ainda não usa sub-collections — multi-strategy chega no PR 4 — mas o
helper já consolida regex e separator para evitar mágica espalhada.
"""

from __future__ import annotations

import re
from typing import Optional

LOGICAL_NAME_RE = re.compile(r"^[a-zA-Z0-9_]{1,40}$")
PHYSICAL_SUFFIX_SEPARATOR = "__"


class CollectionNaming:
    @staticmethod
    def is_valid_logical(name: str) -> bool:
        return bool(LOGICAL_NAME_RE.match(name))

    @staticmethod
    def to_physical(logical: str, strategy: str) -> str:
        return f"{logical}{PHYSICAL_SUFFIX_SEPARATOR}{strategy}"

    @staticmethod
    def to_logical(physical: str) -> Optional[tuple[str, str]]:
        """Retorna `(logical, strategy)` se `physical` segue a convenção, senão None."""
        if PHYSICAL_SUFFIX_SEPARATOR not in physical:
            return None
        logical, _, strategy = physical.rpartition(PHYSICAL_SUFFIX_SEPARATOR)
        if not logical or not strategy:
            return None
        return logical, strategy

    @staticmethod
    def is_physical_subcollection(name: str) -> bool:
        return CollectionNaming.to_logical(name) is not None
