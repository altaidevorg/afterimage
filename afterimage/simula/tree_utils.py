"""Helpers for taxonomy paths and validation."""

from __future__ import annotations

from .types import FactorTaxonomy


def path_from_root(t: FactorTaxonomy, node_id: str) -> list[str]:
    """Return node ids from root to node_id (inclusive)."""
    if node_id not in t.nodes:
        raise KeyError(node_id)
    out: list[str] = []
    cur: str | None = node_id
    while cur is not None:
        out.append(cur)
        cur = t.nodes[cur].parent_id
    out.reverse()
    if not out or out[0] != t.root_id:
        raise ValueError("Node not under declared root")
    return out
