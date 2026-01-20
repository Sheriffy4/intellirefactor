"""
Direct Neighbor Extractor for LLM Context Navigation.

Extracts direct dependencies (imports, calls) for a target file to build
an "allowed_files" list for LLM context generation.

This module provides efficient queries to find:
- Files that import the target
- Files that the target imports
- Files that call functions/methods in the target
- Files whose functions/methods are called by the target

Used by llm_context_generator to build precise navigation maps.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from .store import IndexStore

logger = logging.getLogger(__name__)


@dataclass
class NeighborFile:
    """Represents a neighboring file with relationship info."""

    file_path: str
    relationship_type: str  # "imports", "imported_by", "calls", "called_by"
    edge_count: int  # how many edges of this type
    symbols_involved: List[str]  # list of symbols involved in the relationship
    confidence: float  # average confidence of edges


@dataclass
class NeighborGraph:
    """Graph of direct neighbors for a target file."""

    target_file: str
    neighbors: List[NeighborFile]
    total_edges: int
    relationship_summary: Dict[str, int]  # type -> count


class NeighborExtractor:
    """
    Extracts direct neighbors from the index database.

    Provides efficient queries to build "allowed_files" lists for LLM context.
    """

    def __init__(self, index_store: IndexStore):
        """
        Initialize the NeighborExtractor.

        Args:
            index_store: IndexStore instance for database access
        """
        self.store = index_store
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def _connect(self):
        """
        Compatibility layer: IndexStore implementations differ.

        - Some versions expose get_connection()
        - Current IntelliRefactor IndexStore exposes _get_connection() (context manager)
        """
        if hasattr(self.store, "get_connection"):
            conn = self.store.get_connection()  # type: ignore[attr-defined]
            try:
                yield conn
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            return

        if hasattr(self.store, "_get_connection"):
            with self.store._get_connection() as conn:  # type: ignore[attr-defined]
                yield conn
            return

        raise AttributeError("IndexStore has neither get_connection() nor _get_connection()")

    def extract_neighbors(
        self,
        target_file: str,
        max_neighbors: int = 30,
        min_confidence: float = 0.5,
    ) -> NeighborGraph:
        """
        Extract direct neighbors for a target file.

        Args:
            target_file: Normalized file path (e.g., "src/module.py")
            max_neighbors: Maximum number of neighbors to return
            min_confidence: Minimum confidence threshold for edges

        Returns:
            NeighborGraph with sorted neighbors
        """
        target_file = self._normalize_path(target_file)

        try:
            with self._connect() as conn:
                neighbors = self._query_neighbors(conn, target_file, min_confidence)
        except Exception as e:
            self.logger.error("Failed to extract neighbors for %s: %s", target_file, e)
            return NeighborGraph(target_file=target_file, neighbors=[], total_edges=0, relationship_summary={})

        # Sort by edge_count descending, then by file_path
        neighbors.sort(key=lambda n: (-n.edge_count, n.file_path))

        # Limit to max_neighbors
        neighbors = neighbors[:max_neighbors]

        # Build summary
        summary = {}
        total_edges = 0
        for neighbor in neighbors:
            summary[neighbor.relationship_type] = summary.get(neighbor.relationship_type, 0) + neighbor.edge_count
            total_edges += neighbor.edge_count

        return NeighborGraph(
            target_file=target_file,
            neighbors=neighbors,
            total_edges=total_edges,
            relationship_summary=summary,
        )

    def _query_neighbors(
        self,
        conn,
        target_file: str,
        min_confidence: float,
    ) -> List[NeighborFile]:
        """
        Query database for direct neighbors.

        Returns list of NeighborFile objects.
        """
        neighbors_dict: Dict[Tuple[str, str], NeighborFile] = {}

        # Get target file_id
        cursor = conn.execute(
            "SELECT file_id FROM files WHERE file_path = ?",
            (target_file,),
        )
        result = cursor.fetchone()
        if not result:
            return []

        target_file_id = result[0]

        # Query 1: Files that import target (target is imported)
        # Find symbols in target_file, then find dependencies pointing to them
        self._query_imported_by(conn, target_file_id, min_confidence, neighbors_dict)

        # Query 2: Files that target imports (target imports them)
        # Find symbols in target_file, then find their dependencies
        self._query_imports(conn, target_file_id, min_confidence, neighbors_dict)

        # Query 3: Files that call functions in target
        self._query_called_by(conn, target_file_id, min_confidence, neighbors_dict)

        # Query 4: Files whose functions target calls
        self._query_calls(conn, target_file_id, min_confidence, neighbors_dict)

        return list(neighbors_dict.values())

    def _query_imported_by(
        self,
        conn,
        target_file_id: int,
        min_confidence: float,
        neighbors_dict: Dict[Tuple[str, str], NeighborFile],
    ) -> None:
        """Find files that import the target file."""
        query = """
        SELECT DISTINCT
            fsrc.file_path,
            COUNT(d.dependency_id) as edge_count,
            AVG(d.confidence) as avg_confidence,
            GROUP_CONCAT(DISTINCT stgt.name) as symbols
        FROM dependencies d
        JOIN symbols stgt ON d.target_symbol_id = stgt.symbol_id
        JOIN symbols ssrc ON d.source_symbol_id = ssrc.symbol_id
        JOIN files fsrc ON ssrc.file_id = fsrc.file_id
        WHERE stgt.file_id = ?
            AND d.dependency_kind IN ('imports', 'inherits')
            AND d.confidence >= ?
            AND fsrc.file_id != ?
        GROUP BY fsrc.file_path
        ORDER BY edge_count DESC
        """
        cursor = conn.execute(query, (target_file_id, min_confidence, target_file_id))

        for file_path, edge_count, avg_confidence, symbols_str in cursor.fetchall():
            file_path = self._normalize_path(file_path)
            symbols = [s.strip() for s in (symbols_str or "").split(",") if s.strip()]

            key = (file_path, "imported_by")
            if key not in neighbors_dict:
                neighbors_dict[key] = NeighborFile(
                    file_path=file_path,
                    relationship_type="imported_by",
                    edge_count=edge_count,
                    symbols_involved=symbols,
                    confidence=avg_confidence or 0.0,
                )

    def _query_imports(
        self,
        conn,
        target_file_id: int,
        min_confidence: float,
        neighbors_dict: Dict[Tuple[str, str], NeighborFile],
    ) -> None:
        """Find files that the target imports."""
        query = """
        SELECT DISTINCT
            ftgt.file_path,
            COUNT(d.dependency_id) as edge_count,
            AVG(d.confidence) as avg_confidence,
            GROUP_CONCAT(DISTINCT stgt.name) as symbols
        FROM dependencies d
        JOIN symbols ssrc ON d.source_symbol_id = ssrc.symbol_id
        JOIN symbols stgt ON d.target_symbol_id = stgt.symbol_id
        JOIN files ftgt ON stgt.file_id = ftgt.file_id
        WHERE ssrc.file_id = ?
            AND d.dependency_kind IN ('imports', 'inherits')
            AND d.confidence >= ?
            AND ftgt.file_id != ?
        GROUP BY ftgt.file_path
        ORDER BY edge_count DESC
        """
        cursor = conn.execute(query, (target_file_id, min_confidence, target_file_id))

        for file_path, edge_count, avg_confidence, symbols_str in cursor.fetchall():
            file_path = self._normalize_path(file_path)
            symbols = [s.strip() for s in (symbols_str or "").split(",") if s.strip()]

            key = (file_path, "imports")
            if key not in neighbors_dict:
                neighbors_dict[key] = NeighborFile(
                    file_path=file_path,
                    relationship_type="imports",
                    edge_count=edge_count,
                    symbols_involved=symbols,
                    confidence=avg_confidence or 0.0,
                )

    def _query_called_by(
        self,
        conn,
        target_file_id: int,
        min_confidence: float,
        neighbors_dict: Dict[Tuple[str, str], NeighborFile],
    ) -> None:
        """Find files that call functions in the target."""
        query = """
        SELECT DISTINCT
            f.file_path,
            COUNT(d.dependency_id) as edge_count,
            AVG(d.confidence) as avg_confidence,
            GROUP_CONCAT(DISTINCT s2.name) as symbols
        FROM dependencies d
        JOIN symbols s1 ON d.source_symbol_id = s1.symbol_id
        JOIN files f ON s1.file_id = f.file_id
        JOIN symbols s2 ON d.target_symbol_id = s2.symbol_id
        WHERE s2.file_id = ?
            AND d.dependency_kind = 'calls'
            AND d.confidence >= ?
            AND f.file_id != ?
        GROUP BY f.file_path
        ORDER BY edge_count DESC
        """
        cursor = conn.execute(query, (target_file_id, min_confidence, target_file_id))

        for file_path, edge_count, avg_confidence, symbols_str in cursor.fetchall():
            file_path = self._normalize_path(file_path)
            symbols = [s.strip() for s in (symbols_str or "").split(",") if s.strip()]

            key = (file_path, "called_by")
            if key not in neighbors_dict:
                neighbors_dict[key] = NeighborFile(
                    file_path=file_path,
                    relationship_type="called_by",
                    edge_count=edge_count,
                    symbols_involved=symbols,
                    confidence=avg_confidence or 0.0,
                )

    def _query_calls(
        self,
        conn,
        target_file_id: int,
        min_confidence: float,
        neighbors_dict: Dict[Tuple[str, str], NeighborFile],
    ) -> None:
        """Find files whose functions the target calls."""
        query = """
        SELECT DISTINCT
            f.file_path,
            COUNT(d.dependency_id) as edge_count,
            AVG(d.confidence) as avg_confidence,
            GROUP_CONCAT(DISTINCT s2.name) as symbols
        FROM dependencies d
        JOIN symbols s1 ON d.source_symbol_id = s1.symbol_id
        JOIN symbols s2 ON d.target_symbol_id = s2.symbol_id
        JOIN files f ON s2.file_id = f.file_id
        WHERE s1.file_id = ?
            AND d.dependency_kind = 'calls'
            AND d.confidence >= ?
            AND f.file_id != ?
        GROUP BY f.file_path
        ORDER BY edge_count DESC
        """
        cursor = conn.execute(query, (target_file_id, min_confidence, target_file_id))

        for file_path, edge_count, avg_confidence, symbols_str in cursor.fetchall():
            file_path = self._normalize_path(file_path)
            symbols = [s.strip() for s in (symbols_str or "").split(",") if s.strip()]

            key = (file_path, "calls")
            if key not in neighbors_dict:
                neighbors_dict[key] = NeighborFile(
                    file_path=file_path,
                    relationship_type="calls",
                    edge_count=edge_count,
                    symbols_involved=symbols,
                    confidence=avg_confidence or 0.0,
                )

    @staticmethod
    def _normalize_path(file_path: str) -> str:
        """Normalize file path to posix format."""
        return str(Path(file_path).as_posix())

    def to_allowed_files_list(self, graph: NeighborGraph) -> List[str]:
        """
        Convert neighbor graph to allowed_files list for LLM context.

        Returns de-duplicated list of file paths, preserving neighbor priority order.
        """
        seen: Set[str] = set()
        out: List[str] = []
        for n in graph.neighbors:
            if n.file_path in seen:
                continue
            seen.add(n.file_path)
            out.append(n.file_path)
        return out

    def to_dict(self, graph: NeighborGraph) -> Dict[str, Any]:
        """
        Convert neighbor graph to dictionary for JSON serialization.

        Useful for debugging and inspection.
        """
        return {
            "target_file": graph.target_file,
            "total_edges": graph.total_edges,
            "relationship_summary": graph.relationship_summary,
            "neighbors": [
                {
                    "file_path": n.file_path,
                    "relationship_type": n.relationship_type,
                    "edge_count": n.edge_count,
                    "confidence": round(n.confidence, 3),
                    "symbols_involved": n.symbols_involved[:5],  # Top 5 symbols
                }
                for n in graph.neighbors
            ],
        }
