"""
IndexQuery for IntelliRefactor persistent index.

This module implements the IndexQuery class that provides optimized query operations
for duplicate detection, similarity matching, and analysis workflows.

Architecture principles:
1. Optimized queries for duplicate detection and similarity matching
2. Efficient aggregation and filtering capabilities
3. Support for complex cross-table queries
4. Memory-efficient result streaming for large datasets
5. Query result caching for frequently accessed data
"""

import sqlite3
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re

from .index_store import IndexStore

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class DuplicateGroup:
    """Represents a group of duplicate symbols or blocks."""

    fingerprint: str
    similarity_type: str  # exact, structural, normalized
    members: List[Dict[str, Any]]
    similarity_score: float = 1.0


@dataclass
class SimilarityMatch:
    """Represents a similarity match between two symbols."""

    source_symbol: Dict[str, Any]
    target_symbol: Dict[str, Any]
    similarity_score: float
    similarity_type: str
    common_features: List[str]


@dataclass
class QueryResult:
    """Generic query result with metadata."""

    data: List[Dict[str, Any]]
    total_count: int
    query_time_ms: float
    has_more: bool = False


class IndexQuery:
    """
    Optimized query operations for the persistent index.

    Provides efficient queries for duplicate detection, similarity matching,
    and complex analysis workflows.
    """

    def __init__(self, index_store: IndexStore):
        """
        Initialize the IndexQuery.

        Args:
            index_store: IndexStore instance for database access
        """
        self.store = index_store
        self.logger = logging.getLogger(__name__)
        self._query_cache = {}  # Simple query result cache
        self._schema_cache: Dict[str, Any] = {}

        # allowlist for dynamic column selection (prevents SQL injection)
        self._symbol_fingerprint_cols = {"token_fingerprint", "ast_fingerprint"}
        self._block_fingerprint_cols = {"normalized_fingerprint", "token_fingerprint", "ast_fingerprint"}

    def _quote_ident(self, name: str) -> str:
        if not _IDENT_RE.match(name):
            raise ValueError(f"Invalid identifier: {name!r}")
        return f'"{name}"'

    # Duplicate detection queries

    def find_exact_duplicates(
        self, fingerprint_type: str = "token_fingerprint", min_group_size: int = 2
    ) -> List[DuplicateGroup]:
        """
        Find exact duplicates based on fingerprints.

        Args:
            fingerprint_type: Type of fingerprint to use (token_fingerprint, ast_fingerprint)
            min_group_size: Minimum number of symbols in a duplicate group

        Returns:
            List of duplicate groups
        """
        cache_key = f"exact_duplicates_{fingerprint_type}_{min_group_size}"
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        if fingerprint_type not in self._symbol_fingerprint_cols:
            raise ValueError(
                f"Unsupported fingerprint_type={fingerprint_type!r}. "
                f"Allowed: {sorted(self._symbol_fingerprint_cols)}"
            )

        with self.store._get_connection() as conn:
            col = self._quote_ident(fingerprint_type)
            sql = (
                "SELECT " + col + ", COUNT(*) as count, GROUP_CONCAT(symbol_id) as symbol_ids "
                "FROM symbols "
                "WHERE " + col + " IS NOT NULL AND " + col + " != '' "
                "GROUP BY " + col + " "
                "HAVING count >= ? "
                "ORDER BY count DESC"
            )
            cursor = conn.execute(sql, (min_group_size,))

            duplicate_groups = []
            for row in cursor.fetchall():
                fingerprint, count, symbol_ids_str = row
                symbol_ids = [int(sid) for sid in symbol_ids_str.split(",")]

                # Get detailed symbol information
                members = []
                for symbol_id in symbol_ids:
                    symbol_cursor = conn.execute(
                        """
                        SELECT s.symbol_uid, s.name, s.qualified_name, s.kind,
                               s.line_start, s.line_end, s.complexity_score,
                               f.file_path
                        FROM symbols s
                        JOIN files f ON s.file_id = f.file_id
                        WHERE s.symbol_id = ?
                    """,
                        (symbol_id,),
                    )

                    symbol_row = symbol_cursor.fetchone()
                    if symbol_row:
                        members.append(
                            {
                                "symbol_uid": symbol_row[0],
                                "name": symbol_row[1],
                                "qualified_name": symbol_row[2],
                                "kind": symbol_row[3],
                                "line_start": symbol_row[4],
                                "line_end": symbol_row[5],
                                "complexity_score": symbol_row[6],
                                "file_path": symbol_row[7],
                            }
                        )

                duplicate_groups.append(
                    DuplicateGroup(
                        fingerprint=fingerprint,
                        similarity_type="exact",
                        members=members,
                        similarity_score=1.0,
                    )
                )

            self._query_cache[cache_key] = duplicate_groups
            return duplicate_groups

    def find_structural_duplicates(
        self, min_similarity: float = 0.8, min_group_size: int = 2
    ) -> List[DuplicateGroup]:
        """
        Find structural duplicates based on AST fingerprints.

        Args:
            min_similarity: Minimum similarity score (0.0 to 1.0)
            min_group_size: Minimum number of symbols in a duplicate group

        Returns:
            List of duplicate groups
        """
        with self.store._get_connection() as conn:
            # Group by AST fingerprint for structural similarity
            cursor = conn.execute(
                """
                SELECT ast_fingerprint, COUNT(*) as count,
                       GROUP_CONCAT(symbol_id) as symbol_ids
                FROM symbols 
                WHERE ast_fingerprint IS NOT NULL AND ast_fingerprint != ''
                GROUP BY ast_fingerprint
                HAVING count >= ?
                ORDER BY count DESC
            """,
                (min_group_size,),
            )

            duplicate_groups = []
            for row in cursor.fetchall():
                fingerprint, count, symbol_ids_str = row
                symbol_ids = [int(sid) for sid in symbol_ids_str.split(",")]

                # Get detailed symbol information
                members = self._get_symbols_by_ids(conn, symbol_ids)

                duplicate_groups.append(
                    DuplicateGroup(
                        fingerprint=fingerprint,
                        similarity_type="structural",
                        members=members,
                        similarity_score=min_similarity,
                    )
                )

            return duplicate_groups

    def find_block_clones(
        self, min_lines: int = 3, fingerprint_type: str = "normalized_fingerprint"
    ) -> List[DuplicateGroup]:
        """
        Find duplicate code blocks within methods.

        Args:
            min_lines: Minimum number of lines for a block to be considered
            fingerprint_type: Type of fingerprint to use for comparison

        Returns:
            List of block clone groups
        """
        if fingerprint_type not in self._block_fingerprint_cols:
            raise ValueError(
                f"Unsupported fingerprint_type={fingerprint_type!r}. "
                f"Allowed: {sorted(self._block_fingerprint_cols)}"
            )

        with self.store._get_connection() as conn:
            col = self._quote_ident(fingerprint_type)
            sql = (
                "SELECT " + col + ", COUNT(*) as count, GROUP_CONCAT(block_id) as block_ids "
                "FROM blocks "
                "WHERE " + col + " IS NOT NULL AND " + col + " != '' "
                "AND lines_of_code >= ? "
                "GROUP BY " + col + " "
                "HAVING count >= 2 "
                "ORDER BY count DESC"
            )
            cursor = conn.execute(sql, (min_lines,))

            clone_groups = []
            for row in cursor.fetchall():
                fingerprint, count, block_ids_str = row
                block_ids = [int(bid) for bid in block_ids_str.split(",")]

                # Get detailed block information
                members = []
                for block_id in block_ids:
                    block_cursor = conn.execute(
                        """
                        SELECT b.block_uid, b.kind, b.line_start, b.line_end,
                               b.lines_of_code, b.nesting_level,
                               s.qualified_name, f.file_path
                        FROM blocks b
                        JOIN symbols s ON b.symbol_id = s.symbol_id
                        JOIN files f ON s.file_id = f.file_id
                        WHERE b.block_id = ?
                    """,
                        (block_id,),
                    )

                    block_row = block_cursor.fetchone()
                    if block_row:
                        members.append(
                            {
                                "block_uid": block_row[0],
                                "kind": block_row[1],
                                "line_start": block_row[2],
                                "line_end": block_row[3],
                                "lines_of_code": block_row[4],
                                "nesting_level": block_row[5],
                                "symbol_name": block_row[6],
                                "file_path": block_row[7],
                            }
                        )

                clone_groups.append(
                    DuplicateGroup(
                        fingerprint=fingerprint,
                        similarity_type="block_clone",
                        members=members,
                        similarity_score=1.0,
                    )
                )

            return clone_groups

    # Similarity matching queries

    def find_similar_symbols(
        self, target_symbol_uid: str, min_similarity: float = 0.7, max_results: int = 10
    ) -> List[SimilarityMatch]:
        """
        Find symbols similar to the target symbol.

        Args:
            target_symbol_uid: UID of the target symbol
            min_similarity: Minimum similarity score
            max_results: Maximum number of results to return

        Returns:
            List of similarity matches
        """
        target_symbol = self.store.get_symbol(target_symbol_uid)
        if not target_symbol:
            return []

        with self.store._get_connection() as conn:
            # Find symbols with similar fingerprints
            cursor = conn.execute(
                """
                SELECT s.symbol_uid, s.name, s.qualified_name, s.kind,
                       s.ast_fingerprint, s.token_fingerprint, s.semantic_category,
                       s.complexity_score, f.file_path
                FROM symbols s
                JOIN files f ON s.file_id = f.file_id
                WHERE s.symbol_uid != ?
                  AND s.kind = ?
                  AND (s.ast_fingerprint = ? OR s.token_fingerprint = ?)
                LIMIT ?
            """,
                (
                    target_symbol_uid,
                    target_symbol["kind"],
                    target_symbol["ast_fingerprint"],
                    target_symbol["token_fingerprint"],
                    max_results,
                ),
            )

            matches = []
            for row in cursor.fetchall():
                candidate = {
                    "symbol_uid": row[0],
                    "name": row[1],
                    "qualified_name": row[2],
                    "kind": row[3],
                    "ast_fingerprint": row[4],
                    "token_fingerprint": row[5],
                    "semantic_category": row[6],
                    "complexity_score": row[7],
                    "file_path": row[8],
                }

                # Calculate similarity score
                similarity_score, similarity_type, common_features = self._calculate_similarity(
                    target_symbol, candidate
                )

                if similarity_score >= min_similarity:
                    matches.append(
                        SimilarityMatch(
                            source_symbol=target_symbol,
                            target_symbol=candidate,
                            similarity_score=similarity_score,
                            similarity_type=similarity_type,
                            common_features=common_features,
                        )
                    )

            # Sort by similarity score
            matches.sort(key=lambda x: x.similarity_score, reverse=True)
            return matches[:max_results]

    def find_symbols_by_category(
        self, semantic_category: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Find symbols by semantic category."""
        with self.store._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT s.symbol_uid, s.name, s.qualified_name, s.kind,
                       s.line_start, s.complexity_score, f.file_path
                FROM symbols s
                JOIN files f ON s.file_id = f.file_id
                WHERE s.semantic_category = ?
                ORDER BY s.complexity_score DESC
                LIMIT ?
            """,
                (semantic_category, limit),
            )

            symbols = []
            for row in cursor.fetchall():
                symbols.append(
                    {
                        "symbol_uid": row[0],
                        "name": row[1],
                        "qualified_name": row[2],
                        "kind": row[3],
                        "line_start": row[4],
                        "complexity_score": row[5],
                        "file_path": row[6],
                    }
                )

            return symbols

    # Analysis queries

    def get_complexity_distribution(self) -> Dict[str, Any]:
        """Get complexity distribution across symbols."""
        with self.store._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT 
                    AVG(complexity_score) as avg_complexity,
                    MIN(complexity_score) as min_complexity,
                    MAX(complexity_score) as max_complexity,
                    COUNT(*) as total_symbols,
                    SUM(CASE WHEN complexity_score > 10 THEN 1 ELSE 0 END) as high_complexity_count
                FROM symbols
                WHERE complexity_score > 0
            """
            )

            row = cursor.fetchone()
            if row and row[3] > 0:  # Check if total_symbols > 0
                return {
                    "avg_complexity": round(row[0] or 0, 2),
                    "min_complexity": row[1] or 0,
                    "max_complexity": row[2] or 0,
                    "total_symbols": row[3],
                    "high_complexity_count": row[4],
                    "high_complexity_percentage": (
                        round((row[4] / row[3]) * 100, 2) if row[3] > 0 else 0
                    ),
                }

            return {}

    def get_file_statistics(self) -> Dict[str, Any]:
        """Get file-level statistics."""
        with self.store._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_files,
                    SUM(CASE WHEN is_test_file THEN 1 ELSE 0 END) as test_files,
                    SUM(lines_of_code) as total_loc,
                    AVG(lines_of_code) as avg_loc_per_file,
                    MAX(lines_of_code) as max_loc_per_file
                FROM files
            """
            )

            row = cursor.fetchone()
            if row and row[0] > 0:  # Check if total_files > 0
                return {
                    "total_files": row[0],
                    "test_files": row[1] or 0,
                    "source_files": row[0] - (row[1] or 0),
                    "total_loc": row[2] or 0,
                    "avg_loc_per_file": round(row[3] or 0, 2),
                    "max_loc_per_file": row[4] or 0,
                }

            return {}

    def get_dependency_graph(self, symbol_uid: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get dependency graph for a symbol."""
        visited = set()
        graph = {"nodes": [], "edges": []}

        def traverse(current_uid: str, depth: int):
            if depth > max_depth or current_uid in visited:
                return

            visited.add(current_uid)
            symbol = self.store.get_symbol(current_uid)
            if not symbol:
                return

            graph["nodes"].append(
                {
                    "uid": current_uid,
                    "name": symbol["name"],
                    "kind": symbol["kind"],
                    "depth": depth,
                }
            )

            # Get dependencies
            with self.store._get_connection() as conn:
                try:
                    cursor = conn.execute(
                        """
                        SELECT d.target_external, d.dependency_kind, d.confidence
                        FROM dependencies d
                        JOIN symbols s ON d.source_symbol_id = s.symbol_id
                        WHERE s.symbol_uid = ?
                        """,
                        (current_uid,),
                    )
                except sqlite3.OperationalError:
                    cursor = conn.execute(
                        """
                        SELECT d.target_external, d.kind, d.confidence
                        FROM dependencies d
                        JOIN symbols s ON d.source_symbol_id = s.symbol_id
                        WHERE s.symbol_uid = ?
                        """,
                        (current_uid,),
                    )

                for row in cursor.fetchall():
                    target, kind, confidence = row
                    graph["edges"].append(
                        {
                            "source": current_uid,
                            "target": target,
                            "kind": kind,
                            "confidence": confidence,
                        }
                    )
                    # Only recurse if target looks like an internal symbol_uid.
                    # Many rows may store external targets; in that case get_symbol() returns None and we stop.
                    if isinstance(target, str) and target:
                        traverse(target, depth + 1)

        traverse(symbol_uid, 0)
        return graph

    # Search and filtering

    def search_symbols(
        self, query: str, kind_filter: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search symbols by name or qualified name."""
        with self.store._get_connection() as conn:
            sql = """
                SELECT s.symbol_uid, s.name, s.qualified_name, s.kind,
                       s.line_start, s.complexity_score, f.file_path
                FROM symbols s
                JOIN files f ON s.file_id = f.file_id
                WHERE (s.name LIKE ? OR s.qualified_name LIKE ?)
            """
            params = [f"%{query}%", f"%{query}%"]

            if kind_filter:
                sql += " AND s.kind = ?"
                params.append(kind_filter)

            sql += " ORDER BY s.name LIMIT ?"
            params.append(limit)

            cursor = conn.execute(sql, params)

            symbols = []
            for row in cursor.fetchall():
                symbols.append(
                    {
                        "symbol_uid": row[0],
                        "name": row[1],
                        "qualified_name": row[2],
                        "kind": row[3],
                        "line_start": row[4],
                        "complexity_score": row[5],
                        "file_path": row[6],
                    }
                )

            return symbols

    def get_symbols_by_complexity(
        self, min_complexity: int = 10, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get symbols with high complexity."""
        with self.store._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT s.symbol_uid, s.name, s.qualified_name, s.kind,
                       s.complexity_score, s.line_start, s.line_end, f.file_path
                FROM symbols s
                JOIN files f ON s.file_id = f.file_id
                WHERE s.complexity_score >= ?
                ORDER BY s.complexity_score DESC
                LIMIT ?
            """,
                (min_complexity, limit),
            )

            symbols = []
            for row in cursor.fetchall():
                symbols.append(
                    {
                        "symbol_uid": row[0],
                        "name": row[1],
                        "qualified_name": row[2],
                        "kind": row[3],
                        "complexity_score": row[4],
                        "line_start": row[5],
                        "line_end": row[6],
                        "file_path": row[7],
                    }
                )

            return symbols

    # Helper methods

    def _get_symbols_by_ids(
        self, conn: sqlite3.Connection, symbol_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """Get detailed symbol information by IDs."""
        if not symbol_ids:
            return []

        placeholders = ",".join(["?"] * len(symbol_ids))
        sql = (
            "SELECT s.symbol_uid, s.name, s.qualified_name, s.kind, "
            "s.line_start, s.line_end, s.complexity_score, f.file_path "
            "FROM symbols s "
            "JOIN files f ON s.file_id = f.file_id "
            "WHERE s.symbol_id IN (" + placeholders + ")"
        )
        cursor = conn.execute(sql, symbol_ids)

        symbols = []
        for row in cursor.fetchall():
            symbols.append(
                {
                    "symbol_uid": row[0],
                    "name": row[1],
                    "qualified_name": row[2],
                    "kind": row[3],
                    "line_start": row[4],
                    "line_end": row[5],
                    "complexity_score": row[6],
                    "file_path": row[7],
                }
            )

        return symbols

    def _calculate_similarity(
        self, symbol1: Dict[str, Any], symbol2: Dict[str, Any]
    ) -> Tuple[float, str, List[str]]:
        """Calculate similarity between two symbols."""
        common_features = []
        similarity_score = 0.0
        similarity_type = "unknown"

        # Exact match on token fingerprint
        if symbol1.get("token_fingerprint") and symbol1["token_fingerprint"] == symbol2.get(
            "token_fingerprint"
        ):
            similarity_score = 1.0
            similarity_type = "exact"
            common_features.append("identical_tokens")

        # Structural match on AST fingerprint
        elif symbol1.get("ast_fingerprint") and symbol1["ast_fingerprint"] == symbol2.get(
            "ast_fingerprint"
        ):
            similarity_score = 0.9
            similarity_type = "structural"
            common_features.append("identical_structure")

        # Semantic similarity
        else:
            score = 0.0

            # Same semantic category
            if symbol1.get("semantic_category") and symbol1["semantic_category"] == symbol2.get(
                "semantic_category"
            ):
                score += 0.3
                common_features.append("same_category")

            # Similar complexity
            complexity1 = symbol1.get("complexity_score", 0)
            complexity2 = symbol2.get("complexity_score", 0)
            if complexity1 > 0 and complexity2 > 0:
                complexity_ratio = min(complexity1, complexity2) / max(complexity1, complexity2)
                if complexity_ratio > 0.8:
                    score += 0.2
                    common_features.append("similar_complexity")

            # Similar name patterns
            name1 = symbol1.get("name", "").lower()
            name2 = symbol2.get("name", "").lower()
            if name1 and name2:
                # Simple name similarity (could be enhanced with edit distance)
                if any(word in name2 for word in name1.split("_")):
                    score += 0.2
                    common_features.append("similar_names")

            similarity_score = score
            similarity_type = "semantic" if score > 0 else "none"

        return similarity_score, similarity_type, common_features

    def find_importers_of_module(self, module_path: str) -> List[Dict[str, Any]]:
        """
        Find which modules import the given module.

        Args:
            module_path: Path to the module to find importers for

        Returns:
            List of modules that import the given module
        """
        with self.store._get_connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT f.file_path, f.module_name
                    FROM imports i
                    JOIN files f ON i.importer_file_id = f.file_id
                    WHERE i.imported_module_path = ?
                    """,
                    (module_path,),
                )
            except sqlite3.OperationalError:
                # Optional table (depends on schema)
                return []

            importers = []
            for row in cursor.fetchall():
                importers.append({"importer_path": row[0], "importer_module_name": row[1]})

            return importers

    def find_usage_of_symbol(self, symbol_uid: str) -> List[Dict[str, Any]]:
        """
        Find where a symbol is used (reverse dependency lookup).

        Args:
            symbol_uid: UID of the symbol to find usage for

        Returns:
            List of locations where the symbol is used
        """
        with self.store._get_connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    SELECT d.source_symbol_id, d.target_external, d.dependency_kind, d.confidence,
                           s.symbol_uid, s.name, s.qualified_name, s.kind as symbol_kind,
                           f.file_path
                    FROM dependencies d
                    JOIN symbols s ON d.source_symbol_id = s.symbol_id
                    JOIN files f ON s.file_id = f.file_id
                    WHERE d.target_external = ?
                    """,
                    (symbol_uid,),
                )
            except sqlite3.OperationalError:
                cursor = conn.execute(
                    """
                    SELECT d.source_symbol_id, d.target_external, d.kind, d.confidence,
                           s.symbol_uid, s.name, s.qualified_name, s.kind as symbol_kind,
                           f.file_path
                    FROM dependencies d
                    JOIN symbols s ON d.source_symbol_id = s.symbol_id
                    JOIN files f ON s.file_id = f.file_id
                    WHERE d.target_external = ?
                    """,
                    (symbol_uid,),
                )

            usages = []
            for row in cursor.fetchall():
                usages.append(
                    {
                        "source_symbol_id": row[0],
                        "target_symbol_uid": row[1],  # This is the symbol_uid we're looking for
                        "dependency_kind": row[2],
                        "confidence": row[3],
                        "using_symbol_uid": row[4],
                        "using_symbol_name": row[5],
                        "using_qualified_name": row[6],
                        "using_symbol_kind": row[7],
                        "file_path": row[8],
                    }
                )

            return usages

    def find_symbol_references(self, symbol_uid: str) -> List[Dict[str, Any]]:
        """
        Find all references to a symbol in the codebase.

        Args:
            symbol_uid: UID of the symbol to find references for

        Returns:
            List of all references to the symbol
        """
        with self.store._get_connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    SELECT r.reference_type, r.line_number, r.column_start, r.column_end,
                           r.context, f.file_path, s.symbol_uid, s.name
                    FROM references r
                    JOIN files f ON r.file_id = f.file_id
                    JOIN symbols s ON r.symbol_id = s.symbol_id
                    WHERE r.target_symbol_uid = ?
                    ORDER BY f.file_path, r.line_number
                    """,
                    (symbol_uid,),
                )
            except sqlite3.OperationalError:
                # Optional table (depends on schema)
                return []

            references = []
            for row in cursor.fetchall():
                references.append(
                    {
                        "reference_type": row[0],
                        "line_number": row[1],
                        "column_start": row[2],
                        "column_end": row[3],
                        "context": row[4],
                        "file_path": row[5],
                        "symbol_uid": row[6],
                        "symbol_name": row[7],
                    }
                )

            return references

    def is_symbol_used_in_project(self, symbol_uid: str) -> bool:
        """
        Check if a symbol is used anywhere in the project.

        Args:
            symbol_uid: UID of the symbol to check

        Returns:
            True if the symbol is used in the project, False otherwise
        """
        # First check direct dependencies
        usages = self.find_usage_of_symbol(symbol_uid)
        if usages:
            return True

        # Then check references
        references = self.find_symbol_references(symbol_uid)
        return len(references) > 0

    def clear_cache(self) -> None:
        """Clear the query result cache."""
        self._query_cache.clear()
        self.logger.debug("Query cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._query_cache),
            "cached_queries": list(self._query_cache.keys()),
        }
