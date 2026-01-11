"""
IndexStore for IntelliRefactor persistent index.

This module implements the IndexStore class that provides thread-safe SQLite operations
for storing and retrieving analysis data.
"""

from __future__ import annotations

import sqlite3
import threading
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime
from contextlib import contextmanager
import json
import re

from .index_schema import IndexSchema

if TYPE_CHECKING:
    from .models import BlockInfo, DeepClassInfo, DeepMethodInfo, DependencyInfo

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class IndexStore:
    """Thread-safe SQLite operations for the persistent index."""

    def __init__(self, db_path: Path):
        """Initialize the IndexStore."""
        self.db_path = Path(db_path) if isinstance(db_path, str) else db_path
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()

        # Initialize database if it doesn't exist or is empty
        needs_init = False
        if not self.db_path.exists():
            needs_init = True
            self.logger.info(f"Creating new database at {self.db_path}")
        else:
            # Check if database has tables (might be an empty file)
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='files'"
                    )
                    if cursor.fetchone() is None:
                        needs_init = True
                        self.logger.info(f"Initializing empty database at {self.db_path}")
            except sqlite3.Error:
                needs_init = True
                self.logger.info(f"Reinitializing corrupted database at {self.db_path}")

        if needs_init:
            conn = IndexSchema.create_database(self.db_path)
            conn.close()  # Close the connection returned by create_database

        self._test_connection()

    def _test_connection(self):
        """Test database connection and schema."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM files")
                cursor.fetchone()
                self.logger.debug("Database connection test successful")
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            raise

    @contextmanager
    def _get_connection(self):
        """Get a thread-safe database connection."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            try:
                yield conn
            finally:
                conn.close()

    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        with self._get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def store_file(
        self,
        file_path: str,
        content_hash: str,
        file_size: int,
        lines_of_code: int,
        is_test_file: bool = False,
    ) -> int:
        """Store or update a file record."""
        with self.transaction() as conn:
            # Delete existing record if it exists
            conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))

            cursor = conn.execute(
                """
                INSERT INTO files (file_path, content_hash, last_modified, file_size, 
                                 lines_of_code, is_test_file, last_analyzed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    file_path,
                    content_hash,
                    datetime.now(),
                    file_size,
                    lines_of_code,
                    is_test_file,
                    datetime.now(),
                ),
            )

            return cursor.lastrowid

    def get_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file record by path."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT file_id, file_path, content_hash, last_modified, 
                       file_size, lines_of_code, is_test_file, last_analyzed
                FROM files WHERE file_path = ?
            """,
                (file_path,),
            )

            row = cursor.fetchone()
            if row:
                return {
                    "file_id": row[0],
                    "file_path": row[1],
                    "content_hash": row[2],
                    "last_modified": row[3],
                    "file_size": row[4],
                    "lines_of_code": row[5],
                    "is_test_file": bool(row[6]),  # Convert SQLite integer to boolean
                    "last_analyzed": row[7],
                }
            return None

    def get_all_file_hashes(self) -> Dict[str, str]:
        """Get all file paths and their content hashes."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT file_path, content_hash FROM files")
            return {row[0]: row[1] for row in cursor.fetchall()}

    def store_symbols(self, file_id: int, symbols: List[Dict[str, Any]]) -> Dict[str, int]:
        """Store multiple symbols for a file."""
        symbol_ids = {}

        with self.transaction() as conn:
            for symbol in symbols:
                # Delete existing symbol if it exists
                conn.execute("DELETE FROM symbols WHERE symbol_uid = ?", (symbol["symbol_uid"],))

                cursor = conn.execute(
                    """
                    INSERT INTO symbols (
                        symbol_uid, file_id, name, qualified_name, kind, 
                        line_start, line_end, signature, ast_fingerprint, 
                        token_fingerprint, operation_signature, semantic_category,
                        responsibility_markers, is_public, is_async, is_property, 
                        complexity_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        symbol["symbol_uid"],
                        file_id,
                        symbol["name"],
                        symbol["qualified_name"],
                        symbol["kind"],
                        symbol["line_start"],
                        symbol["line_end"],
                        symbol["signature"],
                        symbol["ast_fingerprint"],
                        symbol["token_fingerprint"],
                        symbol.get("operation_signature", ""),
                        symbol["semantic_category"],
                        symbol["responsibility_markers"],
                        symbol["is_public"],
                        symbol["is_async"],
                        symbol["is_property"],
                        symbol["complexity_score"],
                    ),
                )

                symbol_ids[symbol["symbol_uid"]] = cursor.lastrowid

        return symbol_ids

    def get_symbol(self, symbol_uid: str) -> Optional[Dict[str, Any]]:
        """Get symbol by UID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT symbol_id, symbol_uid, file_id, name, qualified_name, kind,
                       line_start, line_end, signature, ast_fingerprint, 
                       token_fingerprint, operation_signature, semantic_category,
                       responsibility_markers, is_public, is_async, is_property,
                       complexity_score
                FROM symbols WHERE symbol_uid = ?
            """,
                (symbol_uid,),
            )

            row = cursor.fetchone()
            if row:
                return {
                    "symbol_id": row[0],
                    "symbol_uid": row[1],
                    "file_id": row[2],
                    "name": row[3],
                    "qualified_name": row[4],
                    "kind": row[5],
                    "line_start": row[6],
                    "line_end": row[7],
                    "signature": row[8],
                    "ast_fingerprint": row[9],
                    "token_fingerprint": row[10],
                    "operation_signature": row[11],
                    "semantic_category": row[12],
                    "responsibility_markers": row[13],
                    "is_public": bool(row[14]),  # Convert SQLite integer to boolean
                    "is_async": bool(row[15]),  # Convert SQLite integer to boolean
                    "is_property": bool(row[16]),  # Convert SQLite integer to boolean
                    "complexity_score": row[17],
                }
            return None

    def get_symbols_by_file(self, file_id: int) -> List[Dict[str, Any]]:
        """Get all symbols for a file."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT symbol_id, symbol_uid, name, qualified_name, kind,
                       line_start, line_end, signature, ast_fingerprint,
                       token_fingerprint, semantic_category, complexity_score
                FROM symbols WHERE file_id = ?
                ORDER BY line_start
            """,
                (file_id,),
            )

            symbols = []
            for row in cursor.fetchall():
                symbols.append(
                    {
                        "symbol_id": row[0],
                        "symbol_uid": row[1],
                        "name": row[2],
                        "qualified_name": row[3],
                        "kind": row[4],
                        "line_start": row[5],
                        "line_end": row[6],
                        "signature": row[7],
                        "ast_fingerprint": row[8],
                        "token_fingerprint": row[9],
                        "semantic_category": row[10],
                        "complexity_score": row[11],
                    }
                )

            return symbols

    def store_blocks(self, symbol_ids: Dict[str, int], blocks: List[Dict[str, Any]]) -> None:
        """Store code blocks for symbols."""
        with self.transaction() as conn:
            for block in blocks:
                symbol_id = symbol_ids.get(block["symbol_uid"])
                if not symbol_id:
                    continue

                # Delete existing block if it exists
                conn.execute("DELETE FROM blocks WHERE block_uid = ?", (block["block_uid"],))

                conn.execute(
                    """
                    INSERT INTO blocks (
                        block_uid, symbol_id, kind, line_start, line_end,
                        lines_of_code, nesting_level, ast_fingerprint,
                        token_fingerprint, normalized_fingerprint
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        block["block_uid"],
                        symbol_id,
                        block["kind"],
                        block["line_start"],
                        block["line_end"],
                        block["lines_of_code"],
                        block["nesting_level"],
                        block["ast_fingerprint"],
                        block["token_fingerprint"],
                        block["normalized_fingerprint"],
                    ),
                )

    def get_blocks_by_symbol(self, symbol_id: int) -> List[Dict[str, Any]]:
        """Get all blocks for a symbol."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT block_id, block_uid, kind, line_start, line_end,
                       lines_of_code, nesting_level, ast_fingerprint,
                       token_fingerprint, normalized_fingerprint
                FROM blocks WHERE symbol_id = ?
                ORDER BY line_start
            """,
                (symbol_id,),
            )

            blocks = []
            for row in cursor.fetchall():
                blocks.append(
                    {
                        "block_id": row[0],
                        "block_uid": row[1],
                        "kind": row[2],
                        "line_start": row[3],
                        "line_end": row[4],
                        "lines_of_code": row[5],
                        "nesting_level": row[6],
                        "ast_fingerprint": row[7],
                        "token_fingerprint": row[8],
                        "normalized_fingerprint": row[9],
                    }
                )

            return blocks

    def store_dependencies(
        self, symbol_ids: Dict[str, int], dependencies: List[Dict[str, Any]]
    ) -> None:
        """Store dependencies for symbols."""
        with self.transaction() as conn:
            for dep in dependencies:
                source_symbol_id = symbol_ids.get(dep["source_symbol_uid"])
                if not source_symbol_id:
                    continue

                # Schema drift protection: some schemas use "kind", some "dependency_kind".
                params = (
                    source_symbol_id,
                    dep.get("target_symbol_id"),
                    dep.get("target_external"),
                    dep.get("kind", ""),
                    dep.get("resolution", ""),
                    dep.get("confidence", 0.0),
                    json.dumps(dep.get("evidence", {})),
                    1,
                )

                try:
                    conn.execute(
                        """
                        INSERT INTO dependencies (
                            source_symbol_id, target_symbol_id, target_external,
                            dependency_kind, resolution, confidence, evidence_json, count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        params,
                    )
                except sqlite3.OperationalError:
                    conn.execute(
                        """
                        INSERT INTO dependencies (
                            source_symbol_id, target_symbol_id, target_external,
                            kind, resolution, confidence, evidence_json, count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        params,
                    )

    def get_dependencies_by_symbol(self, symbol_id: int) -> List[Dict[str, Any]]:
        """Get all dependencies for a symbol."""
        with self._get_connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    SELECT dependency_id, target_symbol_id, target_external,
                           dependency_kind, resolution, confidence, evidence_json, count
                    FROM dependencies WHERE source_symbol_id = ?
                    """,
                    (symbol_id,),
                )
            except sqlite3.OperationalError:
                cursor = conn.execute(
                    """
                    SELECT dependency_id, target_symbol_id, target_external,
                           kind, resolution, confidence, evidence_json, count
                    FROM dependencies WHERE source_symbol_id = ?
                    """,
                    (symbol_id,),
                )

            dependencies = []
            for row in cursor.fetchall():
                dependencies.append(
                    {
                        "dependency_id": row[0],
                        "target_symbol_id": row[1],
                        "target_external": row[2],
                        "kind": row[3],
                        "resolution": row[4],
                        "confidence": row[5],
                        "evidence": json.loads(row[6]) if row[6] else {},
                        "count": row[7],
                    }
                )

            return dependencies

    def store_attribute_accesses(
        self, symbol_ids: Dict[str, int], accesses: List[Dict[str, Any]]
    ) -> None:
        """Store attribute access patterns for symbols."""
        with self.transaction() as conn:
            for access in accesses:
                symbol_id = symbol_ids.get(access["symbol_uid"])
                if not symbol_id:
                    continue

                conn.execute(
                    """
                    INSERT INTO attribute_access (
                        symbol_id, attribute_name, access_type, line_number,
                        confidence, evidence_json, count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        symbol_id,
                        access["attribute_name"],
                        access["access_type"],
                        access["line_number"],
                        access["confidence"],
                        json.dumps(access["evidence"]),
                        1,
                    ),
                )

    def get_attribute_accesses_by_symbol(self, symbol_id: int) -> List[Dict[str, Any]]:
        """Get all attribute accesses for a symbol."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT access_id, attribute_name, access_type, line_number,
                       confidence, evidence_json, count
                FROM attribute_access WHERE symbol_id = ?
            """,
                (symbol_id,),
            )

            accesses = []
            for row in cursor.fetchall():
                accesses.append(
                    {
                        "access_id": row[0],
                        "attribute_name": row[1],
                        "access_type": row[2],
                        "line_number": row[3],
                        "confidence": row[4],
                        "evidence": json.loads(row[5]) if row[5] else {},
                        "count": row[6],
                    }
                )

            return accesses

    def bulk_insert_symbols(self, symbols_data: List[Tuple]) -> None:
        """Bulk insert symbols for better performance."""
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO symbols (
                    symbol_uid, file_id, name, qualified_name, kind,
                    line_start, line_end, signature, ast_fingerprint,
                    token_fingerprint, semantic_category, complexity_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                symbols_data,
            )

    def bulk_insert_blocks(self, blocks_data: List[Tuple]) -> None:
        """Bulk insert blocks for better performance."""
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO blocks (
                    block_uid, symbol_id, kind, line_start, line_end,
                    lines_of_code, nesting_level, ast_fingerprint,
                    token_fingerprint, normalized_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                blocks_data,
            )

    def bulk_insert_dependencies(self, dependencies_data: List[Tuple]) -> None:
        """Bulk insert dependencies for better performance."""
        with self.transaction() as conn:
            # Schema drift protection: "kind" vs "dependency_kind"
            try:
                conn.executemany(
                    """
                    INSERT INTO dependencies (
                        source_symbol_id, target_external, dependency_kind, resolution,
                        confidence, evidence_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    dependencies_data,
                )
            except sqlite3.OperationalError:
                conn.executemany(
                    """
                    INSERT INTO dependencies (
                        source_symbol_id, target_external, kind, resolution,
                        confidence, evidence_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    dependencies_data,
                )

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._get_connection() as conn:
            stats = {}

            # Count records in each table
            tables = ("files", "symbols", "blocks", "dependencies", "attribute_access")

            def _quote_ident(name: str) -> str:
                # sqlite identifiers cannot be parametrized; validate + quote
                if not _IDENT_RE.match(name):
                    raise ValueError(f"Invalid identifier: {name!r}")
                return f'"{name}"'

            for table in tables:
                sql = "SELECT COUNT(*) FROM " + _quote_ident(table)
                cursor = conn.execute(sql)
                stats[f"{table}_count"] = cursor.fetchone()[0]

            # Get database size
            cursor = conn.execute(
                "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"
            )
            stats["database_size_bytes"] = cursor.fetchone()[0]

            # Get last analysis time
            cursor = conn.execute("SELECT MAX(last_analyzed) FROM files")
            stats["last_analysis"] = cursor.fetchone()[0]

            return stats

    def vacuum(self) -> None:
        """Vacuum the database to reclaim space."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")
            self.logger.info("Database vacuumed successfully")

    def analyze(self) -> None:
        """Update database statistics for query optimization."""
        with self._get_connection() as conn:
            conn.execute("ANALYZE")
            self.logger.info("Database statistics updated")

    def clear_all_data(self) -> None:
        """Clear all data from the database."""
        with self.transaction() as conn:
            # Delete in reverse dependency order
            tables = (
                "attribute_access",
                "dependencies",
                "blocks",
                "symbols",
                "files",
                "duplicate_members",
                "duplicate_groups",
                "refactoring_decisions",
                "problems",
                "analysis_runs",
            )

            def _quote_ident(name: str) -> str:
                if not _IDENT_RE.match(name):
                    raise ValueError(f"Invalid identifier: {name!r}")
                return f'"{name}"'

            for table in tables:
                try:
                    sql = "DELETE FROM " + _quote_ident(table)
                    conn.execute(sql)
                except sqlite3.OperationalError:
                    # Table might not exist
                    pass

            self.logger.info("All data cleared from database")

    # Model serialization methods for enhanced data models

    def store_deep_method_info(self, method_info: DeepMethodInfo) -> int:
        """Store a DeepMethodInfo object to the database."""
        from .index_schema import IndexSchema

        with self.transaction() as conn:
            # First, ensure the file exists
            cursor = conn.execute(
                "SELECT file_id FROM files WHERE file_path = ?",
                (method_info.file_reference.file_path,),
            )
            file_row = cursor.fetchone()
            if not file_row:
                raise ValueError(
                    f"File not found in database: {method_info.file_reference.file_path}"
                )
            file_id = file_row[0]

            # Generate symbol UID
            symbol_uid = IndexSchema.generate_symbol_uid(
                method_info.file_reference.file_path,
                method_info.qualified_name,
                "method",
                method_info.file_reference.line_start,
                method_info.signature,
            )

            # Store the symbol with enhanced information
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO symbols (
                    symbol_uid, file_id, name, qualified_name, kind, line_start, line_end,
                    signature, complexity_score, is_public, is_async, is_property,
                    is_static, is_classmethod, ast_fingerprint, token_fingerprint,
                    operation_signature, semantic_category, responsibility_markers,
                    side_effects, calls_external, uses_attributes, imports_used,
                    confidence, analysis_version, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    symbol_uid,
                    file_id,
                    method_info.name,
                    method_info.qualified_name,
                    "method",
                    method_info.file_reference.line_start,
                    method_info.file_reference.line_end,
                    method_info.signature,
                    method_info.complexity_score,
                    method_info.is_public,
                    method_info.is_async,
                    method_info.is_property,
                    method_info.is_static,
                    method_info.is_classmethod,
                    method_info.ast_fingerprint,
                    method_info.token_fingerprint,
                    method_info.operation_signature,
                    method_info.semantic_category.value,
                    json.dumps([marker.value for marker in method_info.responsibility_markers]),
                    json.dumps(list(method_info.side_effects)),
                    json.dumps(method_info.calls_external),
                    json.dumps(method_info.uses_attributes),
                    json.dumps(method_info.imports_used),
                    method_info.confidence,
                    method_info.analysis_version,
                    json.dumps(method_info.metadata),
                ),
            )

            return cursor.lastrowid

    def store_block_info(self, block_info: BlockInfo, symbol_id: int) -> int:
        """Store a BlockInfo object to the database."""
        from .index_schema import IndexSchema

        with self.transaction() as conn:
            # Generate block UID
            # We need the symbol_uid for this, so let's get it
            cursor = conn.execute(
                "SELECT symbol_uid FROM symbols WHERE symbol_id = ?", (symbol_id,)
            )
            symbol_row = cursor.fetchone()
            if not symbol_row:
                raise ValueError(f"Symbol not found in database: {symbol_id}")
            symbol_uid = symbol_row[0]

            block_uid = IndexSchema.generate_block_uid(
                symbol_uid,
                block_info.block_type.value,
                block_info.file_reference.line_start,
                block_info.file_reference.line_end,
            )

            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO blocks (
                    block_uid, symbol_id, block_type, line_start, line_end, nesting_level,
                    lines_of_code, statement_count, ast_fingerprint,
                    token_fingerprint, normalized_fingerprint, is_extractable,
                    min_clone_size, confidence, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    block_uid,
                    symbol_id,
                    block_info.block_type.value,
                    block_info.file_reference.line_start,
                    block_info.file_reference.line_end,
                    block_info.nesting_level,
                    block_info.lines_of_code,
                    block_info.statement_count,
                    block_info.ast_fingerprint,
                    block_info.token_fingerprint,
                    block_info.normalized_fingerprint,
                    block_info.is_extractable,
                    block_info.min_clone_size,
                    block_info.confidence,
                    json.dumps(block_info.metadata),
                ),
            )

            return cursor.lastrowid

    def store_dependency_info(
        self,
        dependency_info: DependencyInfo,
        source_symbol_id: int,
        target_symbol_id: Optional[int] = None,
    ) -> int:
        """Store a DependencyInfo object to the database."""

        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO dependencies (
                    source_symbol_id, target_symbol_id, target_external, dependency_kind,
                    resolution, confidence, evidence_json, usage_count, usage_contexts,
                    is_critical, is_circular, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    source_symbol_id,
                    target_symbol_id,
                    dependency_info.target_external,
                    dependency_info.dependency_kind,
                    dependency_info.resolution.value,
                    dependency_info.confidence,
                    json.dumps(dependency_info.evidence.to_dict()),
                    dependency_info.usage_count,
                    json.dumps(dependency_info.usage_contexts),
                    dependency_info.is_critical,
                    dependency_info.is_circular,
                    json.dumps(dependency_info.metadata),
                ),
            )

            return cursor.lastrowid

    def store_deep_class_info(self, class_info: DeepClassInfo) -> int:
        """Store a DeepClassInfo object to the database."""
        from .index_schema import IndexSchema

        with self.transaction() as conn:
            # First, ensure the file exists
            cursor = conn.execute(
                "SELECT file_id FROM files WHERE file_path = ?",
                (class_info.file_reference.file_path,),
            )
            file_row = cursor.fetchone()
            if not file_row:
                raise ValueError(
                    f"File not found in database: {class_info.file_reference.file_path}"
                )
            file_id = file_row[0]

            # Generate symbol UID
            symbol_uid = IndexSchema.generate_symbol_uid(
                class_info.file_reference.file_path,
                class_info.qualified_name,
                "class",
                class_info.file_reference.line_start,
                f"class {class_info.name}:",
            )

            # Store the class as a symbol with enhanced information
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO symbols (
                    symbol_uid, file_id, name, qualified_name, kind, line_start, line_end,
                    signature, complexity_score, methods, attributes, class_attributes,
                    base_classes, derived_classes, responsibility_markers, cohesion_score,
                    method_count, attribute_count, is_abstract, is_interface, is_data_class, is_singleton,
                    confidence, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    symbol_uid,
                    file_id,
                    class_info.name,
                    class_info.qualified_name,
                    "class",
                    class_info.file_reference.line_start,
                    class_info.file_reference.line_end,
                    f"class {class_info.name}:",
                    class_info.complexity_score,
                    json.dumps(class_info.methods),
                    json.dumps(class_info.attributes),
                    json.dumps(class_info.class_attributes),
                    json.dumps(class_info.base_classes),
                    json.dumps(class_info.derived_classes),
                    json.dumps([marker.value for marker in class_info.responsibility_markers]),
                    class_info.cohesion_score,
                    class_info.method_count,
                    class_info.attribute_count,
                    class_info.is_abstract,
                    class_info.is_interface,
                    class_info.is_data_class,
                    class_info.is_singleton,
                    class_info.confidence,
                    json.dumps(class_info.metadata),
                ),
            )

            return cursor.lastrowid

    def bulk_store_method_infos(self, method_infos: List[DeepMethodInfo]) -> List[int]:
        """Bulk store multiple DeepMethodInfo objects."""
        from .index_schema import IndexSchema

        symbol_ids = []

        with self.transaction() as conn:
            for method_info in method_infos:
                # Get file_id
                cursor = conn.execute(
                    "SELECT file_id FROM files WHERE file_path = ?",
                    (method_info.file_reference.file_path,),
                )
                file_row = cursor.fetchone()
                if not file_row:
                    raise ValueError(
                        f"File not found in database: {method_info.file_reference.file_path}"
                    )
                file_id = file_row[0]

                # Generate symbol UID
                symbol_uid = IndexSchema.generate_symbol_uid(
                    method_info.file_reference.file_path,
                    method_info.qualified_name,
                    "method",
                    method_info.file_reference.line_start,
                    method_info.signature,
                )

                cursor = conn.execute(
                    """
                    INSERT OR REPLACE INTO symbols (
                        symbol_uid, file_id, name, qualified_name, kind, line_start, line_end,
                        signature, complexity_score, is_public, is_async, is_property,
                        is_static, is_classmethod, ast_fingerprint, token_fingerprint,
                        operation_signature, semantic_category, responsibility_markers,
                        side_effects, calls_external, uses_attributes, imports_used,
                        confidence, analysis_version, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        symbol_uid,
                        file_id,
                        method_info.name,
                        method_info.qualified_name,
                        "method",
                        method_info.file_reference.line_start,
                        method_info.file_reference.line_end,
                        method_info.signature,
                        method_info.complexity_score,
                        method_info.is_public,
                        method_info.is_async,
                        method_info.is_property,
                        method_info.is_static,
                        method_info.is_classmethod,
                        method_info.ast_fingerprint,
                        method_info.token_fingerprint,
                        method_info.operation_signature,
                        method_info.semantic_category.value,
                        json.dumps([marker.value for marker in method_info.responsibility_markers]),
                        json.dumps(list(method_info.side_effects)),
                        json.dumps(method_info.calls_external),
                        json.dumps(method_info.uses_attributes),
                        json.dumps(method_info.imports_used),
                        method_info.confidence,
                        method_info.analysis_version,
                        json.dumps(method_info.metadata),
                    ),
                )

                symbol_ids.append(cursor.lastrowid)

        return symbol_ids

    def bulk_store_block_infos(self, block_infos: List[Tuple[BlockInfo, int]]) -> List[int]:
        """Bulk store multiple BlockInfo objects with their symbol IDs."""
        from .index_schema import IndexSchema

        block_ids = []

        with self.transaction() as conn:
            for block_info, symbol_id in block_infos:
                # Get symbol_uid for generating block_uid
                cursor = conn.execute(
                    "SELECT symbol_uid FROM symbols WHERE symbol_id = ?", (symbol_id,)
                )
                symbol_row = cursor.fetchone()
                if not symbol_row:
                    raise ValueError(f"Symbol not found in database: {symbol_id}")
                symbol_uid = symbol_row[0]

                # Generate block UID
                block_uid = IndexSchema.generate_block_uid(
                    symbol_uid,
                    block_info.block_type.value,
                    block_info.file_reference.line_start,
                    block_info.file_reference.line_end,
                )

                cursor = conn.execute(
                    """
                    INSERT OR REPLACE INTO blocks (
                        block_uid, symbol_id, block_type, line_start, line_end, nesting_level,
                        lines_of_code, statement_count, ast_fingerprint,
                        token_fingerprint, normalized_fingerprint, is_extractable,
                        min_clone_size, confidence, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        block_uid,
                        symbol_id,
                        block_info.block_type.value,
                        block_info.file_reference.line_start,
                        block_info.file_reference.line_end,
                        block_info.nesting_level,
                        block_info.lines_of_code,
                        block_info.statement_count,
                        block_info.ast_fingerprint,
                        block_info.token_fingerprint,
                        block_info.normalized_fingerprint,
                        block_info.is_extractable,
                        block_info.min_clone_size,
                        block_info.confidence,
                        json.dumps(block_info.metadata),
                    ),
                )

                block_ids.append(cursor.lastrowid)

        return block_ids

    def get_deep_method_info(self, qualified_name: str) -> Optional[DeepMethodInfo]:
        """Retrieve a DeepMethodInfo object by qualified name."""
        from .models import (
            DeepMethodInfo,
            FileReference,
            parse_semantic_category,
            parse_responsibility_markers,
        )

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT s.symbol_id, s.file_id, s.name, s.qualified_name, s.kind, 
                       s.line_start, s.line_end, s.signature, s.complexity_score, 
                       s.is_public, s.is_async, s.is_property, s.is_static,
                       s.is_classmethod, s.ast_fingerprint, s.token_fingerprint, 
                       s.operation_signature, s.semantic_category, s.responsibility_markers, 
                       s.side_effects, s.calls_external, s.uses_attributes, s.imports_used, 
                       s.confidence, s.analysis_version, s.metadata, f.file_path
                FROM symbols s
                JOIN files f ON s.file_id = f.file_id
                WHERE s.qualified_name = ? AND s.kind = 'method'
            """,
                (qualified_name,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            # Parse the row data (27 columns total)
            (
                symbol_id,
                file_id,
                name,
                qualified_name,
                kind,
                line_start,
                line_end,
                signature,
                complexity_score,
                is_public,
                is_async,
                is_property,
                is_static,
                is_classmethod,
                ast_fingerprint,
                token_fingerprint,
                operation_signature,
                semantic_category,
                responsibility_markers,
                side_effects,
                calls_external,
                uses_attributes,
                imports_used,
                confidence,
                analysis_version,
                metadata,
                file_path,
            ) = row

            # Create FileReference
            file_ref = FileReference(file_path, line_start, line_end)

            # Parse JSON fields
            responsibility_markers_set = parse_responsibility_markers(
                json.loads(responsibility_markers) if responsibility_markers else []
            )
            side_effects_set = set(json.loads(side_effects) if side_effects else [])
            calls_external_list = json.loads(calls_external) if calls_external else []
            uses_attributes_list = json.loads(uses_attributes) if uses_attributes else []
            imports_used_list = json.loads(imports_used) if imports_used else []
            metadata_dict = json.loads(metadata) if metadata else {}

            return DeepMethodInfo(
                name=name,
                qualified_name=qualified_name,
                file_reference=file_ref,
                signature=signature,
                ast_fingerprint=ast_fingerprint,
                token_fingerprint=token_fingerprint,
                operation_signature=operation_signature,
                semantic_category=parse_semantic_category(semantic_category),
                responsibility_markers=responsibility_markers_set,
                side_effects=side_effects_set,
                complexity_score=complexity_score or 0,
                cyclomatic_complexity=complexity_score or 0,
                is_public=bool(is_public),
                is_async=bool(is_async),
                is_property=bool(is_property),
                is_static=bool(is_static),
                is_classmethod=bool(is_classmethod),
                calls_external=calls_external_list,
                uses_attributes=uses_attributes_list,
                imports_used=imports_used_list,
                confidence=confidence or 1.0,
                analysis_version=analysis_version or "1.0",
                metadata=metadata_dict,
            )

    def get_all_deep_method_infos(self) -> List[DeepMethodInfo]:
        """Retrieve all DeepMethodInfo objects from the database."""
        from .models import (
            DeepMethodInfo,
            FileReference,
            SemanticCategory,
            ResponsibilityMarker,
        )

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT s.qualified_name, s.name, s.kind, s.line_start, s.line_end,
                       s.complexity_score, s.signature, s.semantic_category,
                       s.responsibility_markers, s.ast_fingerprint,
                       f.file_path
                FROM symbols s
                JOIN files f ON s.file_id = f.file_id
                WHERE s.kind = 'method'
                ORDER BY s.qualified_name
            """
            )

            methods = []
            for row in cursor.fetchall():
                (
                    qualified_name,
                    name,
                    kind,
                    line_start,
                    line_end,
                    complexity_score,
                    signature,
                    semantic_category_str,
                    responsibility_markers_json,
                    ast_fingerprint,
                    file_path,
                ) = row

                # Parse semantic category
                semantic_category = SemanticCategory.TRANSFORMATION  # Default
                if semantic_category_str:
                    try:
                        semantic_category = SemanticCategory(semantic_category_str)
                    except ValueError:
                        pass

                # Parse responsibility markers
                responsibility_markers = set()
                if responsibility_markers_json:
                    try:
                        markers_data = json.loads(responsibility_markers_json)
                        for marker_name in markers_data:
                            try:
                                responsibility_markers.add(ResponsibilityMarker(marker_name))
                            except ValueError:
                                pass  # Skip invalid markers
                    except json.JSONDecodeError:
                        pass

                # Create method info
                method_info = DeepMethodInfo(
                    name=name,
                    qualified_name=qualified_name,
                    file_reference=FileReference(file_path, line_start, line_end),
                    signature=signature or f"{name}()",
                    ast_fingerprint=ast_fingerprint or "",
                    token_fingerprint="",  # nosec B106 - not a password, placeholder for schema
                    operation_signature="",  # nosec B106 - placeholder
                    semantic_category=semantic_category,
                    responsibility_markers=responsibility_markers,
                    complexity_score=complexity_score or 0,
                )

                methods.append(method_info)

            return methods

    def validate_model_consistency(self) -> List[str]:
        """Validate consistency of stored models and return any issues found."""
        issues = []

        with self._get_connection() as conn:
            # Check for orphaned symbols (symbols without files)
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM symbols s
                LEFT JOIN files f ON s.file_id = f.file_id
                WHERE f.file_id IS NULL
            """
            )
            orphaned_symbols = cursor.fetchone()[0]
            if orphaned_symbols > 0:
                issues.append(
                    f"Found {orphaned_symbols} orphaned symbols without corresponding files"
                )

            # Check for orphaned blocks (blocks without symbols)
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM blocks b
                LEFT JOIN symbols s ON b.symbol_id = s.symbol_id
                WHERE s.symbol_id IS NULL
            """
            )
            orphaned_blocks = cursor.fetchone()[0]
            if orphaned_blocks > 0:
                issues.append(
                    f"Found {orphaned_blocks} orphaned blocks without corresponding symbols"
                )

            # Check for orphaned dependencies
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM dependencies d
                LEFT JOIN symbols s ON d.source_symbol_id = s.symbol_id
                WHERE s.symbol_id IS NULL
            """
            )
            orphaned_deps = cursor.fetchone()[0]
            if orphaned_deps > 0:
                issues.append(
                    f"Found {orphaned_deps} orphaned dependencies without corresponding source symbols"
                )

            # Check for invalid confidence values
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM symbols
                WHERE confidence < 0 OR confidence > 1
            """
            )
            invalid_confidence = cursor.fetchone()[0]
            if invalid_confidence > 0:
                issues.append(f"Found {invalid_confidence} symbols with invalid confidence values")

        return issues

    def get_schema_version(self) -> str:
        """Get the current schema version."""
        with self._get_connection() as conn:
            try:
                # First try the schema_info table (used by IndexSchema.create_database)
                cursor = conn.execute("SELECT value FROM schema_info WHERE key = 'version'")
                row = cursor.fetchone()
                if row:
                    return row[0]

                # Fallback to schema_version table if it exists
                cursor = conn.execute(
                    "SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1"
                )
                row = cursor.fetchone()
                return row[0] if row else "1.0"
            except sqlite3.OperationalError:
                # Neither table exists
                return "1.0"

    def update_schema_version(self, version: str) -> None:
        """Update the schema version."""
        with self.transaction() as conn:
            # Update schema_info table (primary method)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """
            )
            conn.execute(
                "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
                ("version", version),
            )

            # Also update schema_version table for backward compatibility
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version TEXT NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
