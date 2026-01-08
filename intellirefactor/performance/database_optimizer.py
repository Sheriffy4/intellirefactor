"""
Database optimization utilities for IntelliRefactor.

Provides database query optimization, indexing strategies,
and performance tuning for SQLite databases used by IntelliRefactor.
"""

import sqlite3
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueryPerformanceStats:
    """Statistics for database query performance."""

    query: str
    execution_time: float
    rows_affected: int
    rows_examined: int
    index_usage: str
    optimization_suggestions: List[str]


class DatabaseOptimizer:
    """
    Database optimization utilities for SQLite databases.

    Provides query optimization, index management, and performance
    analysis for IntelliRefactor's SQLite databases.
    """

    def __init__(self, db_path: Path):
        """
        Initialize database optimizer.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None
        self._performance_stats: List[QueryPerformanceStats] = []

    def connect(self):
        """Establish database connection with optimized settings."""
        if self.connection:
            return

        self.connection = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)

        # Enable optimized SQLite settings
        self._apply_performance_settings()
        logger.info(f"Connected to database: {self.db_path}")

    def disconnect(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Disconnected from database")

    def _apply_performance_settings(self):
        """Apply performance-optimized SQLite settings."""
        if not self.connection:
            return

        cursor = self.connection.cursor()

        # Performance optimizations
        performance_settings = [
            # Memory and caching
            "PRAGMA cache_size = 10000",  # 10MB cache
            "PRAGMA temp_store = MEMORY",  # Store temp tables in memory
            "PRAGMA mmap_size = 268435456",  # 256MB memory-mapped I/O
            # Write optimizations
            "PRAGMA synchronous = NORMAL",  # Balance safety and performance
            "PRAGMA journal_mode = WAL",  # Write-Ahead Logging for better concurrency
            "PRAGMA wal_autocheckpoint = 1000",  # Checkpoint every 1000 pages
            # Query optimizations
            "PRAGMA optimize",  # Enable query optimizer
            "PRAGMA analysis_limit = 1000",  # Limit analysis for large tables
            # Connection optimizations
            "PRAGMA busy_timeout = 30000",  # 30 second timeout for busy database
        ]

        for setting in performance_settings:
            try:
                cursor.execute(setting)
                logger.debug(f"Applied setting: {setting}")
            except sqlite3.Error as e:
                logger.warning(f"Failed to apply setting '{setting}': {e}")

        self.connection.commit()
        logger.info("Applied performance settings to database")

    def analyze_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """
        Analyze table statistics for optimization insights.

        Args:
            table_name: Name of table to analyze

        Returns:
            Dictionary containing table statistics and optimization suggestions
        """
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()
        stats = {}

        try:
            # Basic table info
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            stats["row_count"] = row_count

            # Table size estimation
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            stats["column_count"] = len(columns)
            stats["columns"] = [col[1] for col in columns]  # Column names

            # Index information
            cursor.execute(f"PRAGMA index_list({table_name})")
            indexes = cursor.fetchall()
            stats["index_count"] = len(indexes)
            stats["indexes"] = [idx[1] for idx in indexes]  # Index names

            # Analyze query patterns if we have performance stats
            query_patterns = self._analyze_query_patterns_for_table(table_name)
            stats["query_patterns"] = query_patterns

            # Generate optimization suggestions
            suggestions = self._generate_optimization_suggestions(table_name, stats)
            stats["optimization_suggestions"] = suggestions

            logger.info(f"Analyzed table '{table_name}': {row_count} rows, {len(indexes)} indexes")

        except sqlite3.Error as e:
            logger.error(f"Error analyzing table '{table_name}': {e}")
            stats["error"] = str(e)

        return stats

    def _analyze_query_patterns_for_table(self, table_name: str) -> Dict[str, Any]:
        """Analyze query patterns for a specific table."""
        patterns = {
            "select_queries": 0,
            "insert_queries": 0,
            "update_queries": 0,
            "delete_queries": 0,
            "common_where_columns": [],
            "common_join_columns": [],
            "avg_execution_time": 0.0,
        }

        table_stats = [
            stat for stat in self._performance_stats if table_name.lower() in stat.query.lower()
        ]

        if not table_stats:
            return patterns

        # Analyze query types
        for stat in table_stats:
            query_lower = stat.query.lower().strip()
            if query_lower.startswith("select"):
                patterns["select_queries"] += 1
            elif query_lower.startswith("insert"):
                patterns["insert_queries"] += 1
            elif query_lower.startswith("update"):
                patterns["update_queries"] += 1
            elif query_lower.startswith("delete"):
                patterns["delete_queries"] += 1

        # Calculate average execution time
        if table_stats:
            patterns["avg_execution_time"] = sum(s.execution_time for s in table_stats) / len(
                table_stats
            )

        return patterns

    def _generate_optimization_suggestions(
        self, table_name: str, stats: Dict[str, Any]
    ) -> List[str]:
        """Generate optimization suggestions based on table statistics."""
        suggestions = []

        row_count = stats.get("row_count", 0)
        index_count = stats.get("index_count", 0)
        query_patterns = stats.get("query_patterns", {})

        # Suggest indexes for large tables with few indexes
        if row_count > 1000 and index_count < 2:
            suggestions.append(
                f"Consider adding indexes to table '{table_name}' with {row_count} rows"
            )

        # Suggest composite indexes for frequent joins
        if query_patterns.get("select_queries", 0) > query_patterns.get("insert_queries", 0) * 2:
            suggestions.append(
                f"Table '{table_name}' is read-heavy, consider optimizing for SELECT queries"
            )

        # Suggest partitioning for very large tables
        if row_count > 100000:
            suggestions.append(
                f"Table '{table_name}' is very large ({row_count} rows), consider partitioning strategies"
            )

        # Suggest VACUUM for tables with many modifications
        update_ratio = query_patterns.get("update_queries", 0) + query_patterns.get(
            "delete_queries", 0
        )
        if update_ratio > 100:
            suggestions.append(
                f"Table '{table_name}' has many modifications, consider running VACUUM"
            )

        return suggestions

    def create_optimized_indexes(
        self, table_name: str, column_combinations: List[List[str]]
    ) -> List[str]:
        """
        Create optimized indexes for a table.

        Args:
            table_name: Name of table to create indexes for
            column_combinations: List of column combinations to index

        Returns:
            List of created index names
        """
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()
        created_indexes = []

        for i, columns in enumerate(column_combinations):
            if not columns:
                continue

            # Generate index name
            column_suffix = "_".join(columns)
            index_name = f"idx_{table_name}_{column_suffix}"

            # Create index
            columns_str = ", ".join(columns)
            create_sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({columns_str})"

            try:
                start_time = time.time()
                cursor.execute(create_sql)
                execution_time = time.time() - start_time

                created_indexes.append(index_name)
                logger.info(f"Created index '{index_name}' in {execution_time:.2f}s")

            except sqlite3.Error as e:
                logger.error(f"Failed to create index '{index_name}': {e}")

        if created_indexes:
            self.connection.commit()
            logger.info(f"Created {len(created_indexes)} indexes for table '{table_name}'")

        return created_indexes

    def create_performance_indexes(self) -> Dict[str, List[str]]:
        """
        Create performance-optimized indexes for IntelliRefactor tables.

        Returns:
            Dictionary mapping table names to created index names
        """
        # Define optimal index combinations for IntelliRefactor tables
        index_definitions = {
            "files": [["path"], ["content_hash"], ["last_analyzed"], ["is_test"]],
            "symbols": [
                ["file_id"],
                ["qualified_name"],
                ["kind"],
                ["file_id", "kind"],
                ["qualified_name", "kind"],
                ["ast_fingerprint"],
                ["token_fingerprint"],
                ["is_private"],
            ],
            "blocks": [
                ["symbol_id"],
                ["kind"],
                ["symbol_id", "kind"],
                ["block_fingerprint"],
                ["nesting_level"],
            ],
            "dependencies": [
                ["source_symbol_id"],
                ["target_symbol_id"],
                ["kind"],
                ["source_symbol_id", "kind"],
                ["target_symbol_id", "kind"],
            ],
            "attribute_access": [
                ["symbol_id"],
                ["attribute_name"],
                ["access_type"],
                ["symbol_id", "attribute_name"],
            ],
            "problems": [
                ["symbol_id"],
                ["problem_type"],
                ["severity"],
                ["symbol_id", "problem_type"],
            ],
            "duplicate_groups": [["similarity_type"], ["avg_similarity"]],
            "refactoring_decisions": [
                ["target_symbol_id"],
                ["action"],
                ["priority"],
                ["status"],
                ["action", "priority"],
            ],
        }

        results = {}
        for table_name, column_combinations in index_definitions.items():
            try:
                created_indexes = self.create_optimized_indexes(table_name, column_combinations)
                results[table_name] = created_indexes
            except Exception as e:
                logger.error(f"Failed to create indexes for table '{table_name}': {e}")
                results[table_name] = []

        return results

    def optimize_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Analyze and optimize a SQL query.

        Args:
            query: SQL query to optimize

        Returns:
            Tuple of (optimized_query, optimization_suggestions)
        """
        suggestions = []
        optimized_query = query.strip()

        # Basic query optimizations
        query_lower = optimized_query.lower()

        # Suggest using LIMIT for potentially large result sets
        if "select" in query_lower and "limit" not in query_lower and "count(" not in query_lower:
            suggestions.append("Consider adding LIMIT clause to prevent large result sets")

        # Suggest using indexes for WHERE clauses
        if "where" in query_lower:
            suggestions.append("Ensure WHERE clause columns are indexed for better performance")

        # Suggest avoiding SELECT *
        if "select *" in query_lower:
            suggestions.append("Consider selecting specific columns instead of SELECT *")
            # Could automatically replace with specific columns if we know the schema

        # Suggest using EXISTS instead of IN for subqueries
        if " in (" in query_lower and "select" in query_lower:
            suggestions.append(
                "Consider using EXISTS instead of IN with subqueries for better performance"
            )

        # Suggest using INNER JOIN instead of WHERE for joins
        if "where" in query_lower and "=" in query_lower and "join" not in query_lower:
            # This is a heuristic - might be a join condition in WHERE clause
            suggestions.append("Consider using explicit JOIN syntax instead of WHERE clause joins")

        return optimized_query, suggestions

    def execute_with_performance_tracking(
        self, query: str, params: Optional[Tuple] = None
    ) -> QueryPerformanceStats:
        """
        Execute query with performance tracking.

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            Performance statistics for the query
        """
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()

        # Enable query plan analysis
        explain_query = f"EXPLAIN QUERY PLAN {query}"

        start_time = time.time()
        rows_affected = 0
        rows_examined = 0
        index_usage = "unknown"

        try:
            # Get query plan
            cursor.execute(explain_query, params or ())
            query_plan = cursor.fetchall()

            # Analyze query plan for index usage
            plan_text = " ".join(str(row) for row in query_plan)
            if "USING INDEX" in plan_text.upper():
                index_usage = "index_used"
            elif "SCAN TABLE" in plan_text.upper():
                index_usage = "table_scan"
            else:
                index_usage = "unknown"

            # Execute actual query
            cursor.execute(query, params or ())

            # Get result info
            if cursor.description:  # SELECT query
                results = cursor.fetchall()
                rows_affected = len(results)
            else:  # INSERT/UPDATE/DELETE
                rows_affected = cursor.rowcount

            execution_time = time.time() - start_time

            # Generate optimization suggestions
            _, suggestions = self.optimize_query(query)

            # Create performance stats
            stats = QueryPerformanceStats(
                query=query,
                execution_time=execution_time,
                rows_affected=rows_affected,
                rows_examined=rows_examined,  # SQLite doesn't provide this directly
                index_usage=index_usage,
                optimization_suggestions=suggestions,
            )

            # Store stats for analysis
            self._performance_stats.append(stats)

            logger.debug(
                f"Query executed in {execution_time:.3f}s, {rows_affected} rows, {index_usage}"
            )

            return stats

        except sqlite3.Error as e:
            execution_time = time.time() - start_time
            logger.error(f"Query failed after {execution_time:.3f}s: {e}")

            return QueryPerformanceStats(
                query=query,
                execution_time=execution_time,
                rows_affected=0,
                rows_examined=0,
                index_usage="error",
                optimization_suggestions=[f"Query failed: {e}"],
            )

    def vacuum_database(self) -> Dict[str, Any]:
        """
        Perform database VACUUM operation to reclaim space and optimize.

        Returns:
            Dictionary with vacuum operation results
        """
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()

        # Get database size before vacuum
        cursor.execute("PRAGMA page_count")
        pages_before = cursor.fetchone()[0]
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        size_before_mb = (pages_before * page_size) / (1024 * 1024)

        start_time = time.time()

        try:
            # Perform VACUUM
            cursor.execute("VACUUM")

            # Get database size after vacuum
            cursor.execute("PRAGMA page_count")
            pages_after = cursor.fetchone()[0]
            size_after_mb = (pages_after * page_size) / (1024 * 1024)

            execution_time = time.time() - start_time
            space_reclaimed_mb = size_before_mb - size_after_mb

            result = {
                "success": True,
                "execution_time": execution_time,
                "size_before_mb": size_before_mb,
                "size_after_mb": size_after_mb,
                "space_reclaimed_mb": space_reclaimed_mb,
                "pages_before": pages_before,
                "pages_after": pages_after,
            }

            logger.info(
                f"VACUUM completed in {execution_time:.2f}s, reclaimed {space_reclaimed_mb:.2f}MB"
            )
            return result

        except sqlite3.Error as e:
            execution_time = time.time() - start_time
            logger.error(f"VACUUM failed after {execution_time:.2f}s: {e}")

            return {
                "success": False,
                "execution_time": execution_time,
                "error": str(e),
                "size_before_mb": size_before_mb,
            }

    def analyze_database(self) -> Dict[str, Any]:
        """
        Perform comprehensive database analysis.

        Returns:
            Complete database analysis report
        """
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()
        analysis = {
            "database_path": str(self.db_path),
            "analysis_timestamp": time.time(),
            "tables": {},
            "overall_stats": {},
            "recommendations": [],
        }

        try:
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            # Analyze each table
            total_rows = 0
            total_indexes = 0

            for table_name in tables:
                if table_name.startswith("sqlite_"):  # Skip system tables
                    continue

                table_stats = self.analyze_table_statistics(table_name)
                analysis["tables"][table_name] = table_stats

                total_rows += table_stats.get("row_count", 0)
                total_indexes += table_stats.get("index_count", 0)

            # Overall database statistics
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]

            analysis["overall_stats"] = {
                "total_tables": len(tables),
                "total_rows": total_rows,
                "total_indexes": total_indexes,
                "database_size_mb": (page_count * page_size) / (1024 * 1024),
                "page_count": page_count,
                "page_size": page_size,
            }

            # Generate overall recommendations
            recommendations = []

            if total_rows > 10000 and total_indexes < 5:
                recommendations.append(
                    "Database has many rows but few indexes - consider adding more indexes"
                )

            if analysis["overall_stats"]["database_size_mb"] > 100:
                recommendations.append(
                    "Large database detected - consider regular VACUUM operations"
                )

            if len(self._performance_stats) > 100:
                avg_query_time = sum(s.execution_time for s in self._performance_stats) / len(
                    self._performance_stats
                )
                if avg_query_time > 0.1:
                    recommendations.append(
                        f"Average query time is high ({avg_query_time:.3f}s) - review slow queries"
                    )

            analysis["recommendations"] = recommendations

            logger.info(
                f"Database analysis completed: {len(tables)} tables, {total_rows} total rows"
            )

        except sqlite3.Error as e:
            logger.error(f"Database analysis failed: {e}")
            analysis["error"] = str(e)

        return analysis

    def optimize_common_queries(self) -> Dict[str, str]:
        """
        Create optimized versions of common IntelliRefactor queries.

        Returns:
            Dictionary mapping query names to optimized SQL
        """
        optimized_queries = {
            "find_duplicates_by_fingerprint": """
                SELECT s1.qualified_name, s1.file_id, s1.line_start, s1.line_end
                FROM symbols s1
                INNER JOIN symbols s2 ON s1.ast_fingerprint = s2.ast_fingerprint 
                WHERE s1.symbol_id != s2.symbol_id 
                AND s1.ast_fingerprint IS NOT NULL
                ORDER BY s1.ast_fingerprint, s1.qualified_name
            """,
            "find_unused_symbols": """
                SELECT s.qualified_name, s.file_id, s.kind
                FROM symbols s
                LEFT JOIN dependencies d ON s.symbol_id = d.target_symbol_id
                WHERE d.target_symbol_id IS NULL 
                AND s.is_private = 1
                AND s.kind IN ('method', 'function', 'class')
                ORDER BY s.file_id, s.line_start
            """,
            "find_god_classes": """
                SELECT s.qualified_name, s.file_id, COUNT(s2.symbol_id) as method_count
                FROM symbols s
                LEFT JOIN symbols s2 ON s2.file_id = s.file_id 
                    AND s2.qualified_name LIKE s.qualified_name || '.%'
                    AND s2.kind = 'method'
                WHERE s.kind = 'class'
                GROUP BY s.symbol_id, s.qualified_name, s.file_id
                HAVING method_count > 15
                ORDER BY method_count DESC
            """,
            "find_complex_methods": """
                SELECT qualified_name, file_id, line_start, cyclomatic_complexity
                FROM symbols
                WHERE kind = 'method' 
                AND cyclomatic_complexity > 10
                ORDER BY cyclomatic_complexity DESC
            """,
            "find_similar_blocks": """
                SELECT b1.symbol_id, b2.symbol_id, b1.block_fingerprint
                FROM blocks b1
                INNER JOIN blocks b2 ON b1.block_fingerprint = b2.block_fingerprint
                WHERE b1.block_id != b2.block_id
                AND b1.loc >= 3
                ORDER BY b1.block_fingerprint
            """,
            "get_file_analysis_summary": """
                SELECT f.path, f.loc, 
                       COUNT(DISTINCT s.symbol_id) as symbol_count,
                       COUNT(DISTINCT CASE WHEN p.severity = 'high' THEN p.problem_id END) as high_problems,
                       COUNT(DISTINCT CASE WHEN p.severity = 'medium' THEN p.problem_id END) as medium_problems
                FROM files f
                LEFT JOIN symbols s ON f.file_id = s.file_id
                LEFT JOIN problems p ON s.symbol_id = p.symbol_id
                WHERE f.file_id = ?
                GROUP BY f.file_id, f.path, f.loc
            """,
            "get_dependency_graph": """
                SELECT s1.qualified_name as source, s2.qualified_name as target, d.kind, d.count
                FROM dependencies d
                INNER JOIN symbols s1 ON d.source_symbol_id = s1.symbol_id
                INNER JOIN symbols s2 ON d.target_symbol_id = s2.symbol_id
                WHERE d.kind IN ('calls', 'inherits', 'imports')
                ORDER BY s1.qualified_name, d.kind, d.count DESC
            """,
        }

        return optimized_queries

    def create_materialized_views(self) -> List[str]:
        """
        Create materialized views (tables) for expensive queries.

        Returns:
            List of created view names
        """
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()
        created_views = []

        # View for file statistics
        file_stats_sql = """
            CREATE TABLE IF NOT EXISTS file_stats AS
            SELECT f.file_id, f.path, f.loc,
                   COUNT(DISTINCT s.symbol_id) as symbol_count,
                   COUNT(DISTINCT CASE WHEN s.kind = 'class' THEN s.symbol_id END) as class_count,
                   COUNT(DISTINCT CASE WHEN s.kind = 'method' THEN s.symbol_id END) as method_count,
                   COUNT(DISTINCT CASE WHEN s.kind = 'function' THEN s.symbol_id END) as function_count,
                   AVG(CASE WHEN s.cyclomatic_complexity > 0 THEN s.cyclomatic_complexity END) as avg_complexity,
                   COUNT(DISTINCT p.problem_id) as problem_count
            FROM files f
            LEFT JOIN symbols s ON f.file_id = s.file_id
            LEFT JOIN problems p ON s.symbol_id = p.symbol_id
            GROUP BY f.file_id, f.path, f.loc
        """

        # View for symbol relationships
        symbol_relationships_sql = """
            CREATE TABLE IF NOT EXISTS symbol_relationships AS
            SELECT s.symbol_id, s.qualified_name, s.kind,
                   COUNT(DISTINCT d_out.target_symbol_id) as dependencies_out,
                   COUNT(DISTINCT d_in.source_symbol_id) as dependencies_in,
                   COUNT(DISTINCT aa.attribute_name) as attributes_used
            FROM symbols s
            LEFT JOIN dependencies d_out ON s.symbol_id = d_out.source_symbol_id
            LEFT JOIN dependencies d_in ON s.symbol_id = d_in.target_symbol_id
            LEFT JOIN attribute_access aa ON s.symbol_id = aa.symbol_id
            GROUP BY s.symbol_id, s.qualified_name, s.kind
        """

        # View for duplicate analysis
        duplicate_analysis_sql = """
            CREATE TABLE IF NOT EXISTS duplicate_analysis AS
            SELECT ast_fingerprint, COUNT(*) as duplicate_count,
                   GROUP_CONCAT(qualified_name, ';') as duplicate_symbols
            FROM symbols
            WHERE ast_fingerprint IS NOT NULL
            GROUP BY ast_fingerprint
            HAVING duplicate_count > 1
        """

        views = [
            ("file_stats", file_stats_sql),
            ("symbol_relationships", symbol_relationships_sql),
            ("duplicate_analysis", duplicate_analysis_sql),
        ]

        for view_name, sql in views:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {view_name}")
                cursor.execute(sql)
                created_views.append(view_name)
                logger.info(f"Created materialized view: {view_name}")
            except sqlite3.Error as e:
                logger.error(f"Failed to create view '{view_name}': {e}")

        if created_views:
            self.connection.commit()

        return created_views

    def refresh_materialized_views(self) -> Dict[str, bool]:
        """
        Refresh materialized views with current data.

        Returns:
            Dictionary mapping view names to success status
        """
        view_names = ["file_stats", "symbol_relationships", "duplicate_analysis"]
        results = {}

        for view_name in view_names:
            try:
                # Drop and recreate the view
                self.connection.execute(f"DROP TABLE IF EXISTS {view_name}")

                # Recreate based on view name
                if view_name == "file_stats":
                    self.connection.execute(
                        """
                        CREATE TABLE file_stats AS
                        SELECT f.file_id, f.path, f.loc,
                               COUNT(DISTINCT s.symbol_id) as symbol_count,
                               COUNT(DISTINCT CASE WHEN s.kind = 'class' THEN s.symbol_id END) as class_count,
                               COUNT(DISTINCT CASE WHEN s.kind = 'method' THEN s.symbol_id END) as method_count,
                               COUNT(DISTINCT CASE WHEN s.kind = 'function' THEN s.symbol_id END) as function_count,
                               AVG(CASE WHEN s.cyclomatic_complexity > 0 THEN s.cyclomatic_complexity END) as avg_complexity,
                               COUNT(DISTINCT p.problem_id) as problem_count
                        FROM files f
                        LEFT JOIN symbols s ON f.file_id = s.file_id
                        LEFT JOIN problems p ON s.symbol_id = p.symbol_id
                        GROUP BY f.file_id, f.path, f.loc
                    """
                    )
                elif view_name == "symbol_relationships":
                    self.connection.execute(
                        """
                        CREATE TABLE symbol_relationships AS
                        SELECT s.symbol_id, s.qualified_name, s.kind,
                               COUNT(DISTINCT d_out.target_symbol_id) as dependencies_out,
                               COUNT(DISTINCT d_in.source_symbol_id) as dependencies_in,
                               COUNT(DISTINCT aa.attribute_name) as attributes_used
                        FROM symbols s
                        LEFT JOIN dependencies d_out ON s.symbol_id = d_out.source_symbol_id
                        LEFT JOIN dependencies d_in ON s.symbol_id = d_in.target_symbol_id
                        LEFT JOIN attribute_access aa ON s.symbol_id = aa.symbol_id
                        GROUP BY s.symbol_id, s.qualified_name, s.kind
                    """
                    )
                elif view_name == "duplicate_analysis":
                    self.connection.execute(
                        """
                        CREATE TABLE duplicate_analysis AS
                        SELECT ast_fingerprint, COUNT(*) as duplicate_count,
                               GROUP_CONCAT(qualified_name, ';') as duplicate_symbols
                        FROM symbols
                        WHERE ast_fingerprint IS NOT NULL
                        GROUP BY ast_fingerprint
                        HAVING duplicate_count > 1
                    """
                    )

                results[view_name] = True
                logger.info(f"Refreshed materialized view: {view_name}")

            except sqlite3.Error as e:
                logger.error(f"Failed to refresh view '{view_name}': {e}")
                results[view_name] = False

        self.connection.commit()
        return results

    def optimize_for_large_datasets(self) -> Dict[str, Any]:
        """
        Apply optimizations specifically for large datasets.

        Returns:
            Dictionary with optimization results
        """
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()
        optimizations = {}

        try:
            # Increase cache size for large datasets
            cursor.execute("PRAGMA cache_size = 50000")  # 50MB cache
            optimizations["cache_size"] = "50MB"

            # Optimize for bulk operations
            cursor.execute("PRAGMA synchronous = OFF")  # Faster writes, less safe
            cursor.execute("PRAGMA journal_mode = MEMORY")  # Keep journal in memory
            optimizations["bulk_mode"] = True

            # Analyze tables to update statistics
            cursor.execute("ANALYZE")
            optimizations["statistics_updated"] = True

            # Create performance indexes if not exist
            index_results = self.create_performance_indexes()
            optimizations["indexes_created"] = sum(
                len(indexes) for indexes in index_results.values()
            )

            # Create materialized views
            views_created = self.create_materialized_views()
            optimizations["views_created"] = len(views_created)

            self.connection.commit()
            logger.info("Applied large dataset optimizations")

        except sqlite3.Error as e:
            logger.error(f"Failed to apply large dataset optimizations: {e}")
            optimizations["error"] = str(e)

        return optimizations

    def enable_batch_processing_mode(self) -> bool:
        """
        Enable optimizations for batch processing operations.

        Returns:
            True if successful, False otherwise
        """
        if not self.connection:
            self.connect()

        try:
            cursor = self.connection.cursor()

            # Disable auto-commit for batch operations
            cursor.execute("BEGIN TRANSACTION")

            # Optimize for bulk inserts
            cursor.execute("PRAGMA synchronous = OFF")
            cursor.execute("PRAGMA journal_mode = MEMORY")
            cursor.execute("PRAGMA temp_store = MEMORY")
            cursor.execute("PRAGMA cache_size = 100000")  # 100MB cache

            logger.info("Enabled batch processing mode")
            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to enable batch processing mode: {e}")
            return False

    def disable_batch_processing_mode(self) -> bool:
        """
        Disable batch processing mode and restore normal settings.

        Returns:
            True if successful, False otherwise
        """
        if not self.connection:
            return True

        try:
            cursor = self.connection.cursor()

            # Commit any pending transaction
            cursor.execute("COMMIT")

            # Restore normal settings
            cursor.execute("PRAGMA synchronous = NORMAL")
            cursor.execute("PRAGMA journal_mode = WAL")
            cursor.execute("PRAGMA cache_size = 10000")  # 10MB cache

            logger.info("Disabled batch processing mode")
            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to disable batch processing mode: {e}")
            return False

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate performance report from collected statistics.

        Returns:
            Performance report with query statistics and recommendations
        """
        if not self._performance_stats:
            return {"message": "No performance statistics available"}

        # Analyze query performance
        total_queries = len(self._performance_stats)
        total_time = sum(s.execution_time for s in self._performance_stats)
        avg_time = total_time / total_queries

        # Find slow queries (top 10% by execution time)
        sorted_stats = sorted(self._performance_stats, key=lambda s: s.execution_time, reverse=True)
        slow_query_count = max(1, total_queries // 10)
        slow_queries = sorted_stats[:slow_query_count]

        # Index usage analysis
        index_used_count = sum(1 for s in self._performance_stats if s.index_usage == "index_used")
        table_scan_count = sum(1 for s in self._performance_stats if s.index_usage == "table_scan")

        report = {
            "summary": {
                "total_queries": total_queries,
                "total_execution_time": total_time,
                "average_execution_time": avg_time,
                "index_usage_rate": index_used_count / total_queries if total_queries > 0 else 0,
                "table_scan_rate": table_scan_count / total_queries if total_queries > 0 else 0,
            },
            "slow_queries": [
                {
                    "query": s.query[:100] + "..." if len(s.query) > 100 else s.query,
                    "execution_time": s.execution_time,
                    "index_usage": s.index_usage,
                    "suggestions": s.optimization_suggestions,
                }
                for s in slow_queries
            ],
            "recommendations": [],
        }

        # Generate recommendations
        if avg_time > 0.1:
            report["recommendations"].append(
                "Average query time is high - review query optimization"
            )

        if table_scan_count / total_queries > 0.5:
            report["recommendations"].append(
                "Many queries use table scans - consider adding indexes"
            )

        if slow_query_count > 0:
            report["recommendations"].append(
                f"Found {slow_query_count} slow queries - review and optimize"
            )

        return report

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
