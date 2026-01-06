"""
SQLite schema for IntelliRefactor persistent index.

This module defines the database schema that separates facts from derived analysis.
Facts are immutable data extracted from source code, while derived data is computed
on-demand with versioning support.

Architecture principles:
1. Facts vs Derived Data separation
2. Stable symbol UIDs for incremental updates
3. Dependency confidence levels for Python uncertainty
4. Evidence-based analysis with confidence scores
"""

import sqlite3
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import json


class IndexSchema:
    """Manages SQLite schema creation and migration for the persistent index."""
    
    SCHEMA_VERSION = 2
    
    # Facts tables - immutable data extracted from source code
    FACTS_TABLES = {
        'files': '''
            CREATE TABLE IF NOT EXISTS files (
                file_id INTEGER PRIMARY KEY,
                file_path TEXT UNIQUE NOT NULL,
                content_hash TEXT NOT NULL,
                last_modified TIMESTAMP NOT NULL,
                last_analyzed TIMESTAMP,
                file_size INTEGER NOT NULL,
                lines_of_code INTEGER,
                is_test_file BOOLEAN DEFAULT FALSE,
                encoding TEXT DEFAULT 'utf-8',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''',
        
        'symbols': '''
            CREATE TABLE IF NOT EXISTS symbols (
                symbol_id INTEGER PRIMARY KEY,
                symbol_uid TEXT UNIQUE NOT NULL,  -- stable hash-based ID
                file_id INTEGER NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                qualified_name TEXT NOT NULL,
                kind TEXT NOT NULL,  -- class, method, function, variable
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                signature TEXT,
                ast_fingerprint TEXT,
                token_fingerprint TEXT,
                operation_signature TEXT,
                semantic_category TEXT,  -- validation, transformation, caching, etc.
                responsibility_markers TEXT,  -- JSON array of responsibility keywords
                side_effects TEXT,  -- JSON array of side effects
                calls_external TEXT,  -- JSON array of external calls
                uses_attributes TEXT,  -- JSON array of attributes used
                imports_used TEXT,  -- JSON array of imports used
                is_public BOOLEAN DEFAULT TRUE,
                is_async BOOLEAN DEFAULT FALSE,
                is_property BOOLEAN DEFAULT FALSE,
                is_static BOOLEAN DEFAULT FALSE,
                is_classmethod BOOLEAN DEFAULT FALSE,
                complexity_score INTEGER DEFAULT 0,
                lines_of_code INTEGER DEFAULT 0,
                cyclomatic_complexity INTEGER DEFAULT 0,
                cognitive_complexity INTEGER DEFAULT 0,
                -- Class-specific fields
                methods TEXT,  -- JSON array of method names (for classes)
                attributes TEXT,  -- JSON array of attribute names (for classes)
                class_attributes TEXT,  -- JSON array of class attribute names
                base_classes TEXT,  -- JSON array of base class names
                derived_classes TEXT,  -- JSON array of derived class names
                cohesion_score REAL DEFAULT 0.0,
                method_count INTEGER DEFAULT 0,
                attribute_count INTEGER DEFAULT 0,
                is_abstract BOOLEAN DEFAULT FALSE,
                is_interface BOOLEAN DEFAULT FALSE,
                is_data_class BOOLEAN DEFAULT FALSE,
                is_singleton BOOLEAN DEFAULT FALSE,
                -- Analysis metadata
                confidence REAL DEFAULT 1.0,
                analysis_version TEXT DEFAULT '1.0',
                metadata TEXT,  -- JSON object with additional metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''',
        
        'blocks': '''
            CREATE TABLE IF NOT EXISTS blocks (
                block_id INTEGER PRIMARY KEY,
                block_uid TEXT UNIQUE NOT NULL,  -- stable hash-based ID
                symbol_id INTEGER NOT NULL REFERENCES symbols(symbol_id) ON DELETE CASCADE,
                block_type TEXT NOT NULL,  -- if, for, while, try, statement_group, function_body, class_body
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                lines_of_code INTEGER NOT NULL,
                statement_count INTEGER DEFAULT 0,
                nesting_level INTEGER DEFAULT 0,
                parent_block_id INTEGER REFERENCES blocks(block_id),
                ast_fingerprint TEXT,
                token_fingerprint TEXT,
                normalized_fingerprint TEXT,
                min_clone_size INTEGER DEFAULT 3,
                is_extractable BOOLEAN DEFAULT TRUE,
                confidence REAL DEFAULT 1.0,
                metadata TEXT,  -- JSON object with additional metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''',
        
        'dependencies': '''
            CREATE TABLE IF NOT EXISTS dependencies (
                dependency_id INTEGER PRIMARY KEY,
                source_symbol_id INTEGER NOT NULL REFERENCES symbols(symbol_id) ON DELETE CASCADE,
                target_symbol_id INTEGER REFERENCES symbols(symbol_id) ON DELETE CASCADE,
                target_external TEXT,  -- for external dependencies not in our index
                dependency_kind TEXT NOT NULL,  -- calls, imports, inherits, uses_attr, instantiates
                resolution TEXT NOT NULL DEFAULT 'probable',  -- exact, probable, unknown
                confidence REAL NOT NULL DEFAULT 0.8,  -- 0.0 to 1.0
                evidence_json TEXT,  -- JSON with AST node info, line numbers, etc.
                usage_count INTEGER DEFAULT 1,  -- how many times this dependency occurs
                usage_contexts TEXT,  -- JSON array of usage contexts
                is_critical BOOLEAN DEFAULT FALSE,
                is_circular BOOLEAN DEFAULT FALSE,
                metadata TEXT,  -- JSON object with additional metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''',
        
        'attribute_access': '''
            CREATE TABLE IF NOT EXISTS attribute_access (
                access_id INTEGER PRIMARY KEY,
                symbol_id INTEGER NOT NULL REFERENCES symbols(symbol_id) ON DELETE CASCADE,
                attribute_name TEXT NOT NULL,
                access_type TEXT NOT NULL,  -- read, write, both
                line_number INTEGER NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                evidence_json TEXT,  -- JSON with context info
                count INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        '''
    }
    
    # Derived tables - computed analysis results with versioning
    DERIVED_TABLES = {
        'analysis_runs': '''
            CREATE TABLE IF NOT EXISTS analysis_runs (
                run_id INTEGER PRIMARY KEY,
                run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tool_version TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                project_path TEXT NOT NULL,
                files_analyzed INTEGER DEFAULT 0,
                symbols_found INTEGER DEFAULT 0,
                completed BOOLEAN DEFAULT FALSE,
                metadata_json TEXT  -- JSON with additional run info
            )
        ''',
        
        'problems': '''
            CREATE TABLE IF NOT EXISTS problems (
                problem_id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
                symbol_id INTEGER REFERENCES symbols(symbol_id) ON DELETE CASCADE,
                file_id INTEGER REFERENCES files(file_id) ON DELETE CASCADE,
                problem_type TEXT NOT NULL,  -- god_class, long_method, unused_code, etc.
                severity TEXT NOT NULL,  -- low, medium, high, critical
                confidence REAL NOT NULL DEFAULT 0.8,
                description TEXT NOT NULL,
                evidence_json TEXT NOT NULL,  -- JSON with supporting evidence
                recommendation TEXT,
                line_start INTEGER,
                line_end INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''',
        
        'duplicate_groups': '''
            CREATE TABLE IF NOT EXISTS duplicate_groups (
                group_id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
                group_type TEXT NOT NULL,  -- method_duplicate, block_clone, semantic_similar
                similarity_score REAL NOT NULL,
                detection_method TEXT NOT NULL,  -- exact, structural, normalized, semantic
                recommendation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''',
        
        'duplicate_members': '''
            CREATE TABLE IF NOT EXISTS duplicate_members (
                member_id INTEGER PRIMARY KEY,
                group_id INTEGER NOT NULL REFERENCES duplicate_groups(group_id) ON DELETE CASCADE,
                symbol_id INTEGER REFERENCES symbols(symbol_id) ON DELETE CASCADE,
                block_id INTEGER REFERENCES blocks(block_id) ON DELETE CASCADE,
                similarity_score REAL NOT NULL,
                evidence_json TEXT  -- JSON with specific evidence for this member
            )
        ''',
        
        'refactoring_decisions': '''
            CREATE TABLE IF NOT EXISTS refactoring_decisions (
                decision_id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
                decision_type TEXT NOT NULL,  -- extract_method, split_class, remove_unused, etc.
                priority INTEGER NOT NULL DEFAULT 50,  -- 1-100
                confidence REAL NOT NULL DEFAULT 0.8,
                target_symbol_id INTEGER REFERENCES symbols(symbol_id) ON DELETE CASCADE,
                description TEXT NOT NULL,
                rationale TEXT NOT NULL,
                steps_json TEXT NOT NULL,  -- JSON array of implementation steps
                estimated_effort TEXT,  -- trivial, easy, medium, hard
                risk_level TEXT DEFAULT 'medium',  -- low, medium, high
                prerequisites_json TEXT,  -- JSON array of prerequisite decisions
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        '''
    }
    
    # Indexes for performance optimization
    INDEXES = [
        # Facts table indexes
        'CREATE INDEX IF NOT EXISTS idx_files_content_hash ON files(content_hash)',
        'CREATE INDEX IF NOT EXISTS idx_files_last_modified ON files(last_modified)',
        'CREATE INDEX IF NOT EXISTS idx_symbols_file_id ON symbols(file_id)',
        'CREATE INDEX IF NOT EXISTS idx_symbols_qualified_name ON symbols(qualified_name)',
        'CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind)',
        'CREATE INDEX IF NOT EXISTS idx_symbols_ast_fingerprint ON symbols(ast_fingerprint)',
        'CREATE INDEX IF NOT EXISTS idx_symbols_token_fingerprint ON symbols(token_fingerprint)',
        'CREATE INDEX IF NOT EXISTS idx_symbols_semantic_category ON symbols(semantic_category)',
        'CREATE INDEX IF NOT EXISTS idx_symbols_complexity ON symbols(cyclomatic_complexity)',
        'CREATE INDEX IF NOT EXISTS idx_symbols_confidence ON symbols(confidence)',
        'CREATE INDEX IF NOT EXISTS idx_blocks_symbol_id ON blocks(symbol_id)',
        'CREATE INDEX IF NOT EXISTS idx_blocks_type ON blocks(block_type)',
        'CREATE INDEX IF NOT EXISTS idx_blocks_ast_fingerprint ON blocks(ast_fingerprint)',
        'CREATE INDEX IF NOT EXISTS idx_blocks_normalized_fingerprint ON blocks(normalized_fingerprint)',
        'CREATE INDEX IF NOT EXISTS idx_blocks_extractable ON blocks(is_extractable)',
        'CREATE INDEX IF NOT EXISTS idx_dependencies_source ON dependencies(source_symbol_id)',
        'CREATE INDEX IF NOT EXISTS idx_dependencies_target ON dependencies(target_symbol_id)',
        'CREATE INDEX IF NOT EXISTS idx_dependencies_kind ON dependencies(dependency_kind)',
        'CREATE INDEX IF NOT EXISTS idx_dependencies_resolution ON dependencies(resolution)',
        'CREATE INDEX IF NOT EXISTS idx_dependencies_critical ON dependencies(is_critical)',
        'CREATE INDEX IF NOT EXISTS idx_attribute_access_symbol ON attribute_access(symbol_id)',
        'CREATE INDEX IF NOT EXISTS idx_attribute_access_name ON attribute_access(attribute_name)',
        
        # Derived table indexes
        'CREATE INDEX IF NOT EXISTS idx_problems_run_id ON problems(run_id)',
        'CREATE INDEX IF NOT EXISTS idx_problems_symbol_id ON problems(symbol_id)',
        'CREATE INDEX IF NOT EXISTS idx_problems_type ON problems(problem_type)',
        'CREATE INDEX IF NOT EXISTS idx_problems_severity ON problems(severity)',
        'CREATE INDEX IF NOT EXISTS idx_duplicate_groups_run_id ON duplicate_groups(run_id)',
        'CREATE INDEX IF NOT EXISTS idx_duplicate_groups_type ON duplicate_groups(group_type)',
        'CREATE INDEX IF NOT EXISTS idx_duplicate_members_group_id ON duplicate_members(group_id)',
        'CREATE INDEX IF NOT EXISTS idx_duplicate_members_symbol_id ON duplicate_members(symbol_id)',
        'CREATE INDEX IF NOT EXISTS idx_refactoring_decisions_run_id ON refactoring_decisions(run_id)',
        'CREATE INDEX IF NOT EXISTS idx_refactoring_decisions_priority ON refactoring_decisions(priority)',
        'CREATE INDEX IF NOT EXISTS idx_refactoring_decisions_target ON refactoring_decisions(target_symbol_id)'
    ]
    
    @classmethod
    def create_database(cls, db_path) -> sqlite3.Connection:
        """Create a new database with the complete schema."""
        db_path = Path(db_path) if isinstance(db_path, str) else db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(db_path))
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA journal_mode = WAL')  # Better concurrency
        
        # Create facts tables
        for table_name, schema in cls.FACTS_TABLES.items():
            conn.execute(schema)
        
        # Create derived tables
        for table_name, schema in cls.DERIVED_TABLES.items():
            conn.execute(schema)
        
        # Create indexes
        for index_sql in cls.INDEXES:
            conn.execute(index_sql)
        
        # Store schema version
        conn.execute('''
            CREATE TABLE IF NOT EXISTS schema_info (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        ''')
        conn.execute(
            'INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)',
            ('version', str(cls.SCHEMA_VERSION))
        )
        
        conn.commit()
        return conn
    
    @classmethod
    def generate_symbol_uid(cls, file_path: str, qualified_name: str, 
                           kind: str, line_start: int, signature: str = '') -> str:
        """Generate stable UID for a symbol based on its characteristics."""
        # Create a stable identifier that doesn't depend on insertion order
        uid_string = f"{file_path}::{qualified_name}::{kind}::{line_start}::{signature}"
        return hashlib.sha256(uid_string.encode('utf-8')).hexdigest()[:16]
    
    @classmethod
    def generate_block_uid(cls, symbol_uid: str, kind: str, 
                          line_start: int, line_end: int) -> str:
        """Generate stable UID for a code block."""
        uid_string = f"{symbol_uid}::{kind}::{line_start}::{line_end}"
        return hashlib.sha256(uid_string.encode('utf-8')).hexdigest()[:16]
    
    @classmethod
    def validate_dependency_resolution(cls, resolution: str) -> bool:
        """Validate dependency resolution level."""
        return resolution in ['exact', 'probable', 'unknown']
    
    @classmethod
    def validate_confidence(cls, confidence: float) -> bool:
        """Validate confidence score."""
        return 0.0 <= confidence <= 1.0
    
    @classmethod
    def create_evidence_json(cls, evidence_data: Dict[str, Any]) -> str:
        """Create JSON string for evidence data."""
        return json.dumps(evidence_data, ensure_ascii=False, separators=(',', ':'))
    
    @classmethod
    def get_schema_version(cls, conn: sqlite3.Connection) -> Optional[int]:
        """Get current schema version from database."""
        try:
            cursor = conn.execute(
                'SELECT value FROM schema_info WHERE key = ?', 
                ('version',)
            )
            result = cursor.fetchone()
            return int(result[0]) if result else None
        except sqlite3.OperationalError:
            return None
    
    @classmethod
    def needs_migration(cls, conn: sqlite3.Connection) -> bool:
        """Check if database needs schema migration."""
        current_version = cls.get_schema_version(conn)
        return current_version is None or current_version < cls.SCHEMA_VERSION


# Utility functions for working with the schema

def create_file_record(file_path: str, content_hash: str, file_size: int, 
                      lines_of_code: int, is_test_file: bool = False) -> Dict[str, Any]:
    """Create a file record for insertion."""
    return {
        'file_path': file_path,
        'content_hash': content_hash,
        'last_modified': datetime.now(),
        'file_size': file_size,
        'lines_of_code': lines_of_code,
        'is_test_file': is_test_file
    }


def create_symbol_record(file_id: int, name: str, qualified_name: str, 
                        kind: str, line_start: int, line_end: int,
                        signature: str = '', ast_fingerprint: str = '',
                        semantic_category: str = '', 
                        responsibility_markers: List[str] = None) -> Dict[str, Any]:
    """Create a symbol record for insertion."""
    file_path = f"file_{file_id}"  # This would be resolved from file_id in practice
    symbol_uid = IndexSchema.generate_symbol_uid(
        file_path, qualified_name, kind, line_start, signature
    )
    
    return {
        'symbol_uid': symbol_uid,
        'file_id': file_id,
        'name': name,
        'qualified_name': qualified_name,
        'kind': kind,
        'line_start': line_start,
        'line_end': line_end,
        'signature': signature,
        'ast_fingerprint': ast_fingerprint,
        'semantic_category': semantic_category,
        'responsibility_markers': json.dumps(responsibility_markers or [])
    }


def create_dependency_record(source_symbol_id: int, target_symbol_id: Optional[int],
                           target_external: Optional[str], kind: str,
                           resolution: str = 'probable', confidence: float = 0.8,
                           evidence: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a dependency record for insertion."""
    if not IndexSchema.validate_dependency_resolution(resolution):
        raise ValueError(f"Invalid resolution: {resolution}")
    if not IndexSchema.validate_confidence(confidence):
        raise ValueError(f"Invalid confidence: {confidence}")
    
    return {
        'source_symbol_id': source_symbol_id,
        'target_symbol_id': target_symbol_id,
        'target_external': target_external,
        'kind': kind,
        'resolution': resolution,
        'confidence': confidence,
        'evidence_json': IndexSchema.create_evidence_json(evidence or {})
    }


def create_problem_record(run_id: int, problem_type: str, severity: str,
                         confidence: float, description: str, 
                         evidence: Dict[str, Any], symbol_id: Optional[int] = None,
                         file_id: Optional[int] = None) -> Dict[str, Any]:
    """Create a problem record for insertion."""
    if not IndexSchema.validate_confidence(confidence):
        raise ValueError(f"Invalid confidence: {confidence}")
    
    return {
        'run_id': run_id,
        'symbol_id': symbol_id,
        'file_id': file_id,
        'problem_type': problem_type,
        'severity': severity,
        'confidence': confidence,
        'description': description,
        'evidence_json': IndexSchema.create_evidence_json(evidence)
    }