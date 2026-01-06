"""
IndexBuilder for IntelliRefactor persistent index.

This module implements the IndexBuilder class that creates and maintains
the SQLite index with incremental updates based on file content hashes.

Architecture principles:
1. Incremental processing - only analyze changed files
2. Bounded memory - batch processing for large projects
3. Facts-only storage - minimal data in SQLite
4. Stable symbol UIDs for consistent updates
"""

import ast
import hashlib
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator, Callable, Union
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
import json

from .index_schema import IndexSchema


@dataclass
class IndexBuildResult:
    """Result of index building operation."""
    success: bool
    files_processed: int
    files_skipped: int
    symbols_found: int
    blocks_found: int
    dependencies_found: int
    errors: List[str]
    build_time_seconds: float
    incremental: bool


@dataclass
class FileAnalysisResult:
    """Result of analyzing a single file."""
    file_path: str
    content_hash: str
    file_size: int
    lines_of_code: int
    is_test_file: bool
    last_modified: float
    symbols: List[Dict[str, Any]]
    blocks: List[Dict[str, Any]]
    dependencies: List[Dict[str, Any]]
    attribute_accesses: List[Dict[str, Any]]
    error: Optional[str] = None


class ContextAwareVisitor(ast.NodeVisitor):
    """
    AST Visitor that tracks scope context to correctly identify 
    qualified names, nesting levels, and avoid recursion issues.
    """
    def __init__(self, file_path: Path, project_root: Path, content: str):
        self.file_path = file_path
        self.project_root = project_root
        self.relative_path = str(file_path.relative_to(project_root))
        self.content_lines = content.splitlines()
        
        # Context tracking: Stack of (name, uid) tuples
        self.scope_stack: List[Tuple[str, str]] = [] 
        self.current_class_node: Optional[ast.ClassDef] = None
        
        # Scope occurrence tracker to handle name collisions (e.g. two 'def foo' in same scope)
        # Key: (parent_uid, name), Value: count
        self.scope_occurrence_map: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Deduplication sets and maps
        self._seen_dependencies: Set[Tuple[str, str, str]] = set()
        # Map (symbol_uid, attribute_name) -> attribute_record for O(1) access and counting
        self._attribute_access_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        
        # Results
        self.symbols: List[Dict[str, Any]] = []
        self.blocks: List[Dict[str, Any]] = []
        self.dependencies: List[Dict[str, Any]] = []
        
        # Create a module-level symbol to attach top-level imports/code
        self._create_module_symbol()

    @property
    def attribute_accesses(self) -> List[Dict[str, Any]]:
        """Returns the list of aggregated attribute accesses."""
        return list(self._attribute_access_map.values())

    def _create_module_symbol(self):
        """Creates a symbol representing the module itself."""
        module_name = self.file_path.stem
        # Module has no parent, so parent_uid is empty string
        symbol_uid = IndexSchema.generate_symbol_uid(
            self.relative_path, module_name, 'module', 0, ''
        )
        
        self.symbols.append({
            'symbol_uid': symbol_uid,
            'name': module_name,
            'qualified_name': module_name,
            'kind': 'module',
            'line_start': 1,
            'line_end': len(self.content_lines),
            'signature': f"module {module_name}",
            'ast_fingerprint': '',
            'token_fingerprint': '',
            'semantic_category': 'module',
            'responsibility_markers': '[]',
            'is_public': not module_name.startswith('_'),
            'is_async': False,
            'is_property': False,
            'is_static': False,
            'is_classmethod': False,
            'complexity_score': 0
        })
        # Push module info to stack as base scope
        self.scope_stack.append((module_name, symbol_uid))

    def _get_current_qualified_name(self, name: str) -> str:
        """Builds qualified name based on current stack."""
        # Skip the module name (first item) for cleaner qualified names in code
        parts = [s[0] for s in self.scope_stack[1:]] + [name]
        return ".".join(parts)

    def _get_current_parent_uid(self) -> str:
        """Gets the UID of the current scope owner (parent symbol)."""
        if not self.scope_stack:
            return ""
        return self.scope_stack[-1][1]

    def _get_occurrence_index(self, name: str) -> int:
        """Gets the occurrence index for a name in the current scope to avoid UID collisions."""
        parent_uid = self._get_current_parent_uid()
        key = (parent_uid, name)
        idx = self.scope_occurrence_map[key]
        self.scope_occurrence_map[key] += 1
        return idx

    def visit_ClassDef(self, node: ast.ClassDef):
        qualified_name = self._get_current_qualified_name(node.name)
        occurrence_idx = self._get_occurrence_index(node.name)
        
        # Generate stable UID. We include occurrence_idx in signature part to disambiguate
        # redefinitions in the same scope without relying on line numbers.
        disambiguator = f":{occurrence_idx}" if occurrence_idx > 0 else ""
        symbol_uid = IndexSchema.generate_symbol_uid(
            self.relative_path, qualified_name, 'class', 0, f"class {node.name}{disambiguator}"
        )
        
        symbol_info = {
            'symbol_uid': symbol_uid,
            'name': node.name,
            'qualified_name': qualified_name,
            'kind': 'class',
            'line_start': node.lineno,
            'line_end': getattr(node, 'end_lineno', node.lineno),
            'signature': f"class {node.name}",
            'ast_fingerprint': self._calculate_ast_fingerprint(node),
            'token_fingerprint': self._calculate_token_fingerprint(node),
            'semantic_category': self._determine_semantic_category(node.name),
            'responsibility_markers': json.dumps(self._extract_responsibility_markers(node)),
            'is_public': not node.name.startswith('_'),
            'is_async': False,
            'is_property': False,
            'is_static': False,
            'is_classmethod': False,
            'complexity_score': 0
        }
        self.symbols.append(symbol_info)
        
        # Extract facts from class body (non-recursive for nested defs)
        self._extract_local_facts(node, symbol_uid)
        
        # Update context and recurse
        self.scope_stack.append((node.name, symbol_uid))
        prev_class_node = self.current_class_node
        self.current_class_node = node
        
        self.generic_visit(node)
        
        self.current_class_node = prev_class_node
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._handle_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._handle_function(node, is_async=True)

    def _handle_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool):
        qualified_name = self._get_current_qualified_name(node.name)
        occurrence_idx = self._get_occurrence_index(node.name)
        
        # Determine kind and flags
        kind = 'function'
        is_static = False
        is_classmethod = False
        
        if self.current_class_node:
            kind = 'method'
            # Check for decorators
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    if dec.id == 'staticmethod': 
                        is_static = True
                    elif dec.id == 'classmethod': 
                        is_classmethod = True
        
        signature = self._get_function_signature(node, is_async)
        
        # Disambiguate UID if needed
        disambiguator = f":{occurrence_idx}" if occurrence_idx > 0 else ""
        symbol_uid = IndexSchema.generate_symbol_uid(
            self.relative_path, qualified_name, kind, 0, signature + disambiguator
        )
        
        symbol_info = {
            'symbol_uid': symbol_uid,
            'name': node.name,
            'qualified_name': qualified_name,
            'kind': kind,
            'line_start': node.lineno,
            'line_end': getattr(node, 'end_lineno', node.lineno),
            'signature': signature,
            'ast_fingerprint': self._calculate_ast_fingerprint(node),
            'token_fingerprint': self._calculate_token_fingerprint(node),
            'semantic_category': self._determine_semantic_category(node.name),
            'responsibility_markers': json.dumps(self._extract_responsibility_markers(node)),
            'is_public': not node.name.startswith('_'),
            'is_async': is_async,
            'is_property': self._is_property(node),
            'is_static': is_static,
            'is_classmethod': is_classmethod,
            'complexity_score': self._calculate_complexity(node)
        }
        self.symbols.append(symbol_info)
        
        # Extract facts from function body
        self._extract_local_facts(node, symbol_uid)
        
        # Update context and recurse
        self.scope_stack.append((node.name, symbol_uid))
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_Import(self, node: ast.Import):
        # Imports belong to the current scope (usually module, sometimes function)
        parent_uid = self._get_current_parent_uid()
        for alias in node.names:
            self._add_dependency(
                source_uid=parent_uid,
                target=alias.name,
                kind='imports',
                resolution='exact',
                confidence=1.0,
                evidence={
                    'line_number': node.lineno,
                    'ast_node_type': 'Import',
                    'module_name': alias.name
                }
            )

    def visit_ImportFrom(self, node: ast.ImportFrom):
        parent_uid = self._get_current_parent_uid()
        
        # Correctly handle relative imports
        # node.level: 0 = absolute, 1 = ., 2 = ..
        base = '.' * (node.level or 0)
        if node.module:
            base += node.module
            
        for alias in node.names:
            if node.module:
                # e.g. from pkg import x -> pkg.x
                # e.g. from .pkg import x -> .pkg.x
                full_target = f"{base}.{alias.name}"
            else:
                # e.g. from . import x -> .x
                full_target = f"{base}{alias.name}"
                
            self._add_dependency(
                source_uid=parent_uid,
                target=full_target,
                kind='imports',
                resolution='exact',
                confidence=1.0,
                evidence={
                    'line_number': node.lineno,
                    'ast_node_type': 'ImportFrom',
                    'module_name': full_target
                }
            )

    def _add_dependency(self, source_uid: str, target: str, kind: str, 
                       resolution: str, confidence: float, evidence: Dict):
        """Adds a dependency with O(1) deduplication."""
        key = (source_uid, target, kind)
        if key not in self._seen_dependencies:
            self._seen_dependencies.add(key)
            self.dependencies.append({
                'source_symbol_uid': source_uid,
                'target_external': target,
                'kind': kind,
                'resolution': resolution,
                'confidence': confidence,
                'evidence': evidence
            })

    def _extract_local_facts(self, parent_node: ast.AST, symbol_uid: str):
        """
        Extracts blocks, calls, and attribute accesses from the node's body.
        Does NOT recurse into nested class/function definitions to avoid duplication.
        """
        # We iterate over child nodes manually to control recursion
        for child in ast.iter_child_nodes(parent_node):
            # Skip nested definitions - they will be visited by generic_visit later
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            
            # Extract Blocks (start nesting at 1)
            self._extract_blocks_recursive(child, symbol_uid, nesting_level=1)
            
            # Extract Dependencies (Calls) and Attributes
            self._extract_usage_recursive(child, symbol_uid)

    def _extract_blocks_recursive(self, node: ast.AST, symbol_uid: str, nesting_level: int):
        """Extracts blocks recursively but stops at nested definitions."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return

        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncFor, ast.AsyncWith)):
            block_kind = type(node).__name__.lower()
            # Generate block UID
            block_uid = IndexSchema.generate_block_uid(
                symbol_uid, block_kind, node.lineno, getattr(node, 'end_lineno', node.lineno)
            )
            
            self.blocks.append({
                'block_uid': block_uid,
                'symbol_uid': symbol_uid,
                'kind': block_kind,
                'line_start': node.lineno,
                'line_end': getattr(node, 'end_lineno', node.lineno),
                'lines_of_code': getattr(node, 'end_lineno', node.lineno) - node.lineno + 1,
                'nesting_level': nesting_level,
                'ast_fingerprint': self._calculate_ast_fingerprint(node),
                'token_fingerprint': self._calculate_token_fingerprint(node),
                'normalized_fingerprint': self._calculate_ast_fingerprint(node) # Simplified
            })
            
            # Increase nesting for children
            next_level = nesting_level + 1
        else:
            next_level = nesting_level

        for child in ast.iter_child_nodes(node):
            self._extract_blocks_recursive(child, symbol_uid, next_level)

    def _extract_usage_recursive(self, node: ast.AST, symbol_uid: str):
        """Extracts calls and attribute accesses recursively, stopping at nested defs."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return

        # Calls
        if isinstance(node, ast.Call):
            target = self._extract_call_target(node)
            if target:
                self._add_dependency(
                    source_uid=symbol_uid,
                    target=target,
                    kind='calls',
                    resolution='probable',
                    confidence=0.7,
                    evidence={
                        'line_number': node.lineno,
                        'ast_node_type': 'Call',
                        'call_target': target
                    }
                )

        # Attribute Access
        elif isinstance(node, ast.Attribute):
            # Use map for O(1) access and aggregation
            key = (symbol_uid, node.attr)
            if key in self._attribute_access_map:
                self._attribute_access_map[key]['count'] += 1
            else:
                self._attribute_access_map[key] = {
                    'symbol_uid': symbol_uid,
                    'attribute_name': node.attr,
                    'access_type': 'read', # Simplified
                    'line_number': node.lineno,
                    'confidence': 1.0,
                    'count': 1,
                    'evidence': {
                        'line_number': node.lineno,
                        'ast_node_type': 'Attribute',
                        'attribute_name': node.attr
                    }
                }

        for child in ast.iter_child_nodes(node):
            self._extract_usage_recursive(child, symbol_uid)

    # --- Helpers ---

    def _get_function_signature(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool) -> str:
        args = [arg.arg for arg in node.args.args]
        async_prefix = "async " if is_async else ""
        return f"{async_prefix}def {node.name}({', '.join(args)})"

    def _calculate_ast_fingerprint(self, node: ast.AST) -> str:
        # Simplified structural fingerprint
        node_types = []
        for child in ast.walk(node):
            node_types.append(type(child).__name__)
        fingerprint_str = '|'.join(node_types) # Keep order for structure
        return hashlib.md5(fingerprint_str.encode('utf-8')).hexdigest()[:16]

    def _calculate_token_fingerprint(self, node: ast.AST) -> str:
        try:
            if hasattr(ast, 'unparse'):
                source = ast.unparse(node)
                # Normalize whitespace
                source = ' '.join(source.split())
                return hashlib.md5(source.encode('utf-8')).hexdigest()[:16]
        except:
            pass
        return ""

    def _determine_semantic_category(self, name: str) -> str:
        name_lower = name.lower()
        if any(k in name_lower for k in ['test', 'check', 'verify', 'validate']): return 'validation'
        if any(k in name_lower for k in ['transform', 'convert', 'parse']): return 'transformation'
        if any(k in name_lower for k in ['save', 'load', 'store', 'db']): return 'persistence'
        if any(k in name_lower for k in ['get', 'set', 'is']): return 'access'
        return 'business_logic'

    def _extract_responsibility_markers(self, node: ast.AST) -> List[str]:
        markers = []
        if hasattr(node, 'name'):
            cat = self._determine_semantic_category(node.name)
            if cat != 'business_logic': markers.append(cat)
        return markers

    def _is_property(self, node: ast.FunctionDef) -> bool:
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id == 'property':
                return True
        return False

    def _calculate_complexity(self, node: ast.AST) -> int:
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def _extract_call_target(self, call_node: ast.Call) -> Optional[str]:
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            # Try to resolve simple attributes like self.method or module.func
            if isinstance(call_node.func.value, ast.Name):
                return f"{call_node.func.value.id}.{call_node.func.attr}"
            return call_node.func.attr
        return None


class IndexBuilder:
    """
    Builds and maintains the persistent SQLite index for IntelliRefactor.
    
    Supports incremental updates based on file content hashes and uses
    batch processing to handle large projects without memory exhaustion.
    """
    
    def __init__(self, db_path: Path, batch_size: int = 100):
        """
        Initialize the IndexBuilder.
        
        Args:
            db_path: Path to the SQLite database file
            batch_size: Number of files to process in each batch
        """
        self.db_path = Path(db_path) if isinstance(db_path, str) else db_path
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
        # Ensure database exists with proper schema
        if not self.db_path.exists():
            self.logger.info(f"Creating new database at {self.db_path}")
            self.conn = IndexSchema.create_database(str(self.db_path))
        else:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.execute('PRAGMA foreign_keys = ON')
            
            # Check if migration is needed
            if IndexSchema.needs_migration(self.conn):
                self.logger.warning("Database schema migration needed - recreating database")
                self.conn.close()
                # Remove old database and create new one
                self.db_path.unlink()
                self.conn = IndexSchema.create_database(str(self.db_path))
    
    def build_index(self, project_path: Path, incremental: bool = True, 
                   progress_callback: Optional[Callable[[float, int, int], None]] = None) -> IndexBuildResult:
        """
        Build or update the index for a project.
        
        Args:
            project_path: Path to the project root
            incremental: If True, only process changed files
            progress_callback: Optional callback for progress reporting
            
        Returns:
            IndexBuildResult with statistics and any errors
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting index build for {project_path} (incremental={incremental})")
            
            # Find all Python files
            python_files = list(project_path.rglob("*.py"))
            self.logger.info(f"Found {len(python_files)} Python files")
            
            # Filter files for incremental update
            if incremental:
                files_to_process = self._filter_changed_files(python_files, project_path)
                files_skipped = len(python_files) - len(files_to_process)
                
                # Clean up orphaned records for files that no longer exist
                orphaned_count = self._cleanup_orphaned_records(project_path, python_files)
                if orphaned_count > 0:
                    self.logger.info(f"Cleaned up {orphaned_count} orphaned file records")
            else:
                files_to_process = python_files
                files_skipped = 0
            
            self.logger.info(f"Processing {len(files_to_process)} files (skipped {files_skipped})")
            
            # Process files in batches
            total_symbols = 0
            total_blocks = 0
            total_dependencies = 0
            errors = []
            processed_files = 0
            
            for batch_idx, batch in enumerate(self._batch_files(files_to_process)):
                batch_result = self._process_file_batch(batch, project_path)
                total_symbols += batch_result['symbols']
                total_blocks += batch_result['blocks']
                total_dependencies += batch_result['dependencies']
                errors.extend(batch_result['errors'])
                processed_files += batch_result['files']
                
                # Report progress if callback provided
                if progress_callback:
                    progress = processed_files / len(files_to_process) if files_to_process else 1.0
                    progress_callback(progress, processed_files, len(files_to_process))
                
                # Commit after each batch to avoid memory issues
                self.conn.commit()
                
                self.logger.info(f"Processed batch {batch_idx + 1}: {batch_result['files']} files, "
                               f"{batch_result['symbols']} symbols, "
                               f"{batch_result['blocks']} blocks")
            
            # Final commit
            self.conn.commit()
            
            build_time = (datetime.now() - start_time).total_seconds()
            
            result = IndexBuildResult(
                success=len(errors) == 0,
                files_processed=len(files_to_process),
                files_skipped=files_skipped,
                symbols_found=total_symbols,
                blocks_found=total_blocks,
                dependencies_found=total_dependencies,
                errors=errors,
                build_time_seconds=build_time,
                incremental=incremental
            )
            
            self.logger.info(f"Index build completed in {build_time:.2f}s: "
                           f"{result.files_processed} files, {result.symbols_found} symbols")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Index build failed: {e}")
            return IndexBuildResult(
                success=False,
                files_processed=0,
                files_skipped=0,
                symbols_found=0,
                blocks_found=0,
                dependencies_found=0,
                errors=[str(e)],
                build_time_seconds=(datetime.now() - start_time).total_seconds(),
                incremental=incremental
            )
    
    def rebuild_index(self, project_path: Path, 
                     progress_callback: Optional[Callable[[float, int, int], None]] = None) -> IndexBuildResult:
        """
        Completely rebuild the index from scratch.
        
        Args:
            project_path: Path to the project root
            progress_callback: Optional callback for progress reporting
            
        Returns:
            IndexBuildResult with statistics and any errors
        """
        self.logger.info("Rebuilding index from scratch")
        
        # Clear all existing data
        self._clear_index()
        
        # Build fresh index
        return self.build_index(project_path, incremental=False, progress_callback=progress_callback)

    def _filter_changed_files(self, python_files: List[Path], project_root: Path) -> List[Path]:
        """Filter files that have changed since last analysis using bounded memory approach."""
        changed_files = []
        
        # Process files in chunks to avoid loading all hashes into memory
        chunk_size = 500
        
        for i in range(0, len(python_files), chunk_size):
            chunk = python_files[i:i + chunk_size]
            
            # Prepare relative paths for query
            rel_paths = []
            path_map = {}
            
            for f in chunk:
                try:
                    rel = str(f.relative_to(project_root))
                    rel_paths.append(rel)
                    path_map[rel] = f
                except ValueError:
                    continue
            
            if not rel_paths:
                continue
                
            # Query DB for this chunk
            placeholders = ','.join('?' * len(rel_paths))
            cursor = self.conn.execute(
                f'SELECT file_path, content_hash FROM files WHERE file_path IN ({placeholders})',
                rel_paths
            )
            
            db_hashes = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Compare hashes
            for rel_path in rel_paths:
                file_path = path_map[rel_path]
                
                try:
                    # Skip very large files to avoid memory issues
                    file_size = file_path.stat().st_size
                    if file_size > 10 * 1024 * 1024:  # Skip files larger than 10MB
                        self.logger.warning(f"Skipping large file {file_path} ({file_size} bytes)")
                        continue
                    
                    # Calculate current content hash
                    with open(file_path, 'r', encoding='utf-8-sig') as f:
                        content = f.read()
                    current_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                    
                    if rel_path not in db_hashes or db_hashes[rel_path] != current_hash:
                        changed_files.append(file_path)
                        self.logger.debug(f"File changed: {rel_path}")
                    else:
                        self.logger.debug(f"File unchanged: {rel_path}")
                        
                except (UnicodeDecodeError, PermissionError) as e:
                    self.logger.warning(f"Cannot read file {file_path}: {e}")
                    changed_files.append(file_path)
                except Exception as e:
                    self.logger.error(f"Unexpected error processing {file_path}: {e}")
                    changed_files.append(file_path)
        
        self.logger.info(f"Incremental update: {len(changed_files)} changed files out of {len(python_files)} total")
        return changed_files
    
    def _cleanup_orphaned_records(self, project_root: Path, current_files: List[Path]) -> int:
        """Clean up database records for files that no longer exist."""
        # Get all files currently in the database
        cursor = self.conn.execute('SELECT file_path FROM files')
        db_files = {row[0] for row in cursor.fetchall()}
        
        # Get current files as relative paths
        current_relative_paths = {str(f.relative_to(project_root)) for f in current_files}
        
        # Find orphaned files (in DB but not on disk)
        orphaned_files = db_files - current_relative_paths
        
        if orphaned_files:
            self.logger.info(f"Cleaning up {len(orphaned_files)} orphaned file records")
            for orphaned_file in orphaned_files:
                # Delete file and all related records (cascading deletes will handle symbols, blocks, etc.)
                self.conn.execute('DELETE FROM files WHERE file_path = ?', (orphaned_file,))
        
        return len(orphaned_files)
    
    def _batch_files(self, files: List[Path]) -> Iterator[List[Path]]:
        """Split files into batches for processing."""
        for i in range(0, len(files), self.batch_size):
            yield files[i:i + self.batch_size]
    
    def _process_file_batch(self, batch: List[Path], project_root: Path) -> Dict[str, Any]:
        """Process a batch of files and update the database."""
        batch_stats = {
            'files': 0,
            'symbols': 0,
            'blocks': 0,
            'dependencies': 0,
            'errors': []
        }
        
        for file_path in batch:
            try:
                # Analyze the file
                analysis = self._analyze_file(file_path, project_root)
                
                if analysis.error:
                    batch_stats['errors'].append(f"{file_path}: {analysis.error}")
                    continue
                
                # Store file record
                file_id = self._store_file_record(analysis)
                
                # Store symbols
                symbol_ids = self._store_symbols(file_id, analysis.symbols)
                batch_stats['symbols'] += len(symbol_ids)
                
                # Store blocks
                self._store_blocks(symbol_ids, analysis.blocks)
                batch_stats['blocks'] += len(analysis.blocks)
                
                # Store dependencies
                self._store_dependencies(symbol_ids, analysis.dependencies)
                batch_stats['dependencies'] += len(analysis.dependencies)
                
                # Store attribute accesses
                self._store_attribute_accesses(symbol_ids, analysis.attribute_accesses)
                
                batch_stats['files'] += 1
                
            except Exception as e:
                batch_stats['errors'].append(f"{file_path}: {str(e)}")
                self.logger.error(f"Error processing {file_path}: {e}")
        
        return batch_stats
    
    def _analyze_file(self, file_path: Path, project_root: Path) -> FileAnalysisResult:
        """Analyze a single Python file using ContextAwareVisitor."""
        try:
            # Check file size to avoid memory issues with massive files
            file_size = file_path.stat().st_size
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                return FileAnalysisResult(
                    file_path=str(file_path.relative_to(project_root)),
                    content_hash="",
                    file_size=file_size,
                    lines_of_code=0,
                    is_test_file=self._is_test_file(file_path),
                    last_modified=file_path.stat().st_mtime,
                    symbols=[],
                    blocks=[],
                    dependencies=[],
                    attribute_accesses=[],
                    error="File too large (>10MB)"
                )

            with open(file_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
            
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            last_modified = file_path.stat().st_mtime
            
            # Consistent line counting
            lines_of_code = len([line for line in content.splitlines() if line.strip()])
            
            # Parse AST
            try:
                # Use the visitor to extract everything
                visitor = ContextAwareVisitor(file_path, project_root, content)
                tree = ast.parse(content)
                visitor.visit(tree)
                
                return FileAnalysisResult(
                    file_path=str(file_path.relative_to(project_root)),
                    content_hash=content_hash,
                    file_size=len(content.encode('utf-8')),
                    lines_of_code=lines_of_code,
                    is_test_file=self._is_test_file(file_path),
                    last_modified=last_modified,
                    symbols=visitor.symbols,
                    blocks=visitor.blocks,
                    dependencies=visitor.dependencies,
                    attribute_accesses=visitor.attribute_accesses
                )
                
            except SyntaxError as e:
                return FileAnalysisResult(
                    file_path=str(file_path.relative_to(project_root)),
                    content_hash=content_hash,
                    file_size=len(content.encode('utf-8')),
                    lines_of_code=lines_of_code, # Consistent counting
                    is_test_file=self._is_test_file(file_path),
                    last_modified=last_modified,
                    symbols=[],
                    blocks=[],
                    dependencies=[],
                    attribute_accesses=[],
                    error=f"Syntax error: {e}"
                )
            
        except Exception as e:
            return FileAnalysisResult(
                file_path=str(file_path.relative_to(project_root)),
                content_hash="",
                file_size=0,
                lines_of_code=0,
                is_test_file=False,
                last_modified=0.0,
                symbols=[],
                blocks=[],
                dependencies=[],
                attribute_accesses=[],
                error=str(e)
            )
    
    def _is_test_file(self, file_path: Path) -> bool:
        """Determine if a file is a test file."""
        return (
            file_path.name.startswith('test_') or
            file_path.name.endswith('_test.py') or
            'test' in file_path.parts
        )
    
    def _store_file_record(self, analysis: FileAnalysisResult) -> int:
        """Store file record and return file_id."""
        # Delete existing record if it exists (for updates)
        self.conn.execute('DELETE FROM files WHERE file_path = ?', (analysis.file_path,))
        
        now = datetime.now()
        cursor = self.conn.execute('''
            INSERT INTO files (file_path, content_hash, last_modified, last_analyzed, file_size, lines_of_code, is_test_file)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            analysis.file_path,
            analysis.content_hash,
            datetime.fromtimestamp(analysis.last_modified),
            now,  # Set last_analyzed to current time
            analysis.file_size,
            analysis.lines_of_code,
            analysis.is_test_file
        ))
        
        return cursor.lastrowid
    
    def _store_symbols(self, file_id: int, symbols: List[Dict[str, Any]]) -> Dict[str, int]:
        """Store symbols and return mapping of symbol_uid to symbol_id."""
        symbol_ids = {}
        
        for symbol in symbols:
            # Delete existing symbol if it exists (for updates)
            self.conn.execute('DELETE FROM symbols WHERE symbol_uid = ?', (symbol['symbol_uid'],))
            
            cursor = self.conn.execute('''
                INSERT INTO symbols (
                    symbol_uid, file_id, name, qualified_name, kind, line_start, line_end,
                    signature, ast_fingerprint, token_fingerprint, semantic_category,
                    responsibility_markers, is_public, is_async, is_property, 
                    is_static, is_classmethod, complexity_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol['symbol_uid'], file_id, symbol['name'], symbol['qualified_name'],
                symbol['kind'], symbol['line_start'], symbol['line_end'], symbol['signature'],
                symbol['ast_fingerprint'], symbol['token_fingerprint'], symbol['semantic_category'],
                symbol['responsibility_markers'], symbol['is_public'], symbol['is_async'],
                symbol['is_property'], symbol['is_static'], symbol['is_classmethod'], 
                symbol['complexity_score']
            ))
            
            symbol_ids[symbol['symbol_uid']] = cursor.lastrowid
        
        return symbol_ids
    
    def _store_blocks(self, symbol_ids: Dict[str, int], blocks: List[Dict[str, Any]]) -> None:
        """Store code blocks."""
        for block in blocks:
            symbol_id = symbol_ids.get(block['symbol_uid'])
            if not symbol_id:
                continue
            
            # Delete existing block if it exists (for updates)
            self.conn.execute('DELETE FROM blocks WHERE block_uid = ?', (block['block_uid'],))
            
            self.conn.execute('''
                INSERT INTO blocks (
                    block_uid, symbol_id, block_type, line_start, line_end, lines_of_code,
                    nesting_level, ast_fingerprint, token_fingerprint, normalized_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                block['block_uid'], symbol_id, block['kind'], block['line_start'],
                block['line_end'], block['lines_of_code'], block['nesting_level'],
                block['ast_fingerprint'], block['token_fingerprint'], block['normalized_fingerprint']
            ))
    
    def _store_dependencies(self, symbol_ids: Dict[str, int], dependencies: List[Dict[str, Any]]) -> None:
        """Store dependencies."""
        # Dependencies are already deduplicated in the visitor
        for dep in dependencies:
            source_symbol_id = symbol_ids.get(dep['source_symbol_uid'])
            if not source_symbol_id:
                continue
            
            self.conn.execute('''
                INSERT INTO dependencies (
                    source_symbol_id, target_external, dependency_kind, resolution, confidence, evidence_json
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                source_symbol_id, dep['target_external'], dep['kind'],
                dep['resolution'], dep['confidence'], json.dumps(dep['evidence'])
            ))
    
    def _store_attribute_accesses(self, symbol_ids: Dict[str, int], accesses: List[Dict[str, Any]]) -> None:
        """Store attribute access patterns."""
        for access in accesses:
            symbol_id = symbol_ids.get(access['symbol_uid'])
            if not symbol_id:
                continue
            
            self.conn.execute('''
                INSERT INTO attribute_access (
                    symbol_id, attribute_name, access_type, line_number, confidence, evidence_json, count
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol_id, access['attribute_name'], access['access_type'],
                access['line_number'], access['confidence'], json.dumps(access['evidence']),
                access.get('count', 1)
            ))
    
    def _clear_index(self):
        """Clear all data from the index."""
        self.logger.info("Clearing existing index data")
        
        # Delete in reverse dependency order to avoid foreign key constraints
        tables_to_clear = [
            'attribute_access',
            'dependencies', 
            'blocks',
            'symbols',
            'files',
            'duplicate_members',
            'duplicate_groups',
            'refactoring_decisions',
            'problems',
            'analysis_runs'
        ]
        
        for table in tables_to_clear:
            try:
                self.conn.execute(f'DELETE FROM {table}')
            except sqlite3.OperationalError as e:
                # Table might not exist, which is fine
                self.logger.debug(f"Could not clear table {table}: {e}")
        
        self.conn.commit()
        self.logger.info("Index data cleared")

    def get_index_status(self) -> Dict[str, Any]:
        """Get current index status and statistics."""
        cursor = self.conn.execute('SELECT COUNT(*) FROM files')
        files_count = cursor.fetchone()[0]
        
        cursor = self.conn.execute('SELECT COUNT(*) FROM symbols')
        symbols_count = cursor.fetchone()[0]
        
        cursor = self.conn.execute('SELECT COUNT(*) FROM blocks')
        blocks_count = cursor.fetchone()[0]
        
        cursor = self.conn.execute('SELECT COUNT(*) FROM dependencies')
        dependencies_count = cursor.fetchone()[0]
        
        cursor = self.conn.execute('SELECT MAX(last_analyzed) FROM files')
        last_analysis = cursor.fetchone()[0]
        
        return {
            'files_indexed': files_count,
            'symbols_indexed': symbols_count,
            'blocks_indexed': blocks_count,
            'dependencies_indexed': dependencies_count,
            'last_analysis': last_analysis,
            'schema_version': IndexSchema.get_schema_version(self.conn),
            'database_size_mb': self._get_database_size_mb()
        }
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the index."""
        stats = self.get_index_status()
        
        # Add breakdown by file type
        cursor = self.conn.execute('''
            SELECT is_test_file, COUNT(*) as count, SUM(lines_of_code) as total_loc
            FROM files 
            GROUP BY is_test_file
        ''')
        file_breakdown = {}
        for row in cursor.fetchall():
            file_type = 'test_files' if row[0] else 'source_files'
            file_breakdown[file_type] = {
                'count': row[1],
                'total_lines_of_code': row[2] or 0
            }
        
        # Add breakdown by symbol type
        cursor = self.conn.execute('''
            SELECT kind, COUNT(*) as count, AVG(complexity_score) as avg_complexity
            FROM symbols 
            GROUP BY kind
        ''')
        symbol_breakdown = {}
        for row in cursor.fetchall():
            symbol_breakdown[row[0]] = {
                'count': row[1],
                'avg_complexity': round(row[2] or 0, 2)
            }
        
        # Add breakdown by semantic category
        cursor = self.conn.execute('''
            SELECT semantic_category, COUNT(*) as count
            FROM symbols 
            WHERE semantic_category IS NOT NULL
            GROUP BY semantic_category
        ''')
        category_breakdown = {}
        for row in cursor.fetchall():
            category_breakdown[row[0]] = row[1]
        
        stats.update({
            'file_breakdown': file_breakdown,
            'symbol_breakdown': symbol_breakdown,
            'category_breakdown': category_breakdown
        })
        
        return stats

    def _get_database_size_mb(self) -> float:
        """Get database file size in MB."""
        try:
            if self.db_path.exists():
                size_bytes = self.db_path.stat().st_size
                return round(size_bytes / (1024 * 1024), 2)
            return 0.0
        except Exception:
            return 0.0
    
    def close(self):
        """Close the database connection."""
        if hasattr(self, 'conn') and self.conn:
            try:
                self.conn.close()
                self.conn = None
            except Exception as e:
                self.logger.warning(f"Error closing database connection: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        """Ensure connection is closed when object is destroyed."""
        self.close()