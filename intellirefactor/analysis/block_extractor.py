"""
BlockExtractor for IntelliRefactor multi-channel clone detection.

This module implements the BlockExtractor class that extracts code blocks from Python
source code and generates multiple types of fingerprints for clone detection:
- Channel A (Exact): Token winnowing / k-gram fingerprints
- Channel B (Structural): AST fingerprints
- Channel C (Normalized): Variable/literal normalized AST

The extractor identifies different block types (if/for/while/try/statement_group)
and generates comprehensive fingerprints for each channel.
"""

import ast
import hashlib
import re
from typing import List, Set
from dataclasses import dataclass
from enum import Enum
import logging

from .models import BlockInfo, BlockType, FileReference


class FingerprintChannel(Enum):
    """Types of fingerprint channels for clone detection."""

    EXACT = "exact"  # Token winnowing / k-gram fingerprints
    STRUCTURAL = "structural"  # AST fingerprints
    NORMALIZED = "normalized"  # Variable/literal normalized AST


@dataclass
class BlockFingerprints:
    """Container for all fingerprints of a code block."""

    exact_fingerprint: str  # Channel A: Token-based fingerprint
    structural_fingerprint: str  # Channel B: AST structure fingerprint
    normalized_fingerprint: str  # Channel C: Normalized AST fingerprint

    def get_fingerprint(self, channel: FingerprintChannel) -> str:
        """Get fingerprint for specific channel."""
        if channel == FingerprintChannel.EXACT:
            return self.exact_fingerprint
        elif channel == FingerprintChannel.STRUCTURAL:
            return self.structural_fingerprint
        elif channel == FingerprintChannel.NORMALIZED:
            return self.normalized_fingerprint
        else:
            raise ValueError(f"Unknown fingerprint channel: {channel}")


class BlockExtractor:
    """Extracts code blocks and generates multi-channel fingerprints for clone detection."""

    def __init__(self, k_gram_size: int = 3, winnowing_window: int = 4):
        """Initialize the BlockExtractor.

        Args:
            k_gram_size: Size of k-grams for token fingerprinting
            winnowing_window: Window size for winnowing algorithm
        """
        self.k_gram_size = k_gram_size
        self.winnowing_window = winnowing_window
        self.logger = logging.getLogger(__name__)

        # Patterns for normalization
        self.variable_pattern = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
        self.string_pattern = re.compile(r'["\'].*?["\']')
        self.number_pattern = re.compile(r"\b\d+\.?\d*\b")

    def extract_blocks(self, source_code: str, file_path: str) -> List[BlockInfo]:
        """Extract all code blocks from source code with multi-channel fingerprints.

        Args:
            source_code: Python source code to analyze
            file_path: Path to the source file

        Returns:
            List of BlockInfo objects with fingerprints
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in {file_path}: {e}")
            return []

        lines = source_code.split("\n")

        # Extract blocks using AST visitor
        visitor = BlockVisitor(self, lines, file_path)
        visitor.visit(tree)

        return visitor.blocks

    def generate_fingerprints(
        self, node: ast.AST, source_lines: List[str], start_line: int, end_line: int
    ) -> BlockFingerprints:
        """Generate all three types of fingerprints for a code block.

        Args:
            node: AST node representing the block
            source_lines: Source code lines
            start_line: Starting line number (1-based)
            end_line: Ending line number (1-based)

        Returns:
            BlockFingerprints with all three channel fingerprints
        """
        # Extract source code for the block
        block_source = "\n".join(source_lines[start_line - 1 : end_line])

        # Channel A: Exact token fingerprint
        exact_fp = self._generate_exact_fingerprint(block_source)

        # Channel B: Structural AST fingerprint
        structural_fp = self._generate_structural_fingerprint(node)

        # Channel C: Normalized fingerprint
        normalized_fp = self._generate_normalized_fingerprint(node, block_source)

        return BlockFingerprints(
            exact_fingerprint=exact_fp,
            structural_fingerprint=structural_fp,
            normalized_fingerprint=normalized_fp,
        )

    def _fp16(self, data: bytes) -> str:
        """Stable 16-hex fingerprint (BLAKE2b digest_size=8)."""
        return hashlib.blake2b(data, digest_size=8).hexdigest()

    def _generate_exact_fingerprint(self, source_code: str) -> str:
        """Generate Channel A: Exact token fingerprint using winnowing.

        Uses k-gram winnowing algorithm for robust exact matching.
        """
        # Tokenize the source code
        tokens = self._tokenize_source(source_code)

        if len(tokens) < self.k_gram_size:
            # For very small blocks, use simple hash
            return self._fp16("".join(tokens).encode("utf-8"))

        # Generate k-grams
        k_grams = []
        for i in range(len(tokens) - self.k_gram_size + 1):
            k_gram = "".join(tokens[i : i + self.k_gram_size])
            k_grams.append(k_gram)

        # Apply winnowing algorithm
        fingerprints = self._winnowing(k_grams)

        # Combine fingerprints into single hash
        combined = "".join(sorted(fingerprints))
        return self._fp16(combined.encode("utf-8"))

    def _generate_structural_fingerprint(self, node: ast.AST) -> str:
        """Generate Channel B: Structural AST fingerprint.

        Captures the structural pattern of the AST without considering
        variable names or literal values.
        """
        structure = self._extract_ast_structure(node)
        return self._fp16(structure.encode("utf-8"))

    def _generate_normalized_fingerprint(self, node: ast.AST, source_code: str) -> str:
        """Generate Channel C: Normalized fingerprint.

        Normalizes variable names and literals to detect semantic clones.
        """
        # Normalize the source code
        normalized_source = self._normalize_source(source_code)

        # Generate fingerprint from normalized source
        return self._fp16(normalized_source.encode("utf-8"))

    def _tokenize_source(self, source_code: str) -> List[str]:
        """Tokenize source code for exact fingerprinting."""
        # Simple tokenization - split on whitespace and punctuation
        tokens = []
        buf = ""

        for char in source_code:
            if char.isalnum() or char == "_":
                buf += char
            else:
                if buf:
                    tokens.append(buf)
                    buf = ""
                if not char.isspace():
                    tokens.append(char)

        if buf:
            tokens.append(buf)

        return [t for t in tokens if t.strip()]

    def _winnowing(self, k_grams: List[str]) -> Set[str]:
        """Apply winnowing algorithm to k-grams."""
        if len(k_grams) <= self.winnowing_window:
            return set(k_grams)

        # Hash each k-gram
        hashes = [(self._fp16(kg.encode("utf-8")), kg) for kg in k_grams]

        fingerprints = set()

        # Sliding window winnowing
        for i in range(len(hashes) - self.winnowing_window + 1):
            window = hashes[i : i + self.winnowing_window]
            # Select minimum hash in window
            min_hash = min(window, key=lambda x: x[0])
            fingerprints.add(min_hash[1])

        return fingerprints

    def _extract_ast_structure(self, node: ast.AST) -> str:
        """Extract structural pattern from AST node."""
        if isinstance(node, ast.Module):
            return (
                "Module("
                + ",".join(self._extract_ast_structure(child) for child in node.body)
                + ")"
            )
        elif isinstance(node, ast.FunctionDef):
            return (
                f"FunctionDef({len(node.args.args)},"
                + ",".join(self._extract_ast_structure(child) for child in node.body)
                + ")"
            )
        elif isinstance(node, ast.ClassDef):
            return (
                "ClassDef("
                + ",".join(self._extract_ast_structure(child) for child in node.body)
                + ")"
            )
        elif isinstance(node, ast.If):
            return (
                "If("
                + ",".join(self._extract_ast_structure(child) for child in node.body + node.orelse)
                + ")"
            )
        elif isinstance(node, ast.For):
            return (
                "For("
                + ",".join(self._extract_ast_structure(child) for child in node.body + node.orelse)
                + ")"
            )
        elif isinstance(node, ast.While):
            return (
                "While("
                + ",".join(self._extract_ast_structure(child) for child in node.body + node.orelse)
                + ")"
            )
        elif isinstance(node, ast.Try):
            return (
                "Try("
                + ",".join(
                    self._extract_ast_structure(child)
                    for child in node.body + node.orelse + node.finalbody
                )
                + ")"
            )
        elif isinstance(node, ast.With):
            return (
                "With(" + ",".join(self._extract_ast_structure(child) for child in node.body) + ")"
            )
        elif isinstance(node, ast.Assign):
            return "Assign"
        elif isinstance(node, ast.AugAssign):
            return "AugAssign"
        elif isinstance(node, ast.Return):
            return "Return"
        elif isinstance(node, ast.Expr):
            # For expressions, also check the value
            if hasattr(node, "value"):
                return "Expr(" + self._extract_ast_structure(node.value) + ")"
            return "Expr"
        elif isinstance(node, ast.Call):
            return f"Call({len(node.args)})"
        elif isinstance(node, ast.BinOp):
            return f"BinOp({type(node.op).__name__})"
        elif isinstance(node, ast.Compare):
            return f"Compare({len(node.ops)})"
        else:
            return type(node).__name__

    def _normalize_source(self, source_code: str) -> str:
        """Normalize source code by replacing variables and literals."""
        normalized = source_code

        # Replace string literals first
        normalized = self.string_pattern.sub("STRING_LITERAL", normalized)

        # Replace numeric literals
        normalized = self.number_pattern.sub("NUMBER_LITERAL", normalized)

        # Replace variable names (more sophisticated approach needed)
        # For now, use simple pattern matching
        variable_map = {}
        var_counter = 0

        def replace_variable(match):
            nonlocal var_counter
            var_name = match.group(0)

            # Skip Python keywords and built-ins
            if var_name in {
                "if",
                "else",
                "elif",
                "for",
                "while",
                "try",
                "except",
                "finally",
                "def",
                "class",
                "return",
                "yield",
                "import",
                "from",
                "as",
                "True",
                "False",
                "None",
                "and",
                "or",
                "not",
                "in",
                "is",
                "print",
                "len",
                "str",
                "int",
                "float",
                "list",
                "dict",
                "set",
                "STRING_LITERAL",
                "NUMBER_LITERAL",
            }:  # Also skip our replacements
                return var_name

            if var_name not in variable_map:
                variable_map[var_name] = f"VAR_{var_counter}"
                var_counter += 1

            return variable_map[var_name]

        normalized = self.variable_pattern.sub(replace_variable, normalized)

        return normalized

    def _detect_block_type(self, node: ast.AST) -> BlockType:
        """Detect the type of code block from AST node."""
        if isinstance(node, ast.If):
            return BlockType.IF_BLOCK
        elif isinstance(node, ast.For):
            return BlockType.FOR_LOOP
        elif isinstance(node, ast.While):
            return BlockType.WHILE_LOOP
        elif isinstance(node, ast.Try):
            return BlockType.TRY_BLOCK
        elif isinstance(node, ast.With):
            return BlockType.WITH_BLOCK
        elif isinstance(node, ast.FunctionDef):
            return BlockType.FUNCTION_BODY
        elif isinstance(node, ast.ClassDef):
            return BlockType.CLASS_BODY
        else:
            return BlockType.STATEMENT_GROUP

    def _calculate_nesting_level(self, node: ast.AST, parent_level: int = 0) -> int:
        """Calculate nesting level of a block."""
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
            return parent_level + 1
        return parent_level

    def _count_statements(self, node: ast.AST) -> int:
        """Count the number of statements in a block."""
        if hasattr(node, "body"):
            return len(node.body)
        return 1

    def _calculate_lines_of_code(
        self, start_line: int, end_line: int, source_lines: List[str]
    ) -> int:
        """Calculate lines of code in a block, excluding empty lines and comments."""
        loc = 0
        for i in range(start_line - 1, min(end_line, len(source_lines))):
            line = source_lines[i].strip()
            if line and not line.startswith("#"):
                loc += 1
        return loc


class BlockVisitor(ast.NodeVisitor):
    """AST visitor for extracting code blocks."""

    def __init__(self, extractor: BlockExtractor, source_lines: List[str], file_path: str):
        self.extractor = extractor
        self.source_lines = source_lines
        self.file_path = file_path
        self.blocks = []
        self.nesting_level = 0

    def visit_If(self, node: ast.If) -> None:
        self._extract_block(node)
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1

    def visit_For(self, node: ast.For) -> None:
        self._extract_block(node)
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1

    def visit_While(self, node: ast.While) -> None:
        self._extract_block(node)
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1

    def visit_Try(self, node: ast.Try) -> None:
        self._extract_block(node)
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1

    def visit_With(self, node: ast.With) -> None:
        self._extract_block(node)
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._extract_block(node)
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._extract_block(node)
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1

    def _extract_block(self, node: ast.AST) -> None:
        """Extract a code block and generate its fingerprints."""
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return

        start_line = node.lineno
        end_line = node.end_lineno or start_line

        # Skip very small blocks (but be more lenient for testing)
        if end_line - start_line < 1:
            return

        # Generate fingerprints
        fingerprints = self.extractor.generate_fingerprints(
            node, self.source_lines, start_line, end_line
        )

        # Create file reference
        file_ref = FileReference(self.file_path, start_line, end_line)

        # Detect block type
        block_type = self.extractor._detect_block_type(node)

        # Calculate metrics
        lines_of_code = self.extractor._calculate_lines_of_code(
            start_line, end_line, self.source_lines
        )
        statement_count = self.extractor._count_statements(node)

        # Create BlockInfo
        block_info = BlockInfo(
            block_type=block_type,
            file_reference=file_ref,
            parent_method=None,  # Will be set by caller if needed
            ast_fingerprint=fingerprints.structural_fingerprint,
            token_fingerprint=fingerprints.exact_fingerprint,
            normalized_fingerprint=fingerprints.normalized_fingerprint,
            nesting_level=self.nesting_level,
            lines_of_code=lines_of_code,
            statement_count=statement_count,
            min_clone_size=max(3, lines_of_code // 3),  # Heuristic
            is_extractable=lines_of_code >= 2 and statement_count >= 1,  # More lenient
            confidence=0.9,  # High confidence for AST-based extraction
            metadata={
                "exact_fingerprint": fingerprints.exact_fingerprint,
                "structural_fingerprint": fingerprints.structural_fingerprint,
                "normalized_fingerprint": fingerprints.normalized_fingerprint,
                "extraction_method": "ast_visitor",
            },
        )

        self.blocks.append(block_info)
