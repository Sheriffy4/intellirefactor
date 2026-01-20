"""
Fingerprint Generator

Generates various types of fingerprints for functional blocks to enable
similarity detection and duplicate identification.
"""

from __future__ import annotations

import ast
import hashlib
import logging
from collections import Counter
from typing import List, Set

from .models import FunctionalBlock
from .normalization import normalize_for_hash  # backward compat
from .utils import is_likely_regex

logger = logging.getLogger(__name__)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class FingerprintGenerator:
    """
    Generates multiple types of fingerprints for functional blocks.
    """

    def __init__(self):
        self.logger = logger

    def generate_all_fingerprints(self, block: FunctionalBlock, source_code: str) -> FunctionalBlock:
        """Generate all fingerprints and store them in the block."""
        try:
            block.ast_hash = self.generate_ast_hash(source_code)
            block.token_fingerprint = self.generate_token_fingerprint(source_code)
            block.semantic_fingerprint = self.generate_semantic_fingerprint(block)
            return block
        except Exception as e:
            self.logger.warning(f"Failed to generate fingerprints for {block.id}: {e}")
            return block

    def generate_ast_hash(self, source_code: str) -> str:
        """Generate normalized AST hash from source."""
        try:
            tree = ast.parse(source_code)
            normalized = normalize_for_hash(tree)
            ast_str = ast.dump(normalized, annotate_fields=False, include_attributes=False)
            return _sha256_hex(ast_str)
        except Exception as e:
            self.logger.debug(f"Failed to generate AST hash: {e}")
            return ""

    def generate_ast_hash_from_node(self, node: ast.AST) -> str:
        try:
            # IMPORTANT: never mutate original node
            node_copy = copy.deepcopy(node)

            if not isinstance(node_copy, ast.Module):
                wrapped_node = ast.Module(body=[node_copy], type_ignores=[])
            else:
                wrapped_node = node_copy

            normalized_tree = normalize_for_hash(wrapped_node)
            ast_str = ast.dump(normalized_tree, annotate_fields=False, include_attributes=False)
            return hashlib.sha256(ast_str.encode("utf-8")).hexdigest()
        except Exception as e:
            self.logger.debug(f"Failed to generate AST hash from node: {e}")
            return ""

    def generate_token_fingerprint(self, source_code: str) -> str:
        """
        Generate token-based fingerprint using multiset of tokens.

        Fixes:
        - stable hashing via sha256
        - keyword detection via keyword.iskeyword (more correct than hardcoded list)
        """
        try:
            import keyword
            import tokenize
            from io import StringIO

            tokens: List[str] = []
            readline = StringIO(source_code).readline

            for tok in tokenize.generate_tokens(readline):
                if tok.type in (
                    tokenize.COMMENT,
                    tokenize.NL,
                    tokenize.NEWLINE,
                    tokenize.INDENT,
                    tokenize.DEDENT,
                ):
                    continue
                if tok.type == tokenize.ENDMARKER:
                    continue

                if tok.type == tokenize.NAME:
                    tokens.append(tok.string if keyword.iskeyword(tok.string) else "IDENTIFIER")
                elif tok.type == tokenize.STRING:
                    tokens.append("STRING")
                elif tok.type == tokenize.NUMBER:
                    tokens.append("NUMBER")
                else:
                    # operators / punctuation
                    tokens.append(tok.string)

            token_counts = Counter(tokens)
            token_str = "".join(f"{t}:{c};" for t, c in sorted(token_counts.items()))
            return _sha256_hex(token_str)

        except Exception as e:
            self.logger.debug(f"Failed to generate token fingerprint: {e}")
            return ""

    def generate_semantic_fingerprint(self, block: FunctionalBlock) -> str:
        """Generate semantic fingerprint based on block characteristics."""
        features: List[str] = []

        features.append(f"cat:{block.category}")
        features.append(f"sub:{block.subcategory}")

        features.append(f"loc:{(block.loc // 5) * 5}")
        features.append(f"cyc:{(block.cyclomatic // 3) * 3}")

        features.append(f"calls:{(len(block.calls) // 2) * 2}")
        features.append(f"called_by:{(len(block.called_by) // 2) * 2}")
        features.append(f"imports:{len(block.imports_used)}")

        for cat in sorted(self._categorize_imports(block.imports_used)):
            features.append(f"imp_cat:{cat}")

        for pat in sorted(self._analyze_literal_patterns(block.literals)):
            features.append(f"lit:{pat}")

        for tag in sorted(getattr(block, "tags", []) or []):
            features.append(f"tag:{tag}")

        features.extend(self._analyze_signature(block.signature))

        return _sha256_hex("|".join(features))

    def _categorize_imports(self, imports: List[str]) -> Set[str]:
        """
        Categorize imports into functional groups.

        Fix: do NOT use substring 're' / 'log' checks (requests/catalog/etc).
        Use base module and safer signals.
        """
        categories: Set[str] = set()

        for imp in imports:
            imp_lower = (imp or "").lower()
            base = imp_lower.split(".", 1)[0]

            if base in {"logging", "loguru"}:
                categories.add("logging")
            elif any(x in imp_lower for x in ["json", "yaml", "xml", "csv", "pickle", "msgpack"]):
                categories.add("serialization")
            elif any(x in imp_lower for x in ["http", "request", "urllib", "socket"]):
                categories.add("networking")
            elif any(x in imp_lower for x in ["sqlalchemy", "sqlite", "psycopg", "mysql", "postgres", "db", "database"]):
                categories.add("database")
            elif any(x in imp_lower for x in ["thread", "asyncio", "concurrent", "multiprocessing"]):
                categories.add("concurrency")
            elif any(x in imp_lower for x in ["test", "mock", "pytest", "unittest"]):
                categories.add("testing")
            elif any(x in imp_lower for x in ["cli", "argparse", "click", "typer"]):
                categories.add("cli")
            elif base == "re" or "regex" in imp_lower:
                categories.add("regex")
            else:
                categories.add("other")

        return categories

    def _analyze_literal_patterns(self, literals: List[str]) -> Set[str]:
        """Analyze patterns in string literals."""
        patterns: Set[str] = set()

        for literal in literals:
            if not literal:
                continue

            if is_likely_regex(literal):
                patterns.add("regex")
            elif literal.startswith(("http://", "https://", "ftp://")):
                patterns.add("url")
            elif "/" in literal and len(literal.split("/")) > 2:
                patterns.add("path")
            elif "@" in literal and "." in literal:
                patterns.add("email")
            elif literal.replace(".", "").replace("-", "").isdigit():
                patterns.add("version")
            elif literal.isupper() and len(literal) > 2:
                patterns.add("constant")
            elif literal.lower() in ("true", "false", "yes", "no", "on", "off"):
                patterns.add("boolean")
            elif len(literal) > 20:
                patterns.add("long_string")
            else:
                patterns.add("short_string")

        return patterns

    def _analyze_signature(self, signature: str) -> List[str]:
        """Analyze function signature characteristics (safe param counting)."""
        features: List[str] = []
        if not signature:
            return features

        # robust param count from text in parentheses
        param_count = 0
        if "(" in signature and ")" in signature:
            inside = signature.split("(", 1)[1].rsplit(")", 1)[0].strip()
            if inside:
                param_count = inside.count(",") + 1
            else:
                param_count = 0

        features.append(f"params:{(param_count // 2) * 2}")

        if "->" in signature:
            features.append("has_return_type")
        if ":" in signature and "(" in signature:
            features.append("has_type_hints")
        if "=" in signature and "(" in signature:
            features.append("has_defaults")
        if "*" in signature:
            features.append("has_varargs")

        return features