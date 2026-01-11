"""
Similarity Calculator

Implements the weighted similarity scoring from ref.md for functional blocks.
Calculates similarity based on AST shape, tokens, signatures, dependencies, literals, and names.
"""

from __future__ import annotations

import difflib
import logging
from typing import Any, Dict, List, Set, Tuple, Optional

from .models import FunctionalBlock, DecompositionConfig

logger = logging.getLogger(__name__)


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    """Stable undirected key for pairwise matrices / caches."""
    return (a, b) if a <= b else (b, a)


class SimilarityCalculator:
    """
    Calculates similarity scores between functional blocks using multiple channels.

    Weighted similarity:
    score = 0.30 * ast_shape + 0.20 * token + 0.15 * signature +
            0.15 * dependency + 0.10 * literals + 0.10 * name
    """

    def __init__(self, config: Optional[DecompositionConfig] = None):
        self.config = config or DecompositionConfig.default()
        self.weights = dict(self.config.similarity_weights or {})
        self.logger = logger

        # Pairwise caches (expensive string similarity)
        self._lit_ratio_cache: Dict[Tuple[str, str], float] = {}
        self._name_ratio_cache: Dict[Tuple[str, str], float] = {}

        # Per-block caches
        self._signature_cache: Dict[str, Dict[str, Any]] = {}
        self._name_tokens_cache: Dict[str, List[str]] = {}
        self._deps_set_cache: Dict[str, Set[str]] = {}

        self._calls_set_cache: Dict[str, Set[str]] = {}
        self._imports_set_cache: Dict[str, Set[str]] = {}
        self._globals_set_cache: Dict[str, Set[str]] = {}
        self._lits_set_cache: Dict[str, Set[str]] = {}

    # ------------------------
    # Public API
    # ------------------------

    def calculate_similarity(self, block1: FunctionalBlock, block2: FunctionalBlock) -> float:
        """Calculate overall similarity score between two functional blocks (0..1)."""
        if block1.id == block2.id:
            return 1.0

        components: Dict[str, float] = {}
        informative_weights: Dict[str, float] = {}

        # AST
        if self._is_informative_ast(block1, block2):
            ast_sim = self._ast_shape_similarity(block1, block2)
            components["ast_shape"] = ast_sim
            informative_weights["ast_shape"] = self.weights.get("ast_shape", 0.30)

        # Token
        if self._is_informative_token(block1, block2):
            token_sim = self._token_similarity(block1, block2)
            components["token"] = token_sim
            informative_weights["token"] = self.weights.get("token", 0.20)

        # Signature
        if self._is_informative_signature(block1, block2):
            signature_sim = self._signature_similarity(block1, block2)
            components["signature"] = signature_sim
            informative_weights["signature"] = self.weights.get("signature", 0.15)

        # Dependency
        if self._is_informative_dependency(block1, block2):
            dependency_sim = self._dependency_similarity(block1, block2)
            components["dependency"] = dependency_sim
            informative_weights["dependency"] = self.weights.get("dependency", 0.15)

        # Literals
        if self._is_informative_literals(block1, block2):
            literals_sim = self._literals_similarity(block1, block2)
            components["literals"] = literals_sim
            informative_weights["literals"] = self.weights.get("literals", 0.10)

        # Name (always informative, but capped if it's the only signal)
        name_sim = self._name_similarity(block1, block2)
        components["name"] = name_sim
        informative_weights["name"] = self.weights.get("name", 0.10)

        total_weight = sum(informative_weights.values())
        if total_weight <= 0:
            return 0.0

        # Avoid name-only clustering
        if set(informative_weights.keys()) == {"name"}:
            return min(0.2, components["name"])

        total_score = 0.0
        for key, w in informative_weights.items():
            total_score += components[key] * (w / total_weight)

        return min(1.0, max(0.0, total_score))

    def calculate_similarity_matrix(self, blocks: List[FunctionalBlock]) -> Dict[Tuple[str, str], float]:
        """
        Calculate similarity matrix.

        Return format: {(min_id, max_id): similarity}, one entry per pair.
        """
        similarity_matrix: Dict[Tuple[str, str], float] = {}
        if not blocks:
            return similarity_matrix

        n = len(blocks)

        # Max quality on medium groups: do full up to 200 by default
        candidate_threshold = getattr(self.config, "candidate_generation_threshold", 200)

        if n > candidate_threshold:
            self.logger.info(f"Using candidate generation for {n} blocks")
            candidate_pairs = self._generate_candidate_pairs(blocks)
            self.logger.info(
                f"Generated {len(candidate_pairs)} candidate pairs (vs {n * (n - 1) // 2} full)"
            )

            block_by_id = {b.id: b for b in blocks}
            for id1, id2 in candidate_pairs:
                b1 = block_by_id.get(id1)
                b2 = block_by_id.get(id2)
                if b1 is None or b2 is None or id1 == id2:
                    continue
                similarity_matrix[_pair_key(id1, id2)] = self.calculate_similarity(b1, b2)
        else:
            # Full O(n^2)
            for i in range(n):
                b1 = blocks[i]
                for j in range(i + 1, n):
                    b2 = blocks[j]
                    similarity_matrix[_pair_key(b1.id, b2.id)] = self.calculate_similarity(b1, b2)

        self.logger.debug(f"Calculated similarity matrix for {n} blocks")
        return similarity_matrix

    def find_similar_blocks(
        self,
        blocks: List[FunctionalBlock],
        threshold: float = 0.7,
    ) -> List[Tuple[FunctionalBlock, FunctionalBlock, float]]:
        """Find pairs of blocks with similarity above threshold."""
        similar_pairs: List[Tuple[FunctionalBlock, FunctionalBlock, float]] = []
        block_dict = {b.id: b for b in blocks}

        similarity_matrix = self.calculate_similarity_matrix(blocks)
        for (id1, id2), sim in similarity_matrix.items():
            if sim >= threshold:
                b1 = block_dict.get(id1)
                b2 = block_dict.get(id2)
                if b1 is not None and b2 is not None:
                    similar_pairs.append((b1, b2, sim))

        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        self.logger.info(f"Found {len(similar_pairs)} similar pairs above threshold {threshold}")
        return similar_pairs

    # ------------------------
    # Component similarities
    # ------------------------

    def _ast_shape_similarity(self, block1: FunctionalBlock, block2: FunctionalBlock) -> float:
        if not block1.ast_hash or not block2.ast_hash:
            return 0.0
        return 1.0 if block1.ast_hash == block2.ast_hash else 0.0

    def _token_similarity(self, block1: FunctionalBlock, block2: FunctionalBlock) -> float:
        # Fast-path: identical fingerprints
        if (
            block1.token_fingerprint
            and block2.token_fingerprint
            and block1.token_fingerprint == block2.token_fingerprint
        ):
            return 1.0

        calls1, calls2 = self._get_calls_set(block1), self._get_calls_set(block2)
        imps1, imps2 = self._get_imports_set(block1), self._get_imports_set(block2)

        if not (calls1 or imps1 or calls2 or imps2):
            return 0.0

        calls_sim = self._jaccard_no_signal(calls1, calls2)
        imports_sim = self._jaccard_no_signal(imps1, imps2)
        return (calls_sim + imports_sim) / 2.0

    def _signature_similarity(self, block1: FunctionalBlock, block2: FunctionalBlock) -> float:
        # no signal from both -> 0.0
        if (
            not block1.inputs and not block1.outputs and not block1.signature
            and not block2.inputs and not block2.outputs and not block2.signature
        ):
            return 0.0

        inputs_sim = self._jaccard_similarity(set(block1.inputs), set(block2.inputs))
        outputs_sim = self._jaccard_similarity(set(block1.outputs), set(block2.outputs))

        if block1.signature and block2.signature:
            sig1 = self._get_cached_signature(block1)
            sig2 = self._get_cached_signature(block2)

            arity_sim = self._numeric_similarity(sig1["param_count"], sig2["param_count"], max_val=10)
            params_sim = self._jaccard_similarity(set(sig1["params"]), set(sig2["params"]))
            return_sim = 1.0 if sig1["return_type"] == sig2["return_type"] else 0.0

            sig_sim = (arity_sim + params_sim + return_sim) / 3.0
            return (inputs_sim + outputs_sim + sig_sim) / 3.0

        return (inputs_sim + outputs_sim) / 2.0

    def _dependency_similarity(self, block1: FunctionalBlock, block2: FunctionalBlock) -> float:
        deps1 = self._get_cached_deps_set(block1)
        deps2 = self._get_cached_deps_set(block2)
        return self._jaccard_no_signal(deps1, deps2)

    def _literals_similarity(self, block1: FunctionalBlock, block2: FunctionalBlock) -> float:
        lits1 = self._get_lits_set(block1)
        lits2 = self._get_lits_set(block2)

        if not lits1 and not lits2:
            return 0.0

        exact_sim = self._jaccard_similarity(lits1, lits2)

        # PERF: if exact overlap already gives enough signal, skip fuzzy entirely
        if exact_sim >= 0.30:
            return exact_sim

        fuzzy_sim = self._fuzzy_literals_similarity(list(lits1), list(lits2))
        return max(exact_sim, fuzzy_sim)

    def _name_similarity(self, block1: FunctionalBlock, block2: FunctionalBlock) -> float:
        name1 = (block1.method_name or "").lower()
        name2 = (block2.method_name or "").lower()

        if name1 == name2 and name1:
            return 1.0

        tokens1 = self._get_cached_name_tokens(block1)
        tokens2 = self._get_cached_name_tokens(block2)
        token_sim = self._jaccard_similarity(set(tokens1), set(tokens2))

        # cache expensive SequenceMatcher for repeated name pairs
        key = _pair_key(name1, name2)
        if key in self._name_ratio_cache:
            string_sim = self._name_ratio_cache[key]
        else:
            string_sim = difflib.SequenceMatcher(None, name1, name2).ratio()
            self._name_ratio_cache[key] = string_sim

        return (token_sim + string_sim) / 2.0

    # ------------------------
    # Parsing / tokenization / helpers
    # ------------------------

    def _parse_signature(self, signature: str) -> Dict[str, Any]:
        try:
            if "(" in signature and ")" in signature:
                func_name = signature.split("(", 1)[0].strip()
                params_part = signature.split("(", 1)[1].split(")", 1)[0]

                params: List[str] = []
                if params_part.strip():
                    params = [p.strip() for p in params_part.split(",") if p.strip()]

                return_type = ""
                if "->" in signature:
                    return_type = signature.split("->", 1)[-1].strip()

                return {
                    "func_name": func_name,
                    "params": params,
                    "param_count": len(params),
                    "return_type": return_type,
                }
        except Exception:
            pass

        return {"func_name": signature, "params": [], "param_count": 0, "return_type": ""}

    def _tokenize_name(self, name: str) -> List[str]:
        import re
        tokens = name.split("_")
        result: List[str] = []
        for token in tokens:
            camel_tokens = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|\d+", token)
            if camel_tokens:
                result.extend([t.lower() for t in camel_tokens if t])
            else:
                if token:
                    result.append(token.lower())
        return [t for t in result if t]

    def _jaccard_similarity(self, set1: Set[Any], set2: Set[Any]) -> float:
        if not set1 and not set2:
            return 1.0
        union = len(set1 | set2)
        return (len(set1 & set2) / union) if union > 0 else 0.0

    def _jaccard_no_signal(self, set1: Set[Any], set2: Set[Any]) -> float:
        if not set1 and not set2:
            return 0.0
        union = len(set1 | set2)
        return (len(set1 & set2) / union) if union > 0 else 0.0

    def _numeric_similarity(self, val1: float, val2: float, max_val: float = 100) -> float:
        if val1 == val2:
            return 1.0
        if max_val <= 0:
            return 0.0
        diff = abs(val1 - val2)
        return max(0.0, 1.0 - (diff / max_val))

    def _fuzzy_literals_similarity(self, literals1: List[str], literals2: List[str]) -> float:
        """
        Fuzzy similarity between literal lists.

        PERF/QUALITY balance:
        - keep only informative literals
        - limit to max 5 per side (25 comparisons per lit1 => up to 25*5=125)
        - cache string ratios
        - truncate very long literals (SequenceMatcher can be expensive)
        """
        if not literals1 or not literals2:
            return 0.0

        def is_informative(lit: str) -> bool:
            if not lit:
                return False
            if len(lit) >= 5:
                return True
            return any(c in lit for c in ["[", "]", "(", ")", "*", "+", "?", "^", "$", "|"])

        informative1 = [lit for lit in literals1 if is_informative(lit)][:5]
        informative2 = [lit for lit in literals2 if is_informative(lit)][:5]
        if not informative1 or not informative2:
            return 0.0

        bests: List[float] = []
        for lit1 in informative1:
            lit1s = lit1[:200]
            best = 0.0

            for lit2 in informative2:
                lit2s = lit2[:200]
                k = _pair_key(lit1s, lit2s)
                sim = self._lit_ratio_cache.get(k)
                if sim is None:
                    sim = difflib.SequenceMatcher(None, lit1s, lit2s).ratio()
                    self._lit_ratio_cache[k] = sim

                if sim > best:
                    best = sim
                    if best >= 1.0:
                        break  # cannot do better

            bests.append(best)

        return (sum(bests) / len(bests)) if bests else 0.0

    # ------------------------
    # Candidate generation (LSH-lite)
    # ------------------------

    def _generate_candidate_pairs(self, blocks: List[FunctionalBlock]) -> List[Tuple[str, str]]:
        candidate_pairs: Set[Tuple[str, str]] = set()

        ast_hash_index: Dict[str, List[str]] = {}
        token_fp_index: Dict[str, List[str]] = {}
        name_token_index: Dict[str, List[str]] = {}
        raw_call_index: Dict[str, List[str]] = {}
        import_index: Dict[str, List[str]] = {}

        for block in blocks:
            if block.ast_hash:
                ast_hash_index.setdefault(block.ast_hash, []).append(block.id)

            if block.token_fingerprint:
                token_fp_index.setdefault(block.token_fingerprint, []).append(block.id)

            for token in self._get_cached_name_tokens(block):
                if len(token) >= 3 and token not in {"get", "set", "do", "run", "call", "make"}:
                    name_token_index.setdefault(token, []).append(block.id)

            for raw_call in block.raw_calls:
                normalized_call = raw_call.split(".")[-1] if "." in raw_call else raw_call
                normalized_call = (normalized_call or "").strip()
                if len(normalized_call) >= 3 and normalized_call not in {"get", "set", "call", "run"}:
                    raw_call_index.setdefault(normalized_call, []).append(block.id)

            for import_name in block.imports_used:
                base_import = (import_name.split(".", 1)[0] or "").strip()
                if len(base_import) >= 3:
                    import_index.setdefault(base_import, []).append(block.id)

        def add_bucket_pairs(bucket: List[str], max_bucket: int) -> None:
            # guard against explosions
            if not (2 <= len(bucket) <= max_bucket):
                return
            for i in range(len(bucket)):
                a = bucket[i]
                for j in range(i + 1, len(bucket)):
                    candidate_pairs.add(_pair_key(a, bucket[j]))

        # Exact duplicates
        for bucket in ast_hash_index.values():
            add_bucket_pairs(bucket, max_bucket=10_000)

        # Token fingerprint buckets (limit)
        for bucket in token_fp_index.values():
            add_bucket_pairs(bucket, max_bucket=50)

        # Rare-ish signals
        for bucket in name_token_index.values():
            add_bucket_pairs(bucket, max_bucket=10)

        for bucket in raw_call_index.values():
            add_bucket_pairs(bucket, max_bucket=20)

        for bucket in import_index.values():
            add_bucket_pairs(bucket, max_bucket=20)

        return list(candidate_pairs)

    # ------------------------
    # Cached accessors
    # ------------------------

    def _get_cached_signature(self, block: FunctionalBlock) -> Dict[str, Any]:
        if block.id not in self._signature_cache:
            self._signature_cache[block.id] = self._parse_signature(block.signature or "")
        return self._signature_cache[block.id]

    def _get_cached_name_tokens(self, block: FunctionalBlock) -> List[str]:
        if block.id not in self._name_tokens_cache:
            self._name_tokens_cache[block.id] = self._tokenize_name((block.method_name or "").lower())
        return self._name_tokens_cache[block.id]

    def _get_cached_deps_set(self, block: FunctionalBlock) -> Set[str]:
        if block.id not in self._deps_set_cache:
            deps = self._get_imports_set(block) | self._get_calls_set(block) | self._get_globals_set(block)
            self._deps_set_cache[block.id] = deps
        return self._deps_set_cache[block.id]

    def _get_calls_set(self, block: FunctionalBlock) -> Set[str]:
        if block.id not in self._calls_set_cache:
            self._calls_set_cache[block.id] = set(block.raw_calls or [])
        return self._calls_set_cache[block.id]

    def _get_imports_set(self, block: FunctionalBlock) -> Set[str]:
        if block.id not in self._imports_set_cache:
            self._imports_set_cache[block.id] = set(block.imports_used or [])
        return self._imports_set_cache[block.id]

    def _get_globals_set(self, block: FunctionalBlock) -> Set[str]:
        if block.id not in self._globals_set_cache:
            self._globals_set_cache[block.id] = set(block.globals_used or [])
        return self._globals_set_cache[block.id]

    def _get_lits_set(self, block: FunctionalBlock) -> Set[str]:
        if block.id not in self._lits_set_cache:
            self._lits_set_cache[block.id] = set(block.literals or [])
        return self._lits_set_cache[block.id]

    # ------------------------
    # Informative checks
    # ------------------------

    def _is_informative_ast(self, block1: FunctionalBlock, block2: FunctionalBlock) -> bool:
        return bool(block1.ast_hash and block2.ast_hash)

    def _is_informative_token(self, block1: FunctionalBlock, block2: FunctionalBlock) -> bool:
        return bool(
            (block1.raw_calls or block1.imports_used or block1.token_fingerprint)
            and (block2.raw_calls or block2.imports_used or block2.token_fingerprint)
        )

    def _is_informative_signature(self, block1: FunctionalBlock, block2: FunctionalBlock) -> bool:
        return bool(
            (block1.inputs or block1.outputs or block1.signature)
            and (block2.inputs or block2.outputs or block2.signature)
        )

    def _is_informative_dependency(self, block1: FunctionalBlock, block2: FunctionalBlock) -> bool:
        return bool(
            (block1.imports_used or block1.raw_calls or block1.globals_used)
            and (block2.imports_used or block2.raw_calls or block2.globals_used)
        )

    def _is_informative_literals(self, block1: FunctionalBlock, block2: FunctionalBlock) -> bool:
        return bool(block1.literals or block2.literals)