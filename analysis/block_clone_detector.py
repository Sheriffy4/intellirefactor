"""
BlockCloneDetector for IntelliRefactor multi-channel clone detection.

This module implements the BlockCloneDetector class that identifies clone groups
using intersection and ranking of multiple fingerprint channels. It provides
extraction strategy recommendations with confidence scores and creates
line-level references for each clone instance.
"""

import hashlib
import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

from .models import BlockInfo, Evidence
from .block_extractor import FingerprintChannel


class CloneType(Enum):
    """Types of code clones based on similarity."""
    EXACT = "exact"           # Identical code (Channel A matches)
    STRUCTURAL = "structural" # Same structure, different names (Channel B matches)
    SEMANTIC = "semantic"     # Same logic, different implementation (Channel C matches)
    HYBRID = "hybrid"         # Matches multiple channels


class ExtractionStrategy(Enum):
    """Strategies for extracting cloned code."""
    EXTRACT_METHOD = "extract_method"
    EXTRACT_FUNCTION = "extract_function"
    EXTRACT_CLASS = "extract_class"
    PARAMETERIZE = "parameterize"
    TEMPLATE_METHOD = "template_method"
    NO_EXTRACTION = "no_extraction"


# Type alias for block identification key
BlockKey = Tuple[str, int, int]  # (file_path, line_start, line_end)


@dataclass
class CloneInstance:
    """Represents a single instance of a clone."""
    block_info: BlockInfo
    similarity_score: float
    evidence: Evidence
    extraction_feasibility: float  # 0.0 to 1.0
    
    @property
    def file_path(self) -> str:
        return self.block_info.file_reference.file_path
    
    @property
    def line_start(self) -> int:
        return self.block_info.file_reference.line_start
    
    @property
    def line_end(self) -> int:
        return self.block_info.file_reference.line_end
    
    @property
    def block_key(self) -> BlockKey:
        """Unique identifier for this clone instance."""
        return (self.file_path, self.line_start, self.line_end)


@dataclass
class CloneGroup:
    """Represents a group of similar code blocks."""
    group_id: str
    clone_type: CloneType
    instances: List[CloneInstance] = field(default_factory=list)
    similarity_score: float = 0.0
    detection_channels: Set[FingerprintChannel] = field(default_factory=set)
    extraction_strategy: Optional[ExtractionStrategy] = None
    extraction_confidence: float = 0.0
    evidence: Optional[Evidence] = None
    ranking_score: float = 0.0
    
    def __post_init__(self):
        if self.instances:
            self._calculate_group_metrics()
    
    def _calculate_group_metrics(self):
        """Calculate group-level metrics from instances."""
        if not self.instances:
            return
        
        # Average similarity score (clamped)
        total = sum(inst.similarity_score for inst in self.instances)
        self.similarity_score = self._clamp01(total / len(self.instances))
        
        # Determine extraction strategy and confidence
        self._determine_extraction_strategy()
    
    @staticmethod
    def _clamp01(value: float) -> float:
        """Clamp value to [0.0, 1.0] range."""
        return min(1.0, max(0.0, value))
    
    def _determine_extraction_strategy(self):
        """Determine the best extraction strategy for this clone group."""
        if len(self.instances) < 2:
            self.extraction_strategy = ExtractionStrategy.NO_EXTRACTION
            self.extraction_confidence = 0.0
            return
        
        # Analyze block characteristics
        n = len(self.instances)
        avg_loc = sum(inst.block_info.lines_of_code for inst in self.instances) / n
        avg_statements = sum(inst.block_info.statement_count for inst in self.instances) / n
        avg_feasibility = sum(inst.extraction_feasibility for inst in self.instances) / n
        
        # Determine strategy based on clone type and characteristics
        if self.clone_type == CloneType.EXACT:
            if avg_loc >= 10 and avg_statements >= 5:
                self.extraction_strategy = ExtractionStrategy.EXTRACT_METHOD
                self.extraction_confidence = min(0.9, avg_feasibility + 0.1)
            elif avg_loc >= 5:
                self.extraction_strategy = ExtractionStrategy.EXTRACT_FUNCTION
                self.extraction_confidence = min(0.8, avg_feasibility)
            else:
                self.extraction_strategy = ExtractionStrategy.NO_EXTRACTION
                self.extraction_confidence = 0.0
        
        elif self.clone_type == CloneType.STRUCTURAL:
            if avg_loc >= 15 and avg_statements >= 8:
                self.extraction_strategy = ExtractionStrategy.PARAMETERIZE
                self.extraction_confidence = min(0.8, avg_feasibility)
            elif avg_loc >= 8:
                self.extraction_strategy = ExtractionStrategy.EXTRACT_METHOD
                self.extraction_confidence = min(0.7, avg_feasibility)
            else:
                self.extraction_strategy = ExtractionStrategy.NO_EXTRACTION
                self.extraction_confidence = 0.0
        
        elif self.clone_type == CloneType.SEMANTIC:
            if avg_loc >= 20 and avg_statements >= 10:
                self.extraction_strategy = ExtractionStrategy.TEMPLATE_METHOD
                self.extraction_confidence = min(0.7, avg_feasibility)
            else:
                self.extraction_strategy = ExtractionStrategy.NO_EXTRACTION
                self.extraction_confidence = 0.0
        
        else:  # HYBRID
            if avg_loc >= 12 and avg_statements >= 6:
                self.extraction_strategy = ExtractionStrategy.EXTRACT_METHOD
                self.extraction_confidence = min(0.8, avg_feasibility)
            else:
                self.extraction_strategy = ExtractionStrategy.NO_EXTRACTION
                self.extraction_confidence = 0.0
        
        self.extraction_confidence = self._clamp01(self.extraction_confidence)
    
    def add_instance(self, instance: CloneInstance):
        """Add a clone instance to this group."""
        self.instances.append(instance)
        self._calculate_group_metrics()
    
    def get_representative_instance(self) -> Optional[CloneInstance]:
        """Get the most representative instance of this clone group."""
        if not self.instances:
            return None
        return max(
            self.instances,
            key=lambda inst: inst.similarity_score * inst.extraction_feasibility
        )


class BlockCloneDetector:
    """Detects code clones using multi-channel fingerprint analysis."""
    
    def __init__(self, 
                 exact_threshold: float = 0.95,
                 structural_threshold: float = 0.85,
                 semantic_threshold: float = 0.75,
                 min_clone_size: int = 3,
                 min_instances: int = 2):
        """Initialize the BlockCloneDetector."""
        self.exact_threshold = exact_threshold
        self.structural_threshold = structural_threshold
        self.semantic_threshold = semantic_threshold
        self.min_clone_size = min_clone_size
        self.min_instances = min_instances
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def _clamp01(value: float) -> float:
        """Clamp value to [0.0, 1.0] range."""
        return min(1.0, max(0.0, value))
    
    @staticmethod
    def _get_block_key(block: BlockInfo) -> BlockKey:
        """Get unique key for a block (file_path, line_start, line_end)."""
        ref = block.file_reference
        return (ref.file_path, ref.line_start, ref.line_end)
    
    def _mark_as_covered(self, blocks: List[BlockInfo], covered: Set[BlockKey]) -> None:
        """Mark blocks as covered."""
        for block in blocks:
            covered.add(self._get_block_key(block))
    
    def _filter_uncovered(self, blocks: List[BlockInfo], 
                         covered: Set[BlockKey]) -> List[BlockInfo]:
        """Filter out blocks that are already covered."""
        return [b for b in blocks if self._get_block_key(b) not in covered]
    
    def detect_clones(self, blocks: List[BlockInfo]) -> List[CloneGroup]:
        """Detect clone groups from a list of code blocks."""
        if len(blocks) < self.min_instances:
            return []
        
        # Filter blocks by minimum size
        filtered_blocks = [b for b in blocks if b.lines_of_code >= self.min_clone_size]
        
        if len(filtered_blocks) < self.min_instances:
            return []
        
        # Track covered blocks to avoid duplicates
        covered_blocks: Set[BlockKey] = set()
        
        # Group blocks by fingerprints for each channel
        exact_groups = self._group_by_fingerprint(filtered_blocks, 'token_fingerprint')
        structural_groups = self._group_by_fingerprint(filtered_blocks, 'ast_fingerprint')
        normalized_groups = self._group_by_fingerprint(filtered_blocks, 'normalized_fingerprint')
        
        clone_groups: List[CloneGroup] = []
        
        # 1. Process EXACT matches (Channel A) - highest priority
        for group_blocks in exact_groups.values():
            if len(group_blocks) >= self.min_instances:
                clone_group = self._create_clone_group(
                    group_blocks, CloneType.EXACT, {FingerprintChannel.EXACT}
                )
                if clone_group:
                    clone_groups.append(clone_group)
                    self._mark_as_covered(group_blocks, covered_blocks)
        
        # 2. Process HYBRID clones (blocks matching 2 of 3 channels)
        # Must run BEFORE structural/semantic to catch stronger signals
        hybrid_groups = self._detect_hybrid_clones(filtered_blocks, covered_blocks)
        clone_groups.extend(hybrid_groups)
        
        # 3. Process STRUCTURAL matches (Channel B)
        for group_blocks in structural_groups.values():
            if len(group_blocks) >= self.min_instances:
                uncovered = self._filter_uncovered(group_blocks, covered_blocks)
                if len(uncovered) >= self.min_instances:
                    clone_group = self._create_clone_group(
                        uncovered, CloneType.STRUCTURAL, {FingerprintChannel.STRUCTURAL}
                    )
                    if clone_group:
                        clone_groups.append(clone_group)
                        self._mark_as_covered(uncovered, covered_blocks)
        
        # 4. Process SEMANTIC matches (Channel C)
        for group_blocks in normalized_groups.values():
            if len(group_blocks) >= self.min_instances:
                uncovered = self._filter_uncovered(group_blocks, covered_blocks)
                if len(uncovered) >= self.min_instances:
                    clone_group = self._create_clone_group(
                        uncovered, CloneType.SEMANTIC, {FingerprintChannel.NORMALIZED}
                    )
                    if clone_group:
                        clone_groups.append(clone_group)
                        self._mark_as_covered(uncovered, covered_blocks)
        
        # Rank and filter clone groups
        return self._rank_clone_groups(clone_groups)
    
    def _group_by_fingerprint(self, blocks: List[BlockInfo], 
                            fingerprint_attr: str) -> Dict[str, List[BlockInfo]]:
        """Group blocks by a specific fingerprint attribute."""
        groups: Dict[str, List[BlockInfo]] = defaultdict(list)
        
        for block in blocks:
            fingerprint = getattr(block, fingerprint_attr, None)
            if fingerprint:  # Skip None or empty fingerprints
                groups[fingerprint].append(block)
        
        return dict(groups)
    
    def _make_group_id(self, clone_type: CloneType, blocks: List[BlockInfo],
                      channels: Set[FingerprintChannel]) -> str:
        """Generate a stable, deterministic group ID."""
        sorted_keys = sorted(
            f"{b.file_reference.file_path}:{b.file_reference.line_start}-{b.file_reference.line_end}"
            for b in blocks
        )
        channel_str = ",".join(sorted(c.value for c in channels))
        payload = f"{clone_type.value}|{channel_str}|{'|'.join(sorted_keys)}"
        digest = hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]
        return f"{clone_type.value}_{digest}"

    def _create_clone_group(self, blocks: List[BlockInfo], clone_type: CloneType,
                          channels: Set[FingerprintChannel]) -> Optional[CloneGroup]:
        """Create a clone group from a list of similar blocks."""
        if len(blocks) < self.min_instances:
            return None
        
        group_id = self._make_group_id(clone_type, blocks, channels)
        
        # Create clone instances
        instances = []
        for block in blocks:
            similarity_score = self._calculate_similarity_score(clone_type, len(blocks))
            evidence = self._create_evidence(block, blocks, clone_type, similarity_score)
            extraction_feasibility = self._calculate_extraction_feasibility(block)
            
            instance = CloneInstance(
                block_info=block,
                similarity_score=similarity_score,
                evidence=evidence,
                extraction_feasibility=extraction_feasibility
            )
            instances.append(instance)
        
        group_evidence = self._create_group_evidence(instances, clone_type)
        
        return CloneGroup(
            group_id=group_id,
            clone_type=clone_type,
            instances=instances,
            detection_channels=channels,
            evidence=group_evidence
        )
    
    def _calculate_similarity_score(self, clone_type: CloneType, 
                                   group_size: int) -> float:
        """Calculate similarity score based on clone type and group size."""
        base_scores = {
            CloneType.EXACT: self.exact_threshold,
            CloneType.STRUCTURAL: self.structural_threshold,
            CloneType.SEMANTIC: self.semantic_threshold,
            CloneType.HYBRID: 0.85
        }
        base = base_scores.get(clone_type, 0.8)
        
        # Larger groups indicate higher confidence
        size_bonus = min(0.05, group_size * 0.01)
        
        return self._clamp01(base + size_bonus)
    
    def _calculate_extraction_feasibility(self, block: BlockInfo) -> float:
        """Calculate how feasible it is to extract this block."""
        feasibility = 0.5  # Base feasibility
        
        # Size factor
        if block.lines_of_code >= 10:
            feasibility += 0.2
        elif block.lines_of_code >= 5:
            feasibility += 0.1
        
        # Statement count factor
        if block.statement_count >= 5:
            feasibility += 0.2
        elif block.statement_count >= 3:
            feasibility += 0.1
        
        # Nesting level factor (lower nesting is better)
        if block.nesting_level <= 2:
            feasibility += 0.1
        elif block.nesting_level >= 4:
            feasibility -= 0.1
        
        # Block type factor
        bt_value = (block.block_type.value 
                   if hasattr(block.block_type, 'value') 
                   else str(block.block_type))
        
        if bt_value in ('function_body', 'method_body'):
            feasibility += 0.1
        elif bt_value in ('if_block', 'for_loop'):
            feasibility += 0.05
        
        return self._clamp01(feasibility)
    
    def _create_evidence(self, block: BlockInfo, group_blocks: List[BlockInfo],
                        clone_type: CloneType, similarity_score: float) -> Evidence:
        """Create evidence for a clone instance."""
        file_refs = [block.file_reference]
        
        # Add references to other blocks (limit to avoid large evidence)
        for other_block in group_blocks[:4]:
            if other_block is not block:
                file_refs.append(other_block.file_reference)
        
        ref = block.file_reference
        code_snippets = [f"Block at {ref.file_path}:{ref.line_start}-{ref.line_end}"]
        description = f"{clone_type.value.title()} clone detected with {len(group_blocks)} instances"
        
        metadata = {
            'clone_type': clone_type.value,
            'group_size': len(group_blocks),
            'lines_of_code': block.lines_of_code,
            'statement_count': block.statement_count,
            'nesting_level': block.nesting_level
        }
        
        return Evidence(
            description=description,
            confidence=self._clamp01(similarity_score),
            file_references=file_refs[:5],
            code_snippets=code_snippets,
            metadata=metadata
        )
    
    def _create_group_evidence(self, instances: List[CloneInstance],
                             clone_type: CloneType) -> Evidence:
        """Create evidence for the entire clone group."""
        if not instances:
            return Evidence(
                description="Empty clone group",
                confidence=0.0,
                file_references=[],
                code_snippets=[],
                metadata={}
            )
        
        file_refs = [inst.block_info.file_reference for inst in instances]
        code_snippets = [
            f"Clone instance at {inst.file_path}:{inst.line_start}-{inst.line_end}"
            for inst in instances[:5]
        ]
        
        description = f"{clone_type.value.title()} clone group with {len(instances)} instances"
        
        # Calculate group statistics
        n = len(instances)
        avg_loc = sum(inst.block_info.lines_of_code for inst in instances) / n
        avg_statements = sum(inst.block_info.statement_count for inst in instances) / n
        avg_similarity = sum(inst.similarity_score for inst in instances) / n
        
        metadata = {
            'clone_type': clone_type.value,
            'instance_count': n,
            'average_lines_of_code': round(avg_loc, 1),
            'average_statement_count': round(avg_statements, 1),
            'average_similarity_score': round(avg_similarity, 3),
            'files_affected': len(set(inst.file_path for inst in instances))
        }
        
        return Evidence(
            description=description,
            confidence=self._clamp01(avg_similarity),
            file_references=file_refs,
            code_snippets=code_snippets,
            metadata=metadata
        )
    
    def _detect_hybrid_clones(self, blocks: List[BlockInfo],
                            covered_blocks: Set[BlockKey]) -> List[CloneGroup]:
        """
        Detect hybrid clones matching 2 of 3 channels.
        
        Optimized to O(N) using pairwise fingerprint grouping.
        """
        hybrid_groups: List[CloneGroup] = []
        
        # Group by pairs of fingerprints
        pair_es: Dict[Tuple[str, str], List[BlockInfo]] = defaultdict(list)
        pair_en: Dict[Tuple[str, str], List[BlockInfo]] = defaultdict(list)
        pair_sn: Dict[Tuple[str, str], List[BlockInfo]] = defaultdict(list)
        
        for block in blocks:
            key = self._get_block_key(block)
            if key in covered_blocks:
                continue
            
            tf = block.token_fingerprint
            af = block.ast_fingerprint
            nf = block.normalized_fingerprint
            
            # Only group if both fingerprints in pair exist
            if tf and af:
                pair_es[(tf, af)].append(block)
            if tf and nf:
                pair_en[(tf, nf)].append(block)
            if af and nf:
                pair_sn[(af, nf)].append(block)
        
        # Process each pair type
        seen_group_ids: Set[str] = set()
        
        pair_configs = [
            (pair_es, {FingerprintChannel.EXACT, FingerprintChannel.STRUCTURAL}),
            (pair_en, {FingerprintChannel.EXACT, FingerprintChannel.NORMALIZED}),
            (pair_sn, {FingerprintChannel.STRUCTURAL, FingerprintChannel.NORMALIZED}),
        ]
        
        for pair_map, channels in pair_configs:
            for group_blocks in pair_map.values():
                if len(group_blocks) < self.min_instances:
                    continue
                
                # Re-filter for blocks covered in previous iterations
                uncovered = self._filter_uncovered(group_blocks, covered_blocks)
                if len(uncovered) < self.min_instances:
                    continue
                
                clone_group = self._create_clone_group(
                    uncovered, CloneType.HYBRID, channels
                )
                
                if clone_group and clone_group.group_id not in seen_group_ids:
                    hybrid_groups.append(clone_group)
                    seen_group_ids.add(clone_group.group_id)
                    self._mark_as_covered(uncovered, covered_blocks)
        
        return hybrid_groups
    
    def _rank_clone_groups(self, clone_groups: List[CloneGroup]) -> List[CloneGroup]:
        """Rank clone groups by importance and filter low-quality groups."""
        for group in clone_groups:
            group.ranking_score = self._calculate_ranking_score(group)
        
        # Sort by ranking score (descending)
        ranked_groups = sorted(clone_groups, key=lambda g: g.ranking_score, reverse=True)
        
        # Filter out low-quality groups
        return [g for g in ranked_groups if g.ranking_score > 0.3]
    
    def _calculate_ranking_score(self, group: CloneGroup) -> float:
        """Calculate ranking score for a clone group."""
        if not group.instances:
            return 0.0
        
        n = len(group.instances)
        
        # Base score from similarity (30%)
        score = group.similarity_score * 0.3
        
        # Instance count factor (20%) - cap at 10 instances
        instance_factor = min(1.0, n / 10.0)
        score += instance_factor * 0.2
        
        # Size factor (20%) - average LOC, cap at 20
        avg_loc = sum(inst.block_info.lines_of_code for inst in group.instances) / n
        size_factor = min(1.0, avg_loc / 20.0)
        score += size_factor * 0.2
        
        # Extraction confidence factor (20%)
        score += group.extraction_confidence * 0.2
        
        # Clone type weight multiplier
        type_weights = {
            CloneType.EXACT: 1.0,
            CloneType.STRUCTURAL: 0.9,
            CloneType.HYBRID: 0.85,
            CloneType.SEMANTIC: 0.7
        }
        score *= type_weights.get(group.clone_type, 0.5)
        
        # File diversity bonus (clones across files are more important)
        unique_files = len(set(inst.file_path for inst in group.instances))
        if unique_files > 1:
            score += 0.1
        
        return self._clamp01(score)
    
    def get_clone_statistics(self, clone_groups: List[CloneGroup]) -> Dict[str, Any]:
        """Get statistics about detected clone groups."""
        if not clone_groups:
            return {
                'total_groups': 0,
                'total_instances': 0,
                'by_type': {},
                'by_extraction_strategy': {},
                'average_similarity': 0.0,
                'files_affected': 0
            }
        
        total_instances = sum(len(g.instances) for g in clone_groups)
        
        by_type: Dict[str, int] = defaultdict(int)
        for group in clone_groups:
            by_type[group.clone_type.value] += 1
        
        by_strategy: Dict[str, int] = defaultdict(int)
        for group in clone_groups:
            if group.extraction_strategy:
                by_strategy[group.extraction_strategy.value] += 1
        
        avg_similarity = sum(g.similarity_score for g in clone_groups) / len(clone_groups)
        
        all_files = {inst.file_path for g in clone_groups for inst in g.instances}
        
        return {
            'total_groups': len(clone_groups),
            'total_instances': total_instances,
            'by_type': dict(by_type),
            'by_extraction_strategy': dict(by_strategy),
            'average_similarity': round(avg_similarity, 3),
            'files_affected': len(all_files)
        }