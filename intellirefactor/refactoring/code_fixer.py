"""Automatic code fixes and learning integration.

This module provides the CodeFixer class that applies automatic fixes to generated
code, including learned patterns, import normalization, and logging standardization.

The CodeFixer integrates with the self-learning system to continuously improve
code generation quality based on manual corrections.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional integrations
try:
    from ..knowledge.import_fixing_patterns import ImportFixingPatterns
    from ..knowledge.self_learning_patterns import get_learning_system

    IMPORT_FIXING_AVAILABLE = True
    SELF_LEARNING_AVAILABLE = True
except ImportError:
    IMPORT_FIXING_AVAILABLE = False
    SELF_LEARNING_AVAILABLE = False
    logger.debug("Import fixing patterns and/or self-learning not available")


class CodeFixer:
    """Applies automatic fixes to generated code.
    
    The CodeFixer provides intelligent code fixing capabilities including:
    - Application of learned patterns from previous corrections
    - Import statement normalization and fixing
    - Logging standard enforcement
    - Module availability checking with fallbacks
    
    Attributes:
        codebase_analysis: Dictionary containing codebase standards and patterns
    
    Example:
        >>> fixer = CodeFixer(codebase_analysis={'logging_standard': 'logger'})
        >>> fixed_code = fixer.apply_automatic_fixes(code, Path('module.py'))
        >>> # Code is now fixed according to standards
    """
    
    def __init__(self, codebase_analysis: Optional[Dict[str, Any]] = None):
        """Initialize CodeFixer.
        
        Args:
            codebase_analysis: Dictionary with codebase standards including:
                - logging_standard: Preferred logging variable name
                - existing_modules: Set of available modules
        """
        self._codebase_analysis = codebase_analysis or {}
    
    def apply_automatic_fixes(self, content: str, path: Path) -> str:
        """Apply learned patterns and automatic fixes to generated code.
        
        This method applies multiple types of fixes in sequence:
        1. Learned patterns from self-learning system
        2. Import statement fixes
        3. Logging standard enforcement
        4. Module availability checks with fallbacks
        
        Args:
            content: Source code to fix
            path: Path to the file (for logging and context)
            
        Returns:
            Fixed source code
            
        Example:
            >>> fixer = CodeFixer()
            >>> code = "LOG = logging.getLogger(__name__)"
            >>> fixed = fixer.apply_automatic_fixes(code, Path('test.py'))
            >>> # Returns: "logger = logging.getLogger(__name__)"
        """
        original_content = content
        try:
            new_content = content

            # Apply learned patterns
            if SELF_LEARNING_AVAILABLE:
                new_content = self._apply_learned_patterns(new_content, path)

            # Apply import fixes
            if IMPORT_FIXING_AVAILABLE:
                new_content = self._apply_import_fixes(new_content)

            # Apply codebase-specific fixes
            if self._codebase_analysis:
                new_content = self._apply_codebase_fixes(new_content)

            if new_content != original_content:
                logger.info("Applied automatic fixes to %s", path.name)

            return new_content

        except Exception as e:
            logger.warning("Failed to apply automatic fixes to %s: %s", path.name, e)
            return original_content
    
    def _apply_learned_patterns(self, content: str, path: Path) -> str:
        """Apply patterns learned from previous manual corrections.
        
        Args:
            content: Source code to fix
            path: Path to the file
            
        Returns:
            Code with learned patterns applied
        """
        try:
            learning_system = get_learning_system()
            return learning_system.apply_learned_patterns(content, str(path))
        except Exception as e:
            logger.debug("Failed to apply learned patterns: %s", e)
            return content
    
    def _apply_import_fixes(self, content: str) -> str:
        """Apply import statement fixes and normalization.
        
        Args:
            content: Source code to fix
            
        Returns:
            Code with fixed imports
        """
        try:
            return ImportFixingPatterns.apply_all_fixes(content)
        except Exception as e:
            logger.debug("Failed to apply import fixes: %s", e)
            return content
    
    def _apply_codebase_fixes(self, content: str) -> str:
        """Apply codebase-specific fixes based on analysis.
        
        This includes:
        - Logging standard enforcement
        - Module availability checks
        - Import fallbacks for missing modules
        
        Args:
            content: Source code to fix
            
        Returns:
            Code with codebase-specific fixes applied
        """
        new_content = content
        
        # Apply logging standard
        new_content = self._fix_logging_standard(new_content)
        
        # Apply module availability checks
        new_content = self._fix_module_imports(new_content)
        
        return new_content
    
    def _fix_logging_standard(self, content: str) -> str:
        """Enforce logging standard (e.g., 'logger' vs 'LOG').
        
        Args:
            content: Source code to fix
            
        Returns:
            Code with standardized logging
        """
        logging_standard = self._codebase_analysis.get("logging_standard", "logger")
        
        if logging_standard == "logger":
            # Replace LOG = logging.getLogger(...) -> logger = ...
            content = re.sub(
                r"\bLOG\s*=\s*logging\.getLogger",
                "logger = logging.getLogger",
                content,
            )
            content = re.sub(r"\bLOG\.", "logger.", content)
        
        return content
    
    def _fix_module_imports(self, content: str) -> str:
        """Add fallbacks for imports of potentially missing modules.
        
        Args:
            content: Source code to fix
            
        Returns:
            Code with import fallbacks added
        """
        existing_modules = self._codebase_analysis.get("existing_modules", set())
        lines = content.splitlines()
        fixed_lines: List[str] = []

        for line in lines:
            # Remove known non-existent imports
            if "from core.cli" in line:
                fixed_lines.append(f"# Removed non-existent import: {line.strip()}")
                continue

            # Add fallback for potentially missing modules
            import_match = re.match(r"\s*from\s+(core\.[^\s]+)\s+import\s+", line)
            if import_match:
                module = import_match.group(1)
                if module not in existing_modules:
                    fixed_lines.append("# Import fallback for missing module")
                    fixed_lines.append("try:")
                    fixed_lines.append(f"    {line}")
                    fixed_lines.append("except ImportError:")
                    fixed_lines.append("    pass  # Module not available")
                    continue

            fixed_lines.append(line)

        return "\n".join(fixed_lines) + ("\n" if content.endswith("\n") else "")
    
    def learn_from_manual_corrections(
        self,
        original_file: Path,
        corrected_files: List[Path],
        description: str = ""
    ) -> Dict[str, Any]:
        """Learn patterns from manual corrections to improve future generations.
        
        This method analyzes differences between original generated code and
        manually corrected versions to extract reusable patterns.
        
        Args:
            original_file: Path to original generated file
            corrected_files: List of paths to manually corrected versions
            description: Optional description of the corrections
            
        Returns:
            Dictionary with learning results:
                - success: Whether learning succeeded
                - patterns_extracted: Number of patterns learned
                - pattern_names: Names of extracted patterns
                - total_patterns: Total patterns in knowledge base
                - learning_sessions: Number of learning sessions
                - codebase_standards: Detected codebase standards
                
        Example:
            >>> fixer = CodeFixer()
            >>> result = fixer.learn_from_manual_corrections(
            ...     Path('generated.py'),
            ...     [Path('corrected_v1.py'), Path('corrected_v2.py')],
            ...     'Fixed import order and logging'
            ... )
            >>> print(f"Learned {result['patterns_extracted']} patterns")
        """
        if not SELF_LEARNING_AVAILABLE:
            return {
                "success": False,
                "error": "Self-learning system not available",
                "patterns_extracted": 0,
            }

        try:
            learning_system = get_learning_system()
            patterns = learning_system.analyze_manual_corrections(
                str(original_file),
                [str(f) for f in corrected_files],
                description,
            )
            stats = learning_system.get_pattern_statistics()
            logger.info("Learning completed: extracted %d patterns", len(patterns))

            return {
                "success": True,
                "patterns_extracted": len(patterns),
                "pattern_names": [p.name for p in patterns],
                "total_patterns": stats.get("total_patterns"),
                "learning_sessions": stats.get("learning_sessions"),
                "codebase_standards": stats.get("codebase_standards"),
            }

        except Exception as e:
            logger.error("Failed to learn from manual corrections: %s", e)
            return {"success": False, "error": str(e), "patterns_extracted": 0}
