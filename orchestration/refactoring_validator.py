"""
Refactoring Results Validator for IntelliRefactor

Validates refactoring results and project documentation.
Extracted and adapted from the recon project to work with any Python project.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

from ..config import RefactoringConfig


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    success: bool
    message: str
    details: Dict[str, Any] = None


class RefactoringValidator:
    """Validator for refactoring results and project documentation."""
    
    def __init__(self, project_root: Path = None, config: Optional[RefactoringConfig] = None):
        """
        Initialize the validator.
        
        Args:
            project_root: Root directory of the project
            config: Configuration for validation operations
        """
        self.project_root = project_root or Path.cwd()
        self.config = config or RefactoringConfig()
        
        # Default expected documents - can be configured
        self.expected_docs = [
            'PROJECT_STRUCTURE.md',
            'MODULE_REGISTRY.md', 
            'LLM_CONTEXT.md'
        ]
    
    def set_expected_documents(self, documents: List[str]) -> None:
        """Set the list of expected documents to validate."""
        self.expected_docs = documents
    
    def validate_all(self) -> List[ValidationResult]:
        """
        Execute all validation checks.
        
        Returns:
            List of validation results
        """
        results = []
        
        # Check document existence
        results.append(self._check_documents_exist())
        
        # Check links between documents
        results.append(self._check_document_links())
        
        # Check specific document contents
        if 'PROJECT_STRUCTURE.md' in self.expected_docs:
            results.append(self._check_project_structure_content())
        
        if 'MODULE_REGISTRY.md' in self.expected_docs:
            results.append(self._check_module_registry_content())
        
        if 'LLM_CONTEXT.md' in self.expected_docs:
            results.append(self._check_llm_context_content())
        
        # Check cleanup results
        results.append(self._check_cleanup_results())
        
        return results
    
    def validate_refactoring(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a refactoring result for correctness and safety.
        
        Args:
            result: Refactoring result to validate
            
        Returns:
            Validation report
        """
        validation_results = []
        
        # Validate refactoring result structure
        validation_results.append(self._validate_result_structure(result))
        
        # Validate file changes if present
        if 'changes' in result:
            validation_results.append(self._validate_file_changes(result['changes']))
        
        # Validate metrics if present
        if 'metrics_before' in result and 'metrics_after' in result:
            validation_results.append(self._validate_metrics_improvement(
                result['metrics_before'], result['metrics_after']
            ))
        
        # Calculate overall validation status
        overall_success = all(vr.success for vr in validation_results)
        
        return {
            "status": "passed" if overall_success else "failed",
            "overall_success": overall_success,
            "validation_results": [
                {
                    "check": vr.check_name,
                    "success": vr.success,
                    "message": vr.message,
                    "details": vr.details or {}
                }
                for vr in validation_results
            ],
            "summary": {
                "total_checks": len(validation_results),
                "passed_checks": sum(1 for vr in validation_results if vr.success),
                "failed_checks": sum(1 for vr in validation_results if not vr.success)
            }
        }
    
    def _validate_result_structure(self, result: Dict[str, Any]) -> ValidationResult:
        """Validate the structure of a refactoring result."""
        required_fields = ['operation_id', 'status']
        missing_fields = [field for field in required_fields if field not in result]
        
        if missing_fields:
            return ValidationResult(
                check_name="Result Structure",
                success=False,
                message=f"Missing required fields: {', '.join(missing_fields)}",
                details={'missing_fields': missing_fields}
            )
        
        return ValidationResult(
            check_name="Result Structure",
            success=True,
            message="Refactoring result has valid structure",
            details={'fields_present': list(result.keys())}
        )
    
    def _validate_file_changes(self, changes: List[Dict[str, Any]]) -> ValidationResult:
        """Validate file changes in a refactoring result."""
        if not changes:
            return ValidationResult(
                check_name="File Changes",
                success=True,
                message="No file changes to validate",
                details={'changes_count': 0}
            )
        
        invalid_changes = []
        for i, change in enumerate(changes):
            if 'file_path' not in change or 'change_type' not in change:
                invalid_changes.append(f"Change {i}: missing file_path or change_type")
        
        if invalid_changes:
            return ValidationResult(
                check_name="File Changes",
                success=False,
                message=f"Invalid changes found: {len(invalid_changes)}",
                details={'invalid_changes': invalid_changes}
            )
        
        return ValidationResult(
            check_name="File Changes",
            success=True,
            message=f"All {len(changes)} file changes are valid",
            details={'changes_count': len(changes)}
        )
    
    def _validate_metrics_improvement(self, metrics_before: Dict[str, Any], metrics_after: Dict[str, Any]) -> ValidationResult:
        """Validate that refactoring improved code metrics."""
        improvements = []
        regressions = []
        
        # Check common metrics for improvement
        improvement_metrics = ['cyclomatic_complexity', 'maintainability_index', 'code_duplication']
        
        for metric in improvement_metrics:
            if metric in metrics_before and metric in metrics_after:
                before = metrics_before[metric]
                after = metrics_after[metric]
                
                # For complexity and duplication, lower is better
                # For maintainability index, higher is better
                if metric == 'maintainability_index':
                    if after > before:
                        improvements.append(f"{metric}: {before} -> {after}")
                    elif after < before:
                        regressions.append(f"{metric}: {before} -> {after}")
                else:
                    if after < before:
                        improvements.append(f"{metric}: {before} -> {after}")
                    elif after > before:
                        regressions.append(f"{metric}: {before} -> {after}")
        
        if regressions and not improvements:
            return ValidationResult(
                check_name="Metrics Improvement",
                success=False,
                message=f"Metrics regressed: {', '.join(regressions)}",
                details={'regressions': regressions, 'improvements': improvements}
            )
        
        return ValidationResult(
            check_name="Metrics Improvement",
            success=True,
            message=f"Metrics improved or maintained: {len(improvements)} improvements, {len(regressions)} regressions",
            details={'improvements': improvements, 'regressions': regressions}
        )
    
    def _check_documents_exist(self) -> ValidationResult:
        """Check existence of all expected documents."""
        missing_docs = []
        existing_docs = []
        
        for doc_name in self.expected_docs:
            doc_path = self.project_root / doc_name
            if doc_path.exists():
                existing_docs.append(doc_name)
            else:
                missing_docs.append(doc_name)
        
        if missing_docs:
            return ValidationResult(
                check_name="Document Existence",
                success=False,
                message=f"Missing documents: {', '.join(missing_docs)}",
                details={
                    'missing': missing_docs,
                    'existing': existing_docs
                }
            )
        
        return ValidationResult(
            check_name="Document Existence",
            success=True,
            message=f"All {len(self.expected_docs)} documents found",
            details={'existing': existing_docs}
        )
    
    def _check_document_links(self) -> ValidationResult:
        """Check links between documents."""
        broken_links = []
        working_links = []
        
        # Check links in LLM_CONTEXT.md if it exists
        llm_context_path = self.project_root / 'LLM_CONTEXT.md'
        if llm_context_path.exists():
            content = llm_context_path.read_text(encoding='utf-8')
            
            # Look for references to other documents
            for doc_name in ['PROJECT_STRUCTURE.md', 'MODULE_REGISTRY.md']:
                if doc_name in content:
                    target_path = self.project_root / doc_name
                    if target_path.exists():
                        working_links.append(f"LLM_CONTEXT.md -> {doc_name}")
                    else:
                        broken_links.append(f"LLM_CONTEXT.md -> {doc_name}")
        
        if broken_links:
            return ValidationResult(
                check_name="Document Links",
                success=False,
                message=f"Broken links found: {', '.join(broken_links)}",
                details={
                    'broken': broken_links,
                    'working': working_links
                }
            )
        
        return ValidationResult(
            check_name="Document Links",
            success=True,
            message=f"All links working ({len(working_links)} checked)",
            details={'working': working_links}
        )
    
    def _check_project_structure_content(self) -> ValidationResult:
        """Check content of PROJECT_STRUCTURE.md."""
        doc_path = self.project_root / 'PROJECT_STRUCTURE.md'
        
        if not doc_path.exists():
            return ValidationResult(
                check_name="PROJECT_STRUCTURE.md Content",
                success=False,
                message="PROJECT_STRUCTURE.md file not found"
            )
        
        content = doc_path.read_text(encoding='utf-8')
        
        # Check for essential sections (flexible patterns)
        required_patterns = [
            (r'#.*[Ss]tructur.*[Pp]roject|Project.*Structure', 'Project Structure header'),
            (r'Entry Points?|Entry.*Points?', 'Entry Points section'),
            (r'[Cc]onfiguration.*[Ff]iles?|Config.*Files?', 'Configuration Files section')
        ]
        missing_sections = []
        
        for pattern, description in required_patterns:
            if not re.search(pattern, content, re.IGNORECASE):
                missing_sections.append(description)
        
        if missing_sections:
            return ValidationResult(
                check_name="PROJECT_STRUCTURE.md Content",
                success=False,
                message=f"Missing sections: {', '.join(missing_sections)}",
                details={'missing_sections': missing_sections}
            )
        
        return ValidationResult(
            check_name="PROJECT_STRUCTURE.md Content",
            success=True,
            message="All required sections present",
            details={'content_length': len(content)}
        )
    
    def _check_module_registry_content(self) -> ValidationResult:
        """Check content of MODULE_REGISTRY.md."""
        doc_path = self.project_root / 'MODULE_REGISTRY.md'
        
        if not doc_path.exists():
            return ValidationResult(
                check_name="MODULE_REGISTRY.md Content",
                success=False,
                message="MODULE_REGISTRY.md file not found"
            )
        
        content = doc_path.read_text(encoding='utf-8')
        
        # Check for essential sections (flexible patterns)
        required_patterns = [
            (r'#.*[Mm]odule.*[Rr]egistry|Module Registry', 'Module Registry header'),
            (r'[Cc]ategories|Categories', 'Categories section'),
            (r'[Mm]odules|Modules', 'Modules section')
        ]
        missing_sections = []
        
        for pattern, description in required_patterns:
            if not re.search(pattern, content, re.IGNORECASE):
                missing_sections.append(description)
        
        # Count modules (flexible patterns)
        module_patterns = [
            r'^##\s+.*\.py',  # ## module.py
            r'^\*\*.*\.py\*\*',  # **module.py**
            r'###\s+.*\.py'  # ### module.py
        ]
        
        module_count = 0
        for pattern in module_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            module_count += len(matches)
        
        if missing_sections:
            return ValidationResult(
                check_name="MODULE_REGISTRY.md Content",
                success=False,
                message=f"Missing sections: {', '.join(missing_sections)}",
                details={'missing_sections': missing_sections, 'module_count': module_count}
            )
        
        return ValidationResult(
            check_name="MODULE_REGISTRY.md Content",
            success=True,
            message=f"All sections present, found {module_count} modules",
            details={'module_count': module_count, 'content_length': len(content)}
        )
    
    def _check_llm_context_content(self) -> ValidationResult:
        """Check content of LLM_CONTEXT.md."""
        doc_path = self.project_root / 'LLM_CONTEXT.md'
        
        if not doc_path.exists():
            return ValidationResult(
                check_name="LLM_CONTEXT.md Content",
                success=False,
                message="LLM_CONTEXT.md file not found"
            )
        
        content = doc_path.read_text(encoding='utf-8')
        
        # Check for essential rules (flexible patterns)
        required_patterns = [
            (r'MODULE_REGISTRY\.md', 'MODULE_REGISTRY.md reference'),
            (r'PROJECT_STRUCTURE\.md', 'PROJECT_STRUCTURE.md reference'),
            (r'before creating.*functionality|check.*before.*creating', 'check before creating functionality rule'),
            (r'where to place.*code|code.*placement', 'where to place code rule')
        ]
        
        missing_rules = []
        for pattern, description in required_patterns:
            if not re.search(pattern, content, re.IGNORECASE):
                missing_rules.append(description)
        
        if missing_rules:
            return ValidationResult(
                check_name="LLM_CONTEXT.md Content",
                success=False,
                message=f"Missing rules: {', '.join(missing_rules)}",
                details={'missing_rules': missing_rules}
            )
        
        return ValidationResult(
            check_name="LLM_CONTEXT.md Content",
            success=True,
            message="All required rules present",
            details={'content_length': len(content)}
        )
    
    def _check_cleanup_results(self) -> ValidationResult:
        """Check cleanup results."""
        to_delete_path = self.project_root / '_to_delete'
        
        if not to_delete_path.exists():
            return ValidationResult(
                check_name="Cleanup Results",
                success=True,  # Not finding _to_delete is OK - cleanup might not have run
                message="No _to_delete folder found - cleanup may not have been performed"
            )
        
        # Count files in _to_delete folder
        moved_files = list(to_delete_path.rglob('*'))
        moved_files = [f for f in moved_files if f.is_file()]
        
        return ValidationResult(
            check_name="Cleanup Results",
            success=True,
            message=f"_to_delete folder contains {len(moved_files)} moved files",
            details={'moved_files_count': len(moved_files)}
        )
    
    def print_results(self, results: List[ValidationResult]) -> bool:
        """
        Print validation results.
        
        Args:
            results: List of validation results
            
        Returns:
            True if all checks passed successfully
        """
        print("="*60)
        print("REFACTORING VALIDATION RESULTS")
        print("="*60)
        
        success_count = 0
        total_count = len(results)
        
        for result in results:
            status = "✅" if result.success else "❌"
            print(f"{status} {result.check_name}")
            print(f"   {result.message}")
            
            if result.details:
                for key, value in result.details.items():
                    print(f"   {key}: {value}")
            
            if result.success:
                success_count += 1
            
            print()
        
        overall_success = success_count == total_count
        
        print("="*60)
        if overall_success:
            print("🎉 ALL VALIDATION CHECKS PASSED!")
            print(f"Successful: {success_count}/{total_count}")
        else:
            print("❌ VALIDATION ISSUES FOUND")
            print(f"Successful: {success_count}/{total_count}")
            print(f"Failed: {total_count - success_count}")
        
        return overall_success