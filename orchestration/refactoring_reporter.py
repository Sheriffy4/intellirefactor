"""
Refactoring Final Reporter for IntelliRefactor

Creates comprehensive reports on refactoring activities.
Extracted and adapted from the recon project to work with any Python project.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import re

from ..config import RefactoringConfig


@dataclass
class RefactoringStats:
    """Statistics for refactoring operations."""
    garbage_files_found: int = 0
    garbage_files_moved: int = 0
    size_freed_bytes: int = 0
    directories_analyzed: int = 0
    entry_points_found: int = 0
    config_files_found: int = 0
    modules_analyzed: int = 0
    categories_found: int = 0
    documents_created: int = 0
    refactoring_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0


class RefactoringReporter:
    """Generator for comprehensive refactoring reports."""
    
    def __init__(self, project_root: Path = None, config: Optional[RefactoringConfig] = None):
        """
        Initialize the reporter.
        
        Args:
            project_root: Root directory of the project
            config: Configuration for reporting operations
        """
        self.project_root = project_root or Path.cwd()
        self.config = config or RefactoringConfig()
        self.stats = RefactoringStats()
    
    def generate_report(self, results: List[Dict[str, Any]] = None, output_format: str = 'dict') -> str:
        """
        Generate a comprehensive report on refactoring activities.
        
        Args:
            results: List of refactoring results (optional)
            output_format: Output format ('dict', 'json', 'text', 'html')
            
        Returns:
            Generated report in the specified format
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'stats': {},
            'documents': {},
            'cleanup': {},
            'refactoring_results': {},
            'validation': {},
            'summary': {}
        }
        
        # Analyze refactoring results if provided
        if results:
            self._analyze_refactoring_results(report, results)
        
        # Analyze cleanup results
        self._analyze_cleanup_results(report)
        
        # Analyze created documents
        self._analyze_created_documents(report)
        
        # Analyze project structure
        self._analyze_project_structure(report)
        
        # Analyze module registry
        self._analyze_module_registry(report)
        
        # Create summary
        self._create_summary(report)
        
        # Format output based on requested format
        if output_format.lower() == 'json':
            return json.dumps(report, indent=2, default=str)
        elif output_format.lower() == 'text':
            return self._format_text_report(report)
        elif output_format.lower() == 'html':
            return self._format_html_report(report)
        else:
            # Return JSON string for backward compatibility
            return json.dumps(report, indent=2, default=str)
    
    def _analyze_refactoring_results(self, report: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
        """Analyze refactoring operation results."""
        refactoring_data = {
            'total_operations': len(results),
            'successful_operations': 0,
            'failed_operations': 0,
            'operations_by_type': {},
            'metrics_improvements': {},
            'file_changes': 0
        }
        
        for result in results:
            # Count success/failure
            if result.get('status') == 'completed' or result.get('success', False):
                refactoring_data['successful_operations'] += 1
                self.stats.successful_operations += 1
            else:
                refactoring_data['failed_operations'] += 1
                self.stats.failed_operations += 1
            
            # Count by operation type
            operation_type = result.get('operation_type', 'unknown')
            refactoring_data['operations_by_type'][operation_type] = \
                refactoring_data['operations_by_type'].get(operation_type, 0) + 1
            
            # Count file changes
            changes = result.get('changes', [])
            refactoring_data['file_changes'] += len(changes)
            
            # Analyze metrics improvements
            if 'metrics_before' in result and 'metrics_after' in result:
                self._analyze_metrics_improvement(
                    result['metrics_before'], 
                    result['metrics_after'], 
                    refactoring_data['metrics_improvements']
                )
        
        self.stats.refactoring_operations = len(results)
        report['refactoring_results'] = refactoring_data
    
    def _analyze_metrics_improvement(self, before: Dict[str, Any], after: Dict[str, Any], improvements: Dict[str, Any]) -> None:
        """Analyze metrics improvements from refactoring."""
        metrics_to_check = ['cyclomatic_complexity', 'maintainability_index', 'code_duplication', 'lines_of_code']
        
        for metric in metrics_to_check:
            if metric in before and metric in after:
                before_val = before[metric]
                after_val = after[metric]
                
                if metric not in improvements:
                    improvements[metric] = {'improved': 0, 'degraded': 0, 'unchanged': 0}
                
                # For maintainability index, higher is better
                if metric == 'maintainability_index':
                    if after_val > before_val:
                        improvements[metric]['improved'] += 1
                    elif after_val < before_val:
                        improvements[metric]['degraded'] += 1
                    else:
                        improvements[metric]['unchanged'] += 1
                else:
                    # For other metrics, lower is generally better
                    if after_val < before_val:
                        improvements[metric]['improved'] += 1
                    elif after_val > before_val:
                        improvements[metric]['degraded'] += 1
                    else:
                        improvements[metric]['unchanged'] += 1
    
    def _analyze_cleanup_results(self, report: Dict[str, Any]) -> None:
        """Analyze cleanup results."""
        to_delete_path = self.project_root / '_to_delete'
        
        cleanup_data = {
            'to_delete_exists': to_delete_path.exists(),
            'moved_files': 0,
            'categories': {},
            'total_size': 0
        }
        
        if to_delete_path.exists():
            # Count moved files
            moved_files = list(to_delete_path.rglob('*'))
            moved_files = [f for f in moved_files if f.is_file()]
            cleanup_data['moved_files'] = len(moved_files)
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in moved_files if f.exists())
            cleanup_data['total_size'] = total_size
            self.stats.size_freed_bytes = total_size
            
            # Analyze categories by folders
            categories = {}
            for file_path in moved_files:
                # Determine category by parent folder
                parent = file_path.parent.name
                if parent != '_to_delete':
                    categories[parent] = categories.get(parent, 0) + 1
            
            cleanup_data['categories'] = categories
            self.stats.garbage_files_moved = len(moved_files)
        
        report['cleanup'] = cleanup_data
    
    def _analyze_created_documents(self, report: Dict[str, Any]) -> None:
        """Analyze created documents."""
        expected_docs = [
            'PROJECT_STRUCTURE.md',
            'MODULE_REGISTRY.md',
            'LLM_CONTEXT.md'
        ]
        
        documents_data = {
            'expected': len(expected_docs),
            'created': 0,
            'details': {}
        }
        
        for doc_name in expected_docs:
            doc_path = self.project_root / doc_name
            doc_info = {
                'exists': doc_path.exists(),
                'size': 0,
                'lines': 0,
                'created_time': None
            }
            
            if doc_path.exists():
                documents_data['created'] += 1
                doc_info['size'] = doc_path.stat().st_size
                
                try:
                    content = doc_path.read_text(encoding='utf-8')
                    doc_info['lines'] = len(content.splitlines())
                except Exception:
                    doc_info['lines'] = 0
                
                # Creation time
                doc_info['created_time'] = datetime.fromtimestamp(
                    doc_path.stat().st_mtime
                ).isoformat()
            
            documents_data['details'][doc_name] = doc_info
        
        self.stats.documents_created = documents_data['created']
        report['documents'] = documents_data
    
    def _analyze_project_structure(self, report: Dict[str, Any]) -> None:
        """Analyze PROJECT_STRUCTURE.md."""
        doc_path = self.project_root / 'PROJECT_STRUCTURE.md'
        
        structure_data = {
            'exists': doc_path.exists(),
            'directories_documented': 0,
            'entry_points_found': 0,
            'config_files_found': 0
        }
        
        if doc_path.exists():
            try:
                content = doc_path.read_text(encoding='utf-8')
                
                # Count directories (lines with ###)
                directories = len(re.findall(r'^###\s+', content, re.MULTILINE))
                structure_data['directories_documented'] = directories
                self.stats.directories_analyzed = directories
                
                # Count entry points
                entry_points = len(re.findall(r'entry.?point', content, re.IGNORECASE))
                structure_data['entry_points_found'] = entry_points
                self.stats.entry_points_found = entry_points
                
                # Count configuration files
                config_files = len(re.findall(r'\.(json|yaml|yml|ini|conf|toml)', content))
                structure_data['config_files_found'] = config_files
                self.stats.config_files_found = config_files
                
            except Exception as e:
                structure_data['error'] = str(e)
        
        report['structure'] = structure_data
    
    def _analyze_module_registry(self, report: Dict[str, Any]) -> None:
        """Analyze MODULE_REGISTRY.md."""
        doc_path = self.project_root / 'MODULE_REGISTRY.md'
        
        registry_data = {
            'exists': doc_path.exists(),
            'modules_documented': 0,
            'categories_found': 0,
            'categories': {}
        }
        
        if doc_path.exists():
            try:
                content = doc_path.read_text(encoding='utf-8')
                
                # Count modules (flexible patterns)
                module_patterns = [
                    r'^##\s+.*\.py',  # ## module.py
                    r'^\*\*.*\.py\*\*',  # **module.py**
                    r'###\s+.*\.py'  # ### module.py
                ]
                
                modules = 0
                for pattern in module_patterns:
                    matches = re.findall(pattern, content, re.MULTILINE)
                    modules += len(matches)
                
                registry_data['modules_documented'] = modules
                self.stats.modules_analyzed = modules
                
                # Count categories (lines with #, but not document title)
                categories = re.findall(r'^#\s+(.+)', content, re.MULTILINE)
                # Exclude document title and general headers
                categories = [cat for cat in categories if not any(word in cat.lower() 
                    for word in ['module registry', 'overview', 'modules', 'categories'])]
                registry_data['categories_found'] = len(categories)
                self.stats.categories_found = len(categories)
                
                # Details by categories
                for category in categories:
                    # Count modules in each category
                    category_pattern = rf'^#\s+{re.escape(category)}.*?(?=^#|\Z)'
                    category_section = re.search(category_pattern, content, re.MULTILINE | re.DOTALL)
                    if category_section:
                        category_content = category_section.group(0)
                        module_count = 0
                        for pattern in module_patterns:
                            matches = re.findall(pattern, category_content, re.MULTILINE)
                            module_count += len(matches)
                        registry_data['categories'][category] = module_count
                
            except Exception as e:
                registry_data['error'] = str(e)
        
        report['registry'] = registry_data
    
    def _create_summary(self, report: Dict[str, Any]) -> None:
        """Create final summary."""
        summary = {
            'overall_success': True,
            'completed_stages': [],
            'issues': [],
            'recommendations': []
        }
        
        # Check cleanup stage
        if report['cleanup']['to_delete_exists']:
            summary['completed_stages'].append('Project cleanup')
            if report['cleanup']['moved_files'] == 0:
                summary['issues'].append('No garbage files found or moved')
        else:
            summary['issues'].append('Cleanup stage not executed')
        
        # Check document creation
        docs_created = report['documents']['created']
        docs_expected = report['documents']['expected']
        
        if docs_created == docs_expected:
            summary['completed_stages'].append('Documentation creation')
        else:
            summary['issues'].append(f'Created {docs_created} of {docs_expected} documents')
            if docs_created == 0:
                summary['overall_success'] = False
        
        # Check structure analysis
        if report.get('structure', {}).get('exists'):
            summary['completed_stages'].append('Project structure analysis')
        else:
            summary['issues'].append('PROJECT_STRUCTURE.md not created')
        
        # Check module registry
        if report.get('registry', {}).get('exists'):
            summary['completed_stages'].append('Module registry creation')
        else:
            summary['issues'].append('MODULE_REGISTRY.md not created')
        
        # Check refactoring results if present
        if 'refactoring_results' in report:
            refactoring = report['refactoring_results']
            if refactoring['successful_operations'] > 0:
                summary['completed_stages'].append('Refactoring operations')
            if refactoring['failed_operations'] > 0:
                summary['issues'].append(f"{refactoring['failed_operations']} refactoring operations failed")
        
        # Generate recommendations
        if self.stats.modules_analyzed > 100:
            summary['recommendations'].append('Consider splitting large modules for better maintainability')
        
        if self.stats.categories_found < 5:
            summary['recommendations'].append('Consider reviewing module categorization')
        
        if report['cleanup']['moved_files'] > 200:
            summary['recommendations'].append('Consider setting up automatic cleanup processes')
        
        if 'refactoring_results' in report:
            refactoring = report['refactoring_results']
            success_rate = refactoring['successful_operations'] / max(refactoring['total_operations'], 1)
            if success_rate < 0.8:
                summary['recommendations'].append('Review failed refactoring operations for patterns')
        
        report['summary'] = summary
        report['stats'] = asdict(self.stats)
    
    def save_report(self, report: Dict[str, Any], filename: str = None) -> Path:
        """
        Save report to file.
        
        Args:
            report: Report data
            filename: Filename (auto-generated if not provided)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'refactoring_report_{timestamp}.json'
        
        report_path = self.project_root / filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report_path
    
    def print_report(self, report: Dict[str, Any]) -> None:
        """Print report to console."""
        print("="*60)
        print("COMPREHENSIVE REFACTORING REPORT")
        print("="*60)
        
        # Basic information
        print(f"Project: {report['project_root']}")
        print(f"Report generated: {report['timestamp']}")
        print()
        
        # Statistics
        stats = report['stats']
        print("📊 STATISTICS:")
        print(f"  Garbage files moved: {stats['garbage_files_moved']}")
        print(f"  Space freed: {self._format_size(stats['size_freed_bytes'])}")
        print(f"  Directories analyzed: {stats['directories_analyzed']}")
        print(f"  Entry points found: {stats['entry_points_found']}")
        print(f"  Configuration files: {stats['config_files_found']}")
        print(f"  Modules analyzed: {stats['modules_analyzed']}")
        print(f"  Functional categories: {stats['categories_found']}")
        print(f"  Documents created: {stats['documents_created']}")
        if stats['refactoring_operations'] > 0:
            print(f"  Refactoring operations: {stats['refactoring_operations']}")
            print(f"  Successful operations: {stats['successful_operations']}")
            print(f"  Failed operations: {stats['failed_operations']}")
        print()
        
        # Created documents
        print("📄 CREATED DOCUMENTS:")
        for doc_name, doc_info in report['documents']['details'].items():
            status = "✅" if doc_info['exists'] else "❌"
            print(f"  {status} {doc_name}")
            if doc_info['exists']:
                print(f"      Size: {self._format_size(doc_info['size'])}")
                print(f"      Lines: {doc_info['lines']}")
        print()
        
        # Refactoring results
        if 'refactoring_results' in report and report['refactoring_results']['total_operations'] > 0:
            refactoring = report['refactoring_results']
            print("🔧 REFACTORING RESULTS:")
            print(f"  Total operations: {refactoring['total_operations']}")
            print(f"  Successful: {refactoring['successful_operations']}")
            print(f"  Failed: {refactoring['failed_operations']}")
            print(f"  File changes: {refactoring['file_changes']}")
            
            if refactoring['operations_by_type']:
                print("  Operations by type:")
                for op_type, count in refactoring['operations_by_type'].items():
                    print(f"    {op_type}: {count}")
            
            if refactoring['metrics_improvements']:
                print("  Metrics improvements:")
                for metric, improvements in refactoring['metrics_improvements'].items():
                    improved = improvements['improved']
                    degraded = improvements['degraded']
                    print(f"    {metric}: {improved} improved, {degraded} degraded")
            print()
        
        # Module categories
        if report.get('registry', {}).get('categories'):
            print("🏷️  MODULE CATEGORIES:")
            for category, count in report['registry']['categories'].items():
                print(f"  {category}: {count} modules")
            print()
        
        # Cleanup results
        if report['cleanup']['categories']:
            print("🧹 CLEANUP BY CATEGORIES:")
            for category, count in report['cleanup']['categories'].items():
                print(f"  {category}: {count} files")
            print()
        
        # Final summary
        summary = report['summary']
        print("📋 SUMMARY:")
        print(f"  Overall result: {'✅ SUCCESS' if summary['overall_success'] else '⚠️  ISSUES FOUND'}")
        print(f"  Completed stages: {len(summary['completed_stages'])}")
        for stage in summary['completed_stages']:
            print(f"    ✅ {stage}")
        
        if summary['issues']:
            print(f"  Issues found: {len(summary['issues'])}")
            for issue in summary['issues']:
                print(f"    ⚠️  {issue}")
        
        if summary['recommendations']:
            print(f"  Recommendations: {len(summary['recommendations'])}")
            for rec in summary['recommendations']:
                print(f"    💡 {rec}")
        
        print("="*60)
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable form."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def _format_text_report(self, report: Dict[str, Any]) -> str:
        """Format report as plain text."""
        lines = []
        lines.append("=" * 60)
        lines.append("COMPREHENSIVE REFACTORING REPORT")
        lines.append("=" * 60)
        
        # Basic information
        lines.append(f"Project: {report['project_root']}")
        lines.append(f"Report generated: {report['timestamp']}")
        lines.append("")
        
        # Statistics
        stats = report.get('stats', {})
        lines.append("📊 STATISTICS:")
        lines.append(f"  Refactoring operations: {stats.get('refactoring_operations', 0)}")
        lines.append(f"  Successful operations: {stats.get('successful_operations', 0)}")
        lines.append(f"  Failed operations: {stats.get('failed_operations', 0)}")
        lines.append("")
        
        # Refactoring results
        refactoring = report.get('refactoring_results', {})
        if refactoring.get('total_operations', 0) > 0:
            lines.append("🔧 REFACTORING RESULTS:")
            lines.append(f"  Total operations: {refactoring['total_operations']}")
            lines.append(f"  Successful: {refactoring['successful_operations']}")
            lines.append(f"  Failed: {refactoring['failed_operations']}")
            lines.append("")
        
        # Summary
        summary = report.get('summary', {})
        lines.append("📋 SUMMARY:")
        lines.append(f"  Overall result: {'✅ SUCCESS' if summary.get('overall_success', False) else '⚠️  ISSUES FOUND'}")
        
        if summary.get('issues'):
            lines.append(f"  Issues found: {len(summary['issues'])}")
            for issue in summary['issues']:
                lines.append(f"    ⚠️  {issue}")
        
        if summary.get('recommendations'):
            lines.append(f"  Recommendations: {len(summary['recommendations'])}")
            for rec in summary['recommendations']:
                lines.append(f"    💡 {rec}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def _format_html_report(self, report: Dict[str, Any]) -> str:
        """Format report as HTML."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Refactoring Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .stats {{ background-color: #e8f4f8; padding: 10px; border-radius: 5px; }}
        .success {{ color: green; }}
        .warning {{ color: orange; }}
        .error {{ color: red; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Comprehensive Refactoring Report</h1>
        <p><strong>Project:</strong> {report['project_root']}</p>
        <p><strong>Generated:</strong> {report['timestamp']}</p>
    </div>
    
    <div class="section stats">
        <h2>📊 Statistics</h2>
        <ul>
            <li>Refactoring operations: {report.get('stats', {}).get('refactoring_operations', 0)}</li>
            <li>Successful operations: {report.get('stats', {}).get('successful_operations', 0)}</li>
            <li>Failed operations: {report.get('stats', {}).get('failed_operations', 0)}</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>📋 Summary</h2>
        <p class="{'success' if report.get('summary', {}).get('overall_success', False) else 'warning'}">
            Overall result: {'✅ SUCCESS' if report.get('summary', {}).get('overall_success', False) else '⚠️ ISSUES FOUND'}
        </p>
    </div>
</body>
</html>
"""
        return html