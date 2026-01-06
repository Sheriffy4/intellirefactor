"""
Main documentation generator that orchestrates all documentation types.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import asdict

from .architecture_generator import ArchitectureGenerator
from .flowchart_generator import FlowchartGenerator
from .report_generator import ReportGenerator
from .registry_generator import RegistryGenerator
from .llm_context_generator import LLMContextGenerator
from .project_structure_generator import ProjectStructureGenerator

logger = logging.getLogger(__name__)


class DocumentationGenerator:
    """
    Main documentation generator that creates comprehensive documentation
    for Python modules and projects.
    """
    
    def __init__(self, project_path: Union[str, Path]):
        """Initialize documentation generator."""
        self.project_path = Path(project_path)
        self.generators = {
            'architecture': ArchitectureGenerator(),
            'flowchart': FlowchartGenerator(),
            'report': ReportGenerator(),
            'registry': RegistryGenerator(),
            'llm_context': LLMContextGenerator(),
            'project_structure': ProjectStructureGenerator()
        }
        
    def generate_full_documentation(self, 
                                  target_file: Union[str, Path],
                                  output_dir: Optional[Union[str, Path]] = None,
                                  include_types: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Generate complete documentation suite for a target file.
        
        Args:
            target_file: Path to the target Python file
            output_dir: Directory to save documentation (default: current directory)
            include_types: List of documentation types to generate
            
        Returns:
            Dictionary mapping document type to output file path
        """
        target_path = Path(target_file)
        if not target_path.exists():
            raise FileNotFoundError(f"Target file not found: {target_file}")
            
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default documentation types
        if include_types is None:
            include_types = [
                'architecture',
                'flowchart', 
                'call_graph',
                'report',
                'registry',
                'llm_context',
                'project_structure'
            ]
            
        # Generate base name for files
        module_name = target_path.stem.upper()
        
        # Analyze the target file first
        from ..analysis.file_analyzer import FileAnalyzer
        analyzer = FileAnalyzer()
        analysis_result_obj = analyzer.analyze_file(str(target_path))
        
        # Extract the actual data from the GenericAnalysisResult object
        if hasattr(analysis_result_obj, 'data'):
            analysis_result = analysis_result_obj.data
        else:
            # Fallback to converting the object to dict
            analysis_result = asdict(analysis_result_obj) if hasattr(analysis_result_obj, '__dataclass_fields__') else {}
        
        generated_files = {}
        
        try:
            # Generate architecture diagram
            if 'architecture' in include_types:
                arch_file = output_dir / f"{module_name}_ARCHITECTURE_DIAGRAM.md"
                arch_content = self.generators['architecture'].generate_architecture_diagram(
                    analysis_result, target_path.stem
                )
                arch_file.write_text(arch_content, encoding='utf-8')
                generated_files['architecture'] = str(arch_file)
                logger.info(f"Generated architecture diagram: {arch_file}")
                
            # Generate analysis flowchart
            if 'flowchart' in include_types:
                flow_file = output_dir / f"{module_name}_ANALYSIS_FLOWCHART.md"
                flow_content = self.generators['flowchart'].generate_flowchart(analysis_result)
                flow_file.write_text(flow_content, encoding='utf-8')
                generated_files['flowchart'] = str(flow_file)
                logger.info(f"Generated analysis flowchart: {flow_file}")
                
            # Generate detailed call graph (use same flowchart generator)
            if 'call_graph' in include_types:
                call_file = output_dir / f"{module_name}_CALL_GRAPH_DETAILED.md"
                call_content = self._generate_detailed_call_graph(analysis_result, target_path.stem)
                call_file.write_text(call_content, encoding='utf-8')
                generated_files['call_graph'] = str(call_file)
                logger.info(f"Generated detailed call graph: {call_file}")
                
            # Generate refactoring report
            if 'report' in include_types:
                report_file = output_dir / f"{target_path.stem}_refactoring_report.md"
                report_content = self.generators['report'].generate_refactoring_report(target_path)
                report_file.write_text(report_content, encoding='utf-8')
                generated_files['report'] = str(report_file)
                logger.info(f"Generated refactoring report: {report_file}")
                
            # Generate module registry
            if 'registry' in include_types:
                registry_file = output_dir / f"{module_name}_MODULE_REGISTRY.md"
                registry_content = self.generators['registry'].generate_module_registry(target_path)
                registry_file.write_text(registry_content, encoding='utf-8')
                generated_files['registry'] = str(registry_file)
                logger.info(f"Generated module registry: {registry_file}")
                
            # Generate LLM context
            if 'llm_context' in include_types:
                llm_file = output_dir / f"{module_name}_LLM_CONTEXT.md"
                llm_content = self.generators['llm_context'].generate_llm_context(target_path)
                llm_file.write_text(llm_content, encoding='utf-8')
                generated_files['llm_context'] = str(llm_file)
                logger.info(f"Generated LLM context: {llm_file}")
                
            # Generate project structure
            if 'project_structure' in include_types:
                struct_file = output_dir / f"{module_name}_PROJECT_STRUCTURE.md"
                struct_content = self.generators['project_structure'].generate_project_structure(target_path)
                struct_file.write_text(struct_content, encoding='utf-8')
                generated_files['project_structure'] = str(struct_file)
                logger.info(f"Generated project structure: {struct_file}")
                
        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            raise
            
        return generated_files
        
    def _generate_detailed_call_graph(self, analysis_result: Dict[str, Any], module_name: str) -> str:
        """Generate detailed call graph documentation."""
        content = f"""# Detailed Call Graph: {module_name}

## Overview
This document provides a comprehensive call graph analysis for the {module_name} module.

## Call Graph Visualization

```mermaid
graph TD
    subgraph "Module: {module_name}"
"""
        
        # Extract classes and methods from analysis result
        classes = analysis_result.get('classes', {})
        functions = analysis_result.get('functions', {})
        
        # Handle case where classes might be a list instead of dict
        if isinstance(classes, list):
            classes_dict = {}
            for cls in classes:
                if hasattr(cls, 'name'):
                    classes_dict[cls.name] = {
                        'methods': getattr(cls, 'methods', []),
                        'line_start': getattr(cls, 'line_start', 0)
                    }
            classes = classes_dict
        
        # Handle case where functions might be a list instead of dict
        if isinstance(functions, list):
            functions_dict = {}
            for func in functions:
                if hasattr(func, 'name'):
                    functions_dict[func.name] = {
                        'line_start': getattr(func, 'line_start', 0),
                        'complexity': getattr(func, 'complexity', 0)
                    }
            functions = functions_dict
        
        # Add class nodes
        for class_name, class_info in classes.items():
            content += f"        {class_name}[{class_name}]\n"
            methods = class_info.get('methods', [])
            if isinstance(methods, list):
                for method in methods:
                    method_name = getattr(method, 'name', str(method))
                    content += f"        {class_name}_{method_name}[{method_name}]\n"
                    content += f"        {class_name} --> {class_name}_{method_name}\n"
            elif isinstance(methods, dict):
                for method_name in methods:
                    content += f"        {class_name}_{method_name}[{method_name}]\n"
                    content += f"        {class_name} --> {class_name}_{method_name}\n"
        
        # Add function nodes
        for func_name in functions:
            content += f"        {func_name}[{func_name}]\n"
        
        content += """    end
```

## Method Dependencies

"""
        
        # Add method details
        for class_name, class_info in classes.items():
            methods = class_info.get('methods', [])
            if methods:
                content += f"### {class_name} Methods\n\n"
                if isinstance(methods, list):
                    for method in methods:
                        method_name = getattr(method, 'name', str(method))
                        content += f"#### {method_name}\n"
                        content += f"- **Line**: {getattr(method, 'line_start', 'N/A')}\n"
                        content += f"- **Complexity**: {getattr(method, 'complexity', 'N/A')}\n"
                        content += "\n"
                elif isinstance(methods, dict):
                    for method_name, method_info in methods.items():
                        content += f"#### {method_name}\n"
                        content += f"- **Line**: {method_info.get('line_start', 'N/A')}\n"
                        content += f"- **Complexity**: {method_info.get('complexity', 'N/A')}\n"
                        calls = method_info.get('calls', [])
                        if calls:
                            content += f"- **Calls**: {', '.join(calls)}\n"
                        content += "\n"
        
        # Add function details
        if functions:
            content += "### Module Functions\n\n"
            for func_name, func_info in functions.items():
                content += f"#### {func_name}\n"
                if isinstance(func_info, dict):
                    content += f"- **Line**: {func_info.get('line_start', 'N/A')}\n"
                    content += f"- **Complexity**: {func_info.get('complexity', 'N/A')}\n"
                    calls = func_info.get('calls', [])
                    if calls:
                        content += f"- **Calls**: {', '.join(calls)}\n"
                else:
                    content += f"- **Line**: {getattr(func_info, 'line_start', 'N/A')}\n"
                    content += f"- **Complexity**: {getattr(func_info, 'complexity', 'N/A')}\n"
                content += "\n"
        
        content += f"""
## Analysis Summary

- **Total Classes**: {len(classes)}
- **Total Functions**: {len(functions)}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return content
        
    def generate_documentation_type(self, 
                                   target_file: Union[str, Path],
                                   doc_type: str,
                                   output_file: Optional[Union[str, Path]] = None) -> str:
        """
        Generate a specific type of documentation.
        
        Args:
            target_file: Path to the target Python file
            doc_type: Type of documentation to generate
            output_file: Output file path (optional)
            
        Returns:
            Path to generated documentation file
        """
        target_path = Path(target_file)
        
        if doc_type not in self.generators:
            raise ValueError(f"Unknown documentation type: {doc_type}")
            
        # Analyze the target file first
        from ..analysis.file_analyzer import FileAnalyzer
        analyzer = FileAnalyzer()
        analysis_result_obj = analyzer.analyze_file(str(target_path))
        
        # Extract the actual data from the GenericAnalysisResult object
        if hasattr(analysis_result_obj, 'data'):
            analysis_result = analysis_result_obj.data
        else:
            # Fallback to converting the object to dict
            analysis_result = asdict(analysis_result_obj) if hasattr(analysis_result_obj, '__dataclass_fields__') else {}
        
        generator = self.generators[doc_type]
        
        # Generate content based on type
        if doc_type == 'architecture':
            content = generator.generate_architecture_diagram(analysis_result, target_path.stem)
        elif doc_type == 'flowchart':
            content = generator.generate_flowchart(analysis_result)
        elif doc_type == 'report':
            content = generator.generate_refactoring_report(target_path)
        elif doc_type == 'registry':
            content = generator.generate_module_registry(target_path)
        elif doc_type == 'llm_context':
            content = generator.generate_llm_context(target_path)
        elif doc_type == 'project_structure':
            content = generator.generate_project_structure(target_path)
        else:
            raise ValueError(f"Unsupported documentation type: {doc_type}")
            
        # Determine output file
        if output_file is None:
            module_name = target_path.stem.upper()
            output_file = f"{module_name}_{doc_type.upper()}.md"
            
        output_path = Path(output_file)
        output_path.write_text(content, encoding='utf-8')
        
        logger.info(f"Generated {doc_type} documentation: {output_path}")
        return str(output_path)
        
    def list_available_types(self) -> List[str]:
        """List available documentation types."""
        return list(self.generators.keys())
        
    def get_generator_info(self) -> Dict[str, str]:
        """Get information about available generators."""
        info = {}
        for name, generator in self.generators.items():
            if hasattr(generator, '__doc__') and generator.__doc__:
                info[name] = generator.__doc__.strip().split('\n')[0]
            else:
                info[name] = f"{name.title()} generator"
        return info