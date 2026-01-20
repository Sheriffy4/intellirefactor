"""
Decomposition Documentation Generator

A comprehensive documentation generation system for the IntelliRefactor decomposition analysis modules.
Analyzes Python modules and generates structured documentation using a standardized Russian template format.
"""

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('decomposition_doc_generator.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class ImportInfo:
    """Information about a module import."""
    module: str
    names: List[str]
    is_from_import: bool
    level: int  # for relative imports


@dataclass
class FunctionInfo:
    """Information about a function or method."""
    name: str
    signature: str
    docstring: Optional[str]
    is_public: bool
    is_entry_point: bool
    line_number: int


@dataclass
class ClassInfo:
    """Information about a class definition."""
    name: str
    docstring: Optional[str]
    methods: List[FunctionInfo]
    is_public: bool
    line_number: int


@dataclass
class ModuleInfo:
    """Basic information about a Python module."""
    name: str
    path: Path
    relative_path: str
    module_dotted_name: str


@dataclass
class ModuleAnalysis:
    """Complete analysis results for a Python module."""
    module_info: ModuleInfo
    imports: List[ImportInfo]
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    docstring: Optional[str]
    role: str
    entry_points: List[str]
    input_patterns: List[str]
    output_patterns: List[str]
    artifacts: List[str]
    risks: List[str]
    overlaps: List[str]


class ModuleDiscovery:
    """Scans directories and identifies Python modules for analysis."""
    
    def __init__(self, base_path: Path):
        """Initialize the module discovery component.
        
        Args:
            base_path: Base path for the decomposition modules directory.
        """
        self.base_path = base_path
        logger.info(f"Initialized ModuleDiscovery with base path: {self.base_path}")
    
    def discover_modules(self, target_path: Path = None) -> List[ModuleInfo]:
        """Discover all Python modules in the target directory.
        
        Args:
            target_path: Directory to scan for modules. Defaults to base_path.
            
        Returns:
            List of ModuleInfo objects for discovered modules.
        """
        if target_path is None:
            target_path = self.base_path
            
        logger.info(f"Starting module discovery in: {target_path}")
        
        if not target_path.exists():
            logger.warning(f"Target path does not exist: {target_path}")
            return []
            
        if not target_path.is_dir():
            logger.warning(f"Target path is not a directory: {target_path}")
            return []
        
        modules = []
        
        try:
            # Recursively scan for Python files
            for py_file in target_path.rglob("*.py"):
                try:
                    # Skip __init__.py files as specified in requirements
                    if py_file.name == "__init__.py":
                        logger.debug(f"Skipping __init__.py file: {py_file}")
                        continue
                    
                    # Extract module metadata
                    module_info = self._extract_module_metadata(py_file, target_path)
                    if module_info:
                        modules.append(module_info)
                        logger.debug(f"Discovered module: {module_info.name}")
                        
                except (OSError, PermissionError) as e:
                    logger.warning(f"Cannot access file {py_file}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error processing file {py_file}: {e}")
                    continue
                    
        except (OSError, PermissionError) as e:
            logger.error(f"Cannot access directory {target_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during directory scan: {e}")
            return []
        
        logger.info(f"Module discovery completed. Found {len(modules)} modules.")
        return modules
    
    def filter_modules(self, modules: List[ModuleInfo]) -> List[ModuleInfo]:
        """Filter modules based on specific criteria.
        
        Args:
            modules: List of ModuleInfo objects to filter.
            
        Returns:
            Filtered list of ModuleInfo objects.
        """
        logger.info(f"Filtering {len(modules)} discovered modules")
        
        filtered_modules = []
        for module in modules:
            try:
                # Basic filtering - ensure file exists and is readable
                if not module.path.exists():
                    logger.warning(f"Module file does not exist: {module.path}")
                    continue
                    
                if not module.path.is_file():
                    logger.warning(f"Module path is not a file: {module.path}")
                    continue
                
                # Check if file is readable
                try:
                    with open(module.path, 'r', encoding='utf-8') as f:
                        # Just try to read first line to verify accessibility
                        f.readline()
                    filtered_modules.append(module)
                    
                except (OSError, PermissionError, UnicodeDecodeError) as e:
                    logger.warning(f"Cannot read module file {module.path}: {e}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error filtering module {module.name}: {e}")
                continue
        
        logger.info(f"Module filtering completed. {len(filtered_modules)} modules passed filtering.")
        return filtered_modules
    
    def _extract_module_metadata(self, py_file: Path, base_path: Path) -> Optional[ModuleInfo]:
        """Extract metadata from a Python file.
        
        Args:
            py_file: Path to the Python file.
            base_path: Base directory path for calculating relative paths.
            
        Returns:
            ModuleInfo object or None if extraction fails.
        """
        try:
            # Calculate relative path from base directory
            relative_path = str(py_file.relative_to(base_path))
            
            # Extract module name (filename without .py extension)
            module_name = py_file.stem
            
            # Calculate dotted module name
            # Convert path separators to dots and remove .py extension
            path_parts = py_file.relative_to(base_path).parts[:-1]  # Remove filename
            path_parts = path_parts + (module_name,)  # Add module name
            module_dotted_name = ".".join(path_parts)
            
            return ModuleInfo(
                name=module_name,
                path=py_file,
                relative_path=relative_path,
                module_dotted_name=module_dotted_name
            )
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from {py_file}: {e}")
            return None


class CodeAnalyzer:
    """Performs deep static analysis of Python modules using AST parsing."""
    
    def __init__(self):
        """Initialize the code analyzer component."""
        logger.info("Initialized CodeAnalyzer component")
    
    def analyze_module(self, module_info: ModuleInfo) -> ModuleAnalysis:
        """Analyze a Python module and extract comprehensive information.
        
        Args:
            module_info: ModuleInfo object containing basic module metadata.
            
        Returns:
            ModuleAnalysis object with complete analysis results.
        """
        logger.info(f"Starting analysis of module: {module_info.name}")
        
        try:
            # Parse the module file into AST
            ast_tree = self._parse_file_to_ast(module_info.path)
            if ast_tree is None:
                # Fall back to basic analysis if AST parsing fails
                return self._create_fallback_analysis(module_info)
            
            # Extract various components from AST
            imports = self.extract_imports(ast_tree)
            functions = self.extract_functions(ast_tree)
            classes = self.extract_classes(ast_tree)
            docstring = self._extract_module_docstring(ast_tree)
            
            # Determine module role and characteristics
            role = self.determine_role(imports, functions, classes)
            entry_points = self._identify_entry_points(functions, classes)
            input_patterns = self._analyze_input_patterns(functions, classes)
            output_patterns = self._analyze_output_patterns(functions, classes)
            artifacts = self._identify_artifacts(functions, classes)
            risks = self._assess_risks(functions, classes, imports)
            overlaps = []  # Will be populated by cross-module analysis later
            
            analysis = ModuleAnalysis(
                module_info=module_info,
                imports=imports,
                functions=functions,
                classes=classes,
                docstring=docstring,
                role=role,
                entry_points=entry_points,
                input_patterns=input_patterns,
                output_patterns=output_patterns,
                artifacts=artifacts,
                risks=risks,
                overlaps=overlaps
            )
            
            logger.info(f"Successfully analyzed module: {module_info.name}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze module {module_info.name}: {e}")
            return self._create_fallback_analysis(module_info)
    
    def _parse_file_to_ast(self, file_path: Path) -> Optional[ast.Module]:
        """Parse a Python file into an AST tree with encoding fallback strategies.
        
        Args:
            file_path: Path to the Python file to parse.
            
        Returns:
            AST Module object or None if parsing fails.
        """
        # Try multiple encodings as specified in requirements
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                logger.debug(f"Attempting to parse {file_path} with encoding: {encoding}")
                
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                
                # Parse content into AST
                ast_tree = ast.parse(content, filename=str(file_path))
                logger.debug(f"Successfully parsed {file_path} with encoding: {encoding}")
                return ast_tree
                
            except UnicodeDecodeError:
                logger.debug(f"Encoding {encoding} failed for {file_path}, trying next")
                continue
            except SyntaxError as e:
                logger.warning(f"Syntax error in {file_path}: {e}")
                return None
            except Exception as e:
                logger.warning(f"Unexpected error parsing {file_path} with {encoding}: {e}")
                continue
        
        logger.error(f"Failed to parse {file_path} with any encoding")
        return None
    
    def extract_imports(self, ast_tree: ast.Module) -> List[ImportInfo]:
        """Extract import information from an AST tree.
        
        Args:
            ast_tree: AST Module object to analyze.
            
        Returns:
            List of ImportInfo objects representing all imports.
        """
        imports = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        names=[alias.asname or alias.name],
                        is_from_import=False,
                        level=0
                    ))
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                names = [alias.name for alias in node.names]
                imports.append(ImportInfo(
                    module=module_name,
                    names=names,
                    is_from_import=True,
                    level=node.level
                ))
        
        logger.debug(f"Extracted {len(imports)} imports from AST")
        return imports
    
    def extract_functions(self, ast_tree: ast.Module) -> List[FunctionInfo]:
        """Extract function information from an AST tree.
        
        Args:
            ast_tree: AST Module object to analyze.
            
        Returns:
            List of FunctionInfo objects representing all functions.
        """
        functions = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                # Generate function signature
                signature = self._generate_function_signature(node)
                
                # Extract docstring
                docstring = ast.get_docstring(node)
                
                # Determine if function is public (doesn't start with _)
                is_public = not node.name.startswith('_')
                
                # Determine if this could be an entry point
                is_entry_point = self._is_potential_entry_point(node)
                
                functions.append(FunctionInfo(
                    name=node.name,
                    signature=signature,
                    docstring=docstring,
                    is_public=is_public,
                    is_entry_point=is_entry_point,
                    line_number=node.lineno
                ))
        
        logger.debug(f"Extracted {len(functions)} functions from AST")
        return functions
    
    def extract_classes(self, ast_tree: ast.Module) -> List[ClassInfo]:
        """Extract class information from an AST tree.
        
        Args:
            ast_tree: AST Module object to analyze.
            
        Returns:
            List of ClassInfo objects representing all classes.
        """
        classes = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.ClassDef):
                # Extract class docstring
                docstring = ast.get_docstring(node)
                
                # Extract methods
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        signature = self._generate_function_signature(item)
                        method_docstring = ast.get_docstring(item)
                        is_public = not item.name.startswith('_')
                        is_entry_point = self._is_potential_entry_point(item)
                        
                        methods.append(FunctionInfo(
                            name=item.name,
                            signature=signature,
                            docstring=method_docstring,
                            is_public=is_public,
                            is_entry_point=is_entry_point,
                            line_number=item.lineno
                        ))
                
                # Determine if class is public
                is_public = not node.name.startswith('_')
                
                classes.append(ClassInfo(
                    name=node.name,
                    docstring=docstring,
                    methods=methods,
                    is_public=is_public,
                    line_number=node.lineno
                ))
        
        logger.debug(f"Extracted {len(classes)} classes from AST")
        return classes
    
    def _generate_function_signature(self, func_node: ast.FunctionDef) -> str:
        """Generate a string representation of a function signature.
        
        Args:
            func_node: AST FunctionDef node.
            
        Returns:
            String representation of the function signature.
        """
        try:
            args = []
            
            # Regular arguments
            for arg in func_node.args.args:
                args.append(arg.arg)
            
            # *args
            if func_node.args.vararg:
                args.append(f"*{func_node.args.vararg.arg}")
            
            # **kwargs
            if func_node.args.kwarg:
                args.append(f"**{func_node.args.kwarg.arg}")
            
            signature = f"{func_node.name}({', '.join(args)})"
            return signature
            
        except Exception as e:
            logger.warning(f"Failed to generate signature for function {func_node.name}: {e}")
            return f"{func_node.name}(...)"
    
    def _extract_module_docstring(self, ast_tree: ast.Module) -> Optional[str]:
        """Extract the module-level docstring from an AST tree.
        
        Args:
            ast_tree: AST Module object to analyze.
            
        Returns:
            Module docstring or None if not found.
        """
        return ast.get_docstring(ast_tree)
    
    def _is_potential_entry_point(self, func_node: ast.FunctionDef) -> bool:
        """Determine if a function could be an entry point.
        
        Args:
            func_node: AST FunctionDef node to analyze.
            
        Returns:
            True if the function appears to be an entry point.
        """
        # Check for common entry point patterns
        entry_point_names = {
            'main', 'run', 'execute', 'start', 'process', 'analyze', 
            'generate', 'create', 'build', 'parse', 'extract'
        }
        
        # Check if function name suggests it's an entry point
        if func_node.name.lower() in entry_point_names:
            return True
        
        # Check for CLI-related decorators (simplified check)
        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id in ['click.command', 'command', 'cli']:
                    return True
        
        # Check if it's a public function with no parameters (potential main function)
        if not func_node.name.startswith('_') and len(func_node.args.args) == 0:
            return True
        
        return False
    
    def _create_fallback_analysis(self, module_info: ModuleInfo) -> ModuleAnalysis:
        """Create a minimal analysis when AST parsing fails.
        
        Args:
            module_info: ModuleInfo object for the module.
            
        Returns:
            ModuleAnalysis with minimal information.
        """
        logger.warning(f"Creating fallback analysis for {module_info.name}")
        
        return ModuleAnalysis(
            module_info=module_info,
            imports=[],
            functions=[],
            classes=[],
            docstring=None,
            role="Unknown (AST parsing failed)",
            entry_points=[],
            input_patterns=[],
            output_patterns=[],
            artifacts=[],
            risks=["AST parsing failed - analysis incomplete"],
            overlaps=[]
        )

    def determine_role(self, imports: List[ImportInfo], functions: List[FunctionInfo], classes: List[ClassInfo]) -> str:
        """Determine if the module is an analyzer (read-only) or executor (modifies files).
        
        Args:
            imports: List of module imports.
            functions: List of functions in the module.
            classes: List of classes in the module.
            
        Returns:
            String describing the module's role.
        """
        # Check for file modification patterns in imports
        file_modification_imports = {
            'os', 'shutil', 'pathlib', 'tempfile', 'io', 'json', 'yaml', 'pickle',
            'csv', 'sqlite3', 'subprocess', 'zipfile', 'tarfile'
        }
        
        # Check for analysis-only imports
        analysis_imports = {
            'ast', 'inspect', 'dis', 'tokenize', 'parser', 'symbol', 'keyword'
        }
        
        # Count file modification vs analysis imports
        modification_score = 0
        analysis_score = 0
        
        for import_info in imports:
            if import_info.module in file_modification_imports:
                modification_score += 1
            if import_info.module in analysis_imports:
                analysis_score += 1
        
        # Check function names for modification patterns
        modification_keywords = {
            'write', 'save', 'create', 'delete', 'remove', 'modify', 'update',
            'generate', 'build', 'make', 'copy', 'move', 'rename', 'mkdir'
        }
        
        analysis_keywords = {
            'analyze', 'parse', 'extract', 'read', 'scan', 'detect', 'find',
            'search', 'match', 'check', 'validate', 'inspect', 'examine'
        }
        
        for func in functions:
            func_name_lower = func.name.lower()
            for keyword in modification_keywords:
                if keyword in func_name_lower:
                    modification_score += 2
                    break
            for keyword in analysis_keywords:
                if keyword in func_name_lower:
                    analysis_score += 1
                    break
        
        # Determine role based on scores
        if modification_score > analysis_score:
            return "Executor (modifies files/data)"
        elif analysis_score > modification_score:
            return "Analyzer (read-only analysis)"
        else:
            return "Mixed (both analysis and modification)"
    
    def _identify_entry_points(self, functions: List[FunctionInfo], classes: List[ClassInfo]) -> List[str]:
        """Identify entry points in the module.
        
        Args:
            functions: List of functions in the module.
            classes: List of classes in the module.
            
        Returns:
            List of entry point descriptions.
        """
        entry_points = []
        
        # Check for main function
        main_functions = [f for f in functions if f.name == 'main' and f.is_public]
        if main_functions:
            entry_points.append(f"Main function: {main_functions[0].signature}")
        
        # Check for other potential entry point functions
        for func in functions:
            if func.is_entry_point and func.name != 'main':
                entry_points.append(f"Function: {func.signature}")
        
        # Check for public classes that could be APIs
        for cls in classes:
            if cls.is_public:
                entry_points.append(f"Class: {cls.name}")
        
        return entry_points
    
    def _analyze_input_patterns(self, functions: List[FunctionInfo], classes: List[ClassInfo]) -> List[str]:
        """Analyze what types of input the module expects.
        
        Args:
            functions: List of functions in the module.
            classes: List of classes in the module.
            
        Returns:
            List of input pattern descriptions.
        """
        patterns = []
        
        # Look for common input patterns in function signatures
        for func in functions:
            if 'path' in func.signature.lower() or 'file' in func.signature.lower():
                patterns.append("File paths")
            if 'config' in func.signature.lower():
                patterns.append("Configuration objects")
            if 'data' in func.signature.lower():
                patterns.append("Data structures")
            if 'ast' in func.signature.lower():
                patterns.append("AST objects")
        
        # Remove duplicates
        return list(set(patterns))
    
    def _analyze_output_patterns(self, functions: List[FunctionInfo], classes: List[ClassInfo]) -> List[str]:
        """Analyze what types of output the module produces.
        
        Args:
            functions: List of functions in the module.
            classes: List of classes in the module.
            
        Returns:
            List of output pattern descriptions.
        """
        patterns = []
        
        # Look for return type hints and function names
        for func in functions:
            if 'generate' in func.name.lower() or 'create' in func.name.lower():
                patterns.append("Generated objects/data")
            if 'analyze' in func.name.lower() or 'extract' in func.name.lower():
                patterns.append("Analysis results")
            if 'report' in func.name.lower():
                patterns.append("Reports")
        
        # Check for classes that represent output
        for cls in classes:
            if 'result' in cls.name.lower() or 'report' in cls.name.lower():
                patterns.append(f"{cls.name} objects")
        
        return list(set(patterns))
    
    def _identify_artifacts(self, functions: List[FunctionInfo], classes: List[ClassInfo]) -> List[str]:
        """Identify files or artifacts that the module creates.
        
        Args:
            functions: List of functions in the module.
            classes: List of classes in the module.
            
        Returns:
            List of artifact descriptions.
        """
        artifacts = []
        
        # Look for file creation patterns
        file_extensions = ['.json', '.md', '.txt', '.csv', '.yaml', '.xml', '.html']
        
        for func in functions:
            func_name_lower = func.name.lower()
            if any(ext in func_name_lower for ext in file_extensions):
                artifacts.append(f"Files with extension mentioned in {func.name}")
            if 'write' in func_name_lower or 'save' in func_name_lower:
                artifacts.append("Output files")
            if 'report' in func_name_lower:
                artifacts.append("Report files")
        
        return list(set(artifacts))
    
    def _assess_risks(self, functions: List[FunctionInfo], classes: List[ClassInfo], imports: List[ImportInfo]) -> List[str]:
        """Assess potential risks and failure points in the module.
        
        Args:
            functions: List of functions in the module.
            classes: List of classes in the module.
            imports: List of imports in the module.
            
        Returns:
            List of risk descriptions.
        """
        risks = []
        
        # Check for risky imports
        risky_imports = {'subprocess', 'os', 'shutil', 'tempfile'}
        for import_info in imports:
            if import_info.module in risky_imports:
                risks.append(f"Uses potentially risky module: {import_info.module}")
        
        # Check for file operations
        file_operation_functions = [f for f in functions if any(
            keyword in f.name.lower() 
            for keyword in ['write', 'delete', 'remove', 'create', 'modify']
        )]
        if file_operation_functions:
            risks.append("Performs file system modifications")
        
        # Check for external dependencies
        external_imports = [imp for imp in imports if not imp.module.startswith('.') and imp.module not in {
            'os', 'sys', 'pathlib', 'typing', 'dataclasses', 'logging', 'ast', 'json'
        }]
        if len(external_imports) > 5:
            risks.append("High number of external dependencies")
        
        return risks


class TemplateEngine:
    """Populates the Russian documentation template with extracted information."""
    
    def __init__(self):
        """Initialize the template engine component."""
        logger.info("Initialized TemplateEngine component")
    
    def populate_template(self, analysis: ModuleAnalysis) -> str:
        """Populate the Russian documentation template with module analysis data.
        
        Args:
            analysis: ModuleAnalysis object containing extracted information.
            
        Returns:
            Formatted documentation string in Russian template format.
        """
        logger.info(f"Populating template for module: {analysis.module_info.name}")
        
        try:
            # Build the documentation sections
            header = self._format_header_section(analysis.module_info.name)
            role = self._format_role_section(analysis.role)
            functions = self._format_functions_section(analysis.functions, analysis.classes)
            implemented = self._format_implemented_section(analysis)
            missing = self._format_missing_section(analysis)
            risks = self._format_risks_section(analysis.risks)
            overlaps = self._format_overlaps_section(analysis.overlaps)
            improvements = self._format_improvements_section(analysis)
            
            # Combine all sections
            template = f"{header}\n\n{role}\n{functions}\n\n{implemented}\n\n{missing}\n\n{risks}\n\n{overlaps}\n\n{improvements}\n\n---\n"
            
            logger.debug(f"Successfully populated template for module: {analysis.module_info.name}")
            return template
            
        except Exception as e:
            logger.error(f"Failed to populate template for module {analysis.module_info.name}: {e}")
            return self._create_fallback_template(analysis.module_info.name)
    
    def _format_header_section(self, module_name: str) -> str:
        """Format the module header section.
        
        Args:
            module_name: Name of the module.
            
        Returns:
            Formatted header string.
        """
        return f"## {module_name}.py"
    
    def _format_role_section(self, role: str) -> str:
        """Format the role section of the template.
        
        Args:
            role: Module role description.
            
        Returns:
            Formatted role section string.
        """
        if not role or role.strip() == "":
            role = "Не определена (анализ не завершен)"
        
        return f"**Роль:** {role}"
    
    def _format_functions_section(self, functions: List[FunctionInfo], classes: List[ClassInfo]) -> str:
        """Format the key functions section of the template.
        
        Args:
            functions: List of function information.
            classes: List of class information.
            
        Returns:
            Formatted functions section string.
        """
        key_functions = []
        
        # Add public functions
        for func in functions:
            if func.is_public:
                description = func.docstring or "Описание отсутствует"
                # Truncate long descriptions
                if len(description) > 100:
                    description = description[:97] + "..."
                key_functions.append(f"- {func.name}: {description}")
        
        # Add public classes and their key methods
        for cls in classes:
            if cls.is_public:
                class_description = cls.docstring or "Описание отсутствует"
                if len(class_description) > 100:
                    class_description = class_description[:97] + "..."
                key_functions.append(f"- Класс {cls.name}: {class_description}")
                
                # Add key public methods
                for method in cls.methods[:3]:  # Limit to first 3 methods
                    if method.is_public and not method.name.startswith('__'):
                        method_desc = method.docstring or "Описание отсутствует"
                        if len(method_desc) > 80:
                            method_desc = method_desc[:77] + "..."
                        key_functions.append(f"  - {method.name}: {method_desc}")
        
        if not key_functions:
            key_functions = ["- Публичные функции не обнаружены"]
        
        functions_text = "\n".join(key_functions)
        return f"**Ключевые функции:**\n{functions_text}"
    
    def _format_implemented_section(self, analysis: ModuleAnalysis) -> str:
        """Format the implemented features section.
        
        Args:
            analysis: ModuleAnalysis object.
            
        Returns:
            Formatted implemented section string.
        """
        implemented_features = []
        
        # Check for implemented features based on analysis
        if analysis.functions:
            implemented_features.append("Основная функциональность модуля")
        
        if analysis.classes:
            implemented_features.append("Объектно-ориентированная структура")
        
        if analysis.imports:
            implemented_features.append("Интеграция с внешними модулями")
        
        if analysis.entry_points:
            implemented_features.append("Точки входа для использования")
        
        if analysis.artifacts:
            implemented_features.append("Генерация выходных артефактов")
        
        # Add specific patterns based on role
        if "Analyzer" in analysis.role:
            implemented_features.append("Анализ и извлечение данных")
        elif "Executor" in analysis.role:
            implemented_features.append("Модификация файлов и данных")
        
        if not implemented_features:
            implemented_features = ["Базовая структура модуля"]
        
        implemented_text = "\n".join(f"- {feature}" for feature in implemented_features)
        return f"**Есть (реально реализовано):**\n{implemented_text}"
    
    def _format_missing_section(self, analysis: ModuleAnalysis) -> str:
        """Format the missing/unstable features section.
        
        Args:
            analysis: ModuleAnalysis object.
            
        Returns:
            Formatted missing section string.
        """
        missing_features = []
        
        # Check for common missing features
        if not analysis.docstring:
            missing_features.append("Документация модуля (отсутствует docstring)")
        
        # Check for functions without docstrings
        undocumented_functions = [f for f in analysis.functions if f.is_public and not f.docstring]
        if undocumented_functions:
            missing_features.append(f"Документация для {len(undocumented_functions)} публичных функций")
        
        # Check for classes without docstrings
        undocumented_classes = [c for c in analysis.classes if c.is_public and not c.docstring]
        if undocumented_classes:
            missing_features.append(f"Документация для {len(undocumented_classes)} публичных классов")
        
        # Check for error handling
        if not any("error" in func.name.lower() or "exception" in func.name.lower() 
                  for func in analysis.functions):
            missing_features.append("Обработка ошибок (не обнаружена)")
        
        # Check for testing
        if not any("test" in func.name.lower() for func in analysis.functions):
            missing_features.append("Модульные тесты (не обнаружены)")
        
        if not missing_features:
            missing_features = ["Критических недостатков не обнаружено"]
        
        missing_text = "\n".join(f"- {feature}" for feature in missing_features)
        return f"**Нет / нестабильно / заглушка:**\n{missing_text}"
    
    def _format_risks_section(self, risks: List[str]) -> str:
        """Format the risks and dangers section.
        
        Args:
            risks: List of identified risks.
            
        Returns:
            Formatted risks section string.
        """
        if not risks:
            risks = ["Критических рисков не выявлено"]
        
        risks_text = "\n".join(f"- {risk}" for risk in risks)
        return f"**Риски/опасности:**\n{risks_text}"
    
    def _format_overlaps_section(self, overlaps: List[str]) -> str:
        """Format the overlaps and duplicates section.
        
        Args:
            overlaps: List of identified overlaps with other modules.
            
        Returns:
            Formatted overlaps section string.
        """
        if not overlaps:
            overlaps = ["Пересечения с другими модулями требуют дополнительного анализа"]
        
        overlaps_text = "\n".join(f"- {overlap}" for overlap in overlaps)
        return f"**Пересечения/дубли с другими модулями:**\n{overlaps_text}"
    
    def _format_improvements_section(self, analysis: ModuleAnalysis) -> str:
        """Format the improvements suggestions section.
        
        Args:
            analysis: ModuleAnalysis object.
            
        Returns:
            Formatted improvements section string.
        """
        improvements = []
        
        # Suggest improvements based on analysis
        if not analysis.docstring:
            improvements.append("Добавить документацию модуля")
        
        undocumented_functions = [f for f in analysis.functions if f.is_public and not f.docstring]
        if undocumented_functions:
            improvements.append("Документировать публичные функции")
        
        if not any("test" in func.name.lower() for func in analysis.functions):
            improvements.append("Добавить модульные тесты")
        
        if "Executor" in analysis.role and not any("backup" in risk.lower() for risk in analysis.risks):
            improvements.append("Добавить механизм резервного копирования")
        
        if len(analysis.functions) > 10:
            improvements.append("Рассмотреть разделение на более мелкие модули")
        
        if not improvements:
            improvements = ["Модуль в хорошем состоянии, критических улучшений не требуется"]
        
        improvements_text = "\n".join(f"- {improvement}" for improvement in improvements)
        return f"**Что улучшить в первую очередь:**\n{improvements_text}"
    
    def _create_fallback_template(self, module_name: str) -> str:
        """Create a minimal template when template population fails.
        
        Args:
            module_name: Name of the module.
            
        Returns:
            Minimal template string.
        """
        logger.warning(f"Creating fallback template for {module_name}")
        
        return f"""## {module_name}.py

**Роль:** Не удалось определить (ошибка анализа)
**Ключевые функции:**
- Анализ не завершен

**Есть (реально реализовано):**
- Базовая структура файла

**Нет / нестабильно / заглушка:**
- Полный анализ не выполнен

**Риски/опасности:**
- Анализ модуля завершился с ошибкой

**Пересечения/дубли с другими модулями:**
- Требуется повторный анализ

**Что улучшить в первую очередь:**
- Исправить ошибки анализа модуля

---
"""

    def detect_overlaps(self, all_analyses: List[ModuleAnalysis]) -> None:
        """Detect overlaps and duplicate functionality across modules.
        
        Args:
            all_analyses: List of all module analyses to compare.
        """
        logger.info(f"Detecting overlaps across {len(all_analyses)} modules")
        
        try:
            # Compare each module with others to find overlaps
            for i, analysis in enumerate(all_analyses):
                overlaps = []
                
                for j, other_analysis in enumerate(all_analyses):
                    if i == j:
                        continue
                    
                    # Check for similar function names
                    similar_functions = self._find_similar_functions(
                        analysis.functions, other_analysis.functions
                    )
                    if similar_functions:
                        overlaps.append(
                            f"Похожие функции с {other_analysis.module_info.name}: {', '.join(similar_functions)}"
                        )
                    
                    # Check for similar imports
                    common_imports = self._find_common_imports(
                        analysis.imports, other_analysis.imports
                    )
                    if len(common_imports) > 3:  # Threshold for significant overlap
                        overlaps.append(
                            f"Общие зависимости с {other_analysis.module_info.name}: {len(common_imports)} модулей"
                        )
                    
                    # Check for similar roles
                    if analysis.role == other_analysis.role and "Unknown" not in analysis.role:
                        overlaps.append(
                            f"Аналогичная роль с {other_analysis.module_info.name}: {analysis.role}"
                        )
                
                # Update the analysis with detected overlaps
                analysis.overlaps = overlaps
                
        except Exception as e:
            logger.error(f"Failed to detect overlaps: {e}")
    
    def _find_similar_functions(self, functions1: List[FunctionInfo], functions2: List[FunctionInfo]) -> List[str]:
        """Find similar function names between two function lists.
        
        Args:
            functions1: First list of functions.
            functions2: Second list of functions.
            
        Returns:
            List of similar function names.
        """
        similar = []
        
        for func1 in functions1:
            for func2 in functions2:
                # Check for exact matches or similar names
                if func1.name == func2.name:
                    similar.append(func1.name)
                elif self._are_names_similar(func1.name, func2.name):
                    similar.append(f"{func1.name}~{func2.name}")
        
        return similar[:5]  # Limit to first 5 matches
    
    def _find_common_imports(self, imports1: List[ImportInfo], imports2: List[ImportInfo]) -> List[str]:
        """Find common imports between two import lists.
        
        Args:
            imports1: First list of imports.
            imports2: Second list of imports.
            
        Returns:
            List of common import module names.
        """
        modules1 = {imp.module for imp in imports1}
        modules2 = {imp.module for imp in imports2}
        
        return list(modules1.intersection(modules2))
    
    def _are_names_similar(self, name1: str, name2: str) -> bool:
        """Check if two function names are similar.
        
        Args:
            name1: First function name.
            name2: Second function name.
            
        Returns:
            True if names are similar, False otherwise.
        """
        # Simple similarity check - could be enhanced with more sophisticated algorithms
        if len(name1) < 3 or len(name2) < 3:
            return False
        
        # Check for common prefixes/suffixes
        if name1.startswith(name2[:3]) or name2.startswith(name1[:3]):
            return True
        
        if name1.endswith(name2[-3:]) or name2.endswith(name1[-3:]):
            return True
        
        return False


class ExpertQuestionEvaluator:
    """Evaluates each module against the 10 universal expert questions."""
    
    def __init__(self):
        """Initialize the expert question evaluator component."""
        logger.info("Initialized ExpertQuestionEvaluator component")
    
    def evaluate_module(self, analysis: ModuleAnalysis) -> Dict[str, str]:
        """Evaluate a module against the 10 universal checklist questions.
        
        Args:
            analysis: ModuleAnalysis object to evaluate.
            
        Returns:
            Dictionary mapping question numbers to answers.
        """
        logger.info(f"Evaluating expert questions for module: {analysis.module_info.name}")
        
        try:
            answers = {}
            
            # Question 1: Analyzer vs Executor type
            answers["1"] = self.check_analyzer_vs_executor(analysis)
            
            # Question 2: Entry points
            answers["2"] = self.identify_entry_points(analysis)
            
            # Question 3: Artifacts
            answers["3"] = self.identify_artifacts(analysis)
            
            # Question 4: Dry-run support
            answers["4"] = self.check_dry_run_support(analysis)
            
            # Question 5: Output formats
            answers["5"] = self.check_output_formats(analysis)
            
            # Question 6: Security features
            answers["6"] = self.check_security_features(analysis)
            
            # Question 7: Filtering rules
            answers["7"] = self.check_filtering_rules(analysis)
            
            # Question 8: Duplications
            answers["8"] = self.find_duplications(analysis)
            
            # Question 9: Stability
            answers["9"] = self.assess_stability(analysis)
            
            # Question 10: Common failures
            answers["10"] = self.identify_common_failures(analysis)
            
            logger.debug(f"Successfully evaluated expert questions for module: {analysis.module_info.name}")
            return answers
            
        except Exception as e:
            logger.error(f"Failed to evaluate expert questions for module {analysis.module_info.name}: {e}")
            return self._create_fallback_answers()
    
    def check_analyzer_vs_executor(self, analysis: ModuleAnalysis) -> str:
        """Check if the module is an analyzer or executor type.
        
        Args:
            analysis: ModuleAnalysis object.
            
        Returns:
            String describing the module type.
        """
        return f"Тип: {analysis.role}"
    
    def identify_entry_points(self, analysis: ModuleAnalysis) -> str:
        """Identify entry points in the module.
        
        Args:
            analysis: ModuleAnalysis object.
            
        Returns:
            String describing entry points.
        """
        if analysis.entry_points:
            return f"Точки входа: {', '.join(analysis.entry_points[:3])}"
        else:
            return "Точки входа: Не обнаружены явные точки входа"
    
    def identify_artifacts(self, analysis: ModuleAnalysis) -> str:
        """Identify artifacts produced by the module.
        
        Args:
            analysis: ModuleAnalysis object.
            
        Returns:
            String describing artifacts.
        """
        if analysis.artifacts:
            return f"Артефакты: {', '.join(analysis.artifacts[:3])}"
        else:
            return "Артефакты: Не создает файловых артефактов"
    
    def check_dry_run_support(self, analysis: ModuleAnalysis) -> str:
        """Check if the module supports dry-run mode.
        
        Args:
            analysis: ModuleAnalysis object.
            
        Returns:
            String describing dry-run support.
        """
        # Look for dry-run related patterns
        dry_run_indicators = ['dry_run', 'simulate', 'preview', 'test_mode']
        
        for func in analysis.functions:
            if any(indicator in func.name.lower() for indicator in dry_run_indicators):
                return "Dry-run: Поддерживается (обнаружены соответствующие функции)"
            
            if func.signature and any(indicator in func.signature.lower() for indicator in dry_run_indicators):
                return "Dry-run: Поддерживается (параметры в сигнатурах функций)"
        
        return "Dry-run: Не обнаружено явной поддержки"
    
    def check_output_formats(self, analysis: ModuleAnalysis) -> str:
        """Check output formats supported by the module.
        
        Args:
            analysis: ModuleAnalysis object.
            
        Returns:
            String describing output formats.
        """
        formats = []
        
        # Check imports for format indicators
        format_imports = {'json': 'JSON', 'yaml': 'YAML', 'csv': 'CSV', 'xml': 'XML'}
        for imp in analysis.imports:
            if imp.module in format_imports:
                formats.append(format_imports[imp.module])
        
        # Check function names for format indicators
        for func in analysis.functions:
            if 'json' in func.name.lower():
                formats.append('JSON')
            elif 'yaml' in func.name.lower():
                formats.append('YAML')
            elif 'csv' in func.name.lower():
                formats.append('CSV')
        
        if formats:
            return f"Форматы вывода: {', '.join(set(formats))}"
        else:
            return "Форматы вывода: Не определены"
    
    def check_security_features(self, analysis: ModuleAnalysis) -> str:
        """Check security and backup features.
        
        Args:
            analysis: ModuleAnalysis object.
            
        Returns:
            String describing security features.
        """
        security_features = []
        
        # Check for backup-related functions
        backup_keywords = ['backup', 'restore', 'rollback', 'undo']
        for func in analysis.functions:
            if any(keyword in func.name.lower() for keyword in backup_keywords):
                security_features.append("Резервное копирование")
                break
        
        # Check for validation functions
        validation_keywords = ['validate', 'verify', 'check']
        for func in analysis.functions:
            if any(keyword in func.name.lower() for keyword in validation_keywords):
                security_features.append("Валидация данных")
                break
        
        # Check for error handling
        if any('error' in func.name.lower() or 'exception' in func.name.lower() 
              for func in analysis.functions):
            security_features.append("Обработка ошибок")
        
        if security_features:
            return f"Безопасность: {', '.join(security_features)}"
        else:
            return "Безопасность: Механизмы безопасности не обнаружены"
    
    def check_filtering_rules(self, analysis: ModuleAnalysis) -> str:
        """Check filtering and exclusion rules.
        
        Args:
            analysis: ModuleAnalysis object.
            
        Returns:
            String describing filtering rules.
        """
        filter_indicators = ['filter', 'exclude', 'include', 'ignore', 'skip']
        
        for func in analysis.functions:
            if any(indicator in func.name.lower() for indicator in filter_indicators):
                return "Фильтрация: Поддерживается (обнаружены функции фильтрации)"
        
        return "Фильтрация: Не обнаружено явных правил фильтрации"
    
    def find_duplications(self, analysis: ModuleAnalysis) -> str:
        """Find potential duplications with other components.
        
        Args:
            analysis: ModuleAnalysis object.
            
        Returns:
            String describing duplications.
        """
        if analysis.overlaps:
            return f"Дублирование: {len(analysis.overlaps)} потенциальных пересечений обнаружено"
        else:
            return "Дублирование: Требуется межмодульный анализ"
    
    def assess_stability(self, analysis: ModuleAnalysis) -> str:
        """Assess module stability.
        
        Args:
            analysis: ModuleAnalysis object.
            
        Returns:
            String describing stability assessment.
        """
        stability_score = 0
        
        # Positive indicators
        if analysis.docstring:
            stability_score += 1
        
        documented_functions = sum(1 for f in analysis.functions if f.docstring)
        if documented_functions > len(analysis.functions) * 0.5:
            stability_score += 1
        
        if analysis.entry_points:
            stability_score += 1
        
        # Negative indicators
        if analysis.risks:
            stability_score -= len(analysis.risks)
        
        if stability_score >= 2:
            return "Стабильность: Высокая (хорошо документирован и структурирован)"
        elif stability_score >= 0:
            return "Стабильность: Средняя (есть области для улучшения)"
        else:
            return "Стабильность: Низкая (множественные риски и проблемы)"
    
    def identify_common_failures(self, analysis: ModuleAnalysis) -> str:
        """Identify common failure causes.
        
        Args:
            analysis: ModuleAnalysis object.
            
        Returns:
            String describing common failures.
        """
        failure_causes = []
        
        # Check for risky imports
        risky_imports = {'subprocess', 'os', 'shutil'}
        for imp in analysis.imports:
            if imp.module in risky_imports:
                failure_causes.append("Системные операции")
                break
        
        # Check for file operations
        if any('file' in func.name.lower() or 'write' in func.name.lower() 
              for func in analysis.functions):
            failure_causes.append("Файловые операции")
        
        # Check for network operations
        network_imports = {'requests', 'urllib', 'http', 'socket'}
        for imp in analysis.imports:
            if imp.module in network_imports:
                failure_causes.append("Сетевые операции")
                break
        
        # Check for external dependencies
        if len(analysis.imports) > 10:
            failure_causes.append("Множественные зависимости")
        
        if failure_causes:
            return f"Частые сбои: {', '.join(failure_causes)}"
        else:
            return "Частые сбои: Низкий риск сбоев"
    
    def _create_fallback_answers(self) -> Dict[str, str]:
        """Create fallback answers when evaluation fails.
        
        Returns:
            Dictionary with fallback answers.
        """
        return {
            "1": "Тип: Не удалось определить",
            "2": "Точки входа: Анализ не завершен",
            "3": "Артефакты: Анализ не завершен",
            "4": "Dry-run: Анализ не завершен",
            "5": "Форматы вывода: Анализ не завершен",
            "6": "Безопасность: Анализ не завершен",
            "7": "Фильтрация: Анализ не завершен",
            "8": "Дублирование: Анализ не завершен",
            "9": "Стабильность: Анализ не завершен",
            "10": "Частые сбои: Анализ не завершен"
        }


class DocumentGenerator:
    """Assembles the final documentation file with proper formatting."""
    
    def __init__(self, output_path: Path):
        """Initialize the document generator.
        
        Args:
            output_path: Path where the documentation will be written.
        """
        self.output_path = output_path
        logger.info(f"Initialized DocumentGenerator with output path: {self.output_path}")
    
    def generate_document(self, module_docs: List[str], expert_questions: str = None) -> str:
        """Generate the complete documentation document.
        
        Args:
            module_docs: List of individual module documentation strings.
            expert_questions: Optional expert questions section.
            
        Returns:
            Complete documentation string.
        """
        logger.info(f"Generating document with {len(module_docs)} module documentations")
        
        try:
            # Create document header
            header = self._create_document_header()
            
            # Combine all module documentation
            modules_content = "\n".join(module_docs)
            
            # Add expert questions if provided
            expert_section = ""
            if expert_questions:
                expert_section = f"\n\n{expert_questions}"
            
            # Combine all sections
            complete_document = f"{header}\n\n{modules_content}{expert_section}"
            
            logger.info("Successfully generated complete documentation")
            return complete_document
            
        except Exception as e:
            logger.error(f"Failed to generate document: {e}")
            return self._create_fallback_document(module_docs)
    
    def write_to_file(self, content: str) -> bool:
        """Write documentation content to file.
        
        Args:
            content: Documentation content to write.
            
        Returns:
            True if successful, False otherwise.
        """
        logger.info(f"Writing documentation to file: {self.output_path}")
        
        try:
            # Create output directory if it doesn't exist
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content with UTF-8 encoding
            with open(self.output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Successfully wrote documentation to: {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write documentation to file: {e}")
            return False
    
    def _create_document_header(self) -> str:
        """Create the document header section.
        
        Returns:
            Formatted document header string.
        """
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""# Документация модулей декомпозиции IntelliRefactor

Автоматически сгенерированная документация для модулей в директории `intellirefactor/analysis/decomposition`.

**Дата генерации:** {timestamp}
**Система:** Decomposition Documentation Generator

## Обзор модулей

Данная документация содержит подробное описание каждого модуля системы декомпозиции, включая:
- Роль и назначение модуля
- Ключевые функции и возможности  
- Реализованные и отсутствующие функции
- Потенциальные риски и проблемы
- Пересечения с другими модулями
- Рекомендации по улучшению

---"""
    
    def _create_fallback_document(self, module_docs: List[str]) -> str:
        """Create a fallback document when generation fails.
        
        Args:
            module_docs: List of module documentation strings.
            
        Returns:
            Fallback document string.
        """
        logger.warning("Creating fallback document due to generation error")
        
        modules_content = "\n".join(module_docs) if module_docs else "Документация модулей недоступна"
        
        return f"""# Документация модулей декомпозиции IntelliRefactor

**ВНИМАНИЕ:** Произошла ошибка при генерации документации.

## Модули

{modules_content}

---

**Примечание:** Для получения полной документации повторите генерацию."""


class DocumentationGenerator:
    """Main orchestrator for the documentation generation system."""
    
    def __init__(self, base_path: Path = None):
        """Initialize the documentation generator.
        
        Args:
            base_path: Base path for the IntelliRefactor project. 
                      Defaults to current directory.
        """
        self.base_path = base_path or Path.cwd()
        self.decomposition_path = self.base_path / "intellirefactor" / "analysis" / "decomposition"
        self.output_path = self.base_path / "docs" / "readme_decomp.md"
        
        # Initialize components
        self.module_discovery = ModuleDiscovery(self.decomposition_path)
        self.code_analyzer = CodeAnalyzer()
        self.template_engine = TemplateEngine()
        self.expert_evaluator = ExpertQuestionEvaluator()
        self.document_generator = DocumentGenerator(self.output_path)
        
        logger.info(f"Initialized DocumentationGenerator with base path: {self.base_path}")
        logger.info(f"Decomposition modules path: {self.decomposition_path}")
        logger.info(f"Output documentation path: {self.output_path}")
    
    def generate_documentation(self) -> bool:
        """Generate complete documentation for all decomposition modules.
        
        Returns:
            True if documentation was generated successfully, False otherwise.
        """
        try:
            logger.info("Starting documentation generation process")
            
            # Step 1: Module Discovery
            modules = self.module_discovery.discover_modules()
            filtered_modules = self.module_discovery.filter_modules(modules)
            
            logger.info(f"Discovered {len(filtered_modules)} modules for documentation")
            
            # Step 2: Code Analysis
            analyzed_modules = []
            for module_info in filtered_modules:
                analysis = self.code_analyzer.analyze_module(module_info)
                analyzed_modules.append(analysis)
            
            logger.info(f"Analyzed {len(analyzed_modules)} modules")
            
            # Step 3: Detect overlaps across modules
            self.template_engine.detect_overlaps(analyzed_modules)
            
            # Step 4: Template Population
            module_docs = []
            for analysis in analyzed_modules:
                doc = self.template_engine.populate_template(analysis)
                module_docs.append(doc)
            
            logger.info(f"Generated documentation for {len(module_docs)} modules")
            
            # Step 5: Expert Questions Evaluation
            expert_answers = {}
            for analysis in analyzed_modules:
                answers = self.expert_evaluator.evaluate_module(analysis)
                expert_answers[analysis.module_info.name] = answers
            
            # Step 6: Document Generation
            expert_questions_section = self._format_expert_questions_section(expert_answers)
            complete_document = self.document_generator.generate_document(module_docs, expert_questions_section)
            
            # Step 7: Write to file
            success = self.document_generator.write_to_file(complete_document)
            
            if success:
                logger.info("Documentation generation completed successfully")
                return True
            else:
                logger.error("Failed to write documentation to file")
                return False
            
        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            return False
    
    def _format_expert_questions_section(self, expert_answers: Dict[str, Dict[str, str]]) -> str:
        """Format the expert questions section for the document.
        
        Args:
            expert_answers: Dictionary mapping module names to their expert question answers.
            
        Returns:
            Formatted expert questions section string.
        """
        if not expert_answers:
            return ""
        
        section = "# Экспертные вопросы по модулям\n\n"
        section += "Результаты оценки каждого модуля по 10 универсальным вопросам:\n\n"
        
        questions = [
            "1. Тип модуля (анализатор/исполнитель)",
            "2. Точки входа (CLI/API/классы/функции)",
            "3. Создаваемые артефакты",
            "4. Поддержка dry-run режима",
            "5. Форматы вывода",
            "6. Функции безопасности",
            "7. Правила фильтрации",
            "8. Дублирование функциональности",
            "9. Оценка стабильности",
            "10. Частые причины сбоев"
        ]
        
        for module_name, answers in expert_answers.items():
            section += f"## {module_name}.py\n\n"
            
            for i, question in enumerate(questions, 1):
                answer = answers.get(str(i), "Не оценено")
                section += f"**{question}:** {answer}\n\n"
            
            section += "---\n\n"
        
        return section


if __name__ == "__main__":
    generator = DocumentationGenerator()
    success = generator.generate_documentation()
    exit(0 if success else 1)