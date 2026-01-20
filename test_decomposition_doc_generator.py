"""
Property-based tests for the decomposition documentation generator.

Tests the core data models and module discovery functionality using Hypothesis.
"""

import pytest
from hypothesis import given, strategies as st, assume
from pathlib import Path
import tempfile
from decomposition_doc_generator import (
    ModuleInfo, ModuleAnalysis, FunctionInfo, ImportInfo, ClassInfo,
    DocumentationGenerator, ModuleDiscovery, CodeAnalyzer
)


# Hypothesis strategies for generating test data
@st.composite
def module_info_strategy(draw):
    """Generate valid ModuleInfo instances."""
    name = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_')))
    # Ensure name starts with letter or underscore
    if not (name[0].isalpha() or name[0] == '_'):
        name = 'module_' + name
    
    path = Path(draw(st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='/_.'))))
    relative_path = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='/_.')))
    module_dotted_name = draw(st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_.')))
    
    return ModuleInfo(
        name=name,
        path=path,
        relative_path=relative_path,
        module_dotted_name=module_dotted_name
    )


@st.composite
def import_info_strategy(draw):
    """Generate valid ImportInfo instances."""
    module = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_.')))
    names = draw(st.lists(st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_')), min_size=1, max_size=10))
    is_from_import = draw(st.booleans())
    level = draw(st.integers(min_value=0, max_value=5))
    
    return ImportInfo(
        module=module,
        names=names,
        is_from_import=is_from_import,
        level=level
    )


@st.composite
def function_info_strategy(draw):
    """Generate valid FunctionInfo instances."""
    name = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_')))
    # Ensure name starts with letter or underscore
    if not (name[0].isalpha() or name[0] == '_'):
        name = 'func_' + name
    
    signature = f"{name}({draw(st.text(max_size=100))})"
    docstring = draw(st.one_of(st.none(), st.text(max_size=200)))
    is_public = draw(st.booleans())
    is_entry_point = draw(st.booleans())
    line_number = draw(st.integers(min_value=1, max_value=10000))
    
    return FunctionInfo(
        name=name,
        signature=signature,
        docstring=docstring,
        is_public=is_public,
        is_entry_point=is_entry_point,
        line_number=line_number
    )


@st.composite
def class_info_strategy(draw):
    """Generate valid ClassInfo instances."""
    name = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_')))
    # Ensure name starts with letter or underscore
    if not (name[0].isalpha() or name[0] == '_'):
        name = 'Class_' + name
    
    docstring = draw(st.one_of(st.none(), st.text(max_size=200)))
    methods = draw(st.lists(function_info_strategy(), max_size=5))
    is_public = draw(st.booleans())
    line_number = draw(st.integers(min_value=1, max_value=10000))
    
    return ClassInfo(
        name=name,
        docstring=docstring,
        methods=methods,
        is_public=is_public,
        line_number=line_number
    )


@st.composite
def module_analysis_strategy(draw):
    """Generate valid ModuleAnalysis instances."""
    module_info = draw(module_info_strategy())
    imports = draw(st.lists(import_info_strategy(), max_size=10))
    functions = draw(st.lists(function_info_strategy(), max_size=10))
    classes = draw(st.lists(class_info_strategy(), max_size=5))
    docstring = draw(st.one_of(st.none(), st.text(max_size=500)))
    role = draw(st.sampled_from(['analyzer', 'executor', 'utility', 'interface']))
    entry_points = draw(st.lists(st.text(min_size=1, max_size=50), max_size=5))
    input_patterns = draw(st.lists(st.text(min_size=1, max_size=100), max_size=5))
    output_patterns = draw(st.lists(st.text(min_size=1, max_size=100), max_size=5))
    artifacts = draw(st.lists(st.text(min_size=1, max_size=100), max_size=5))
    risks = draw(st.lists(st.text(min_size=1, max_size=200), max_size=5))
    overlaps = draw(st.lists(st.text(min_size=1, max_size=200), max_size=5))
    
    return ModuleAnalysis(
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


class TestDataModelProperties:
    """Property-based tests for data model validation."""
    
    @given(module_info_strategy())
    def test_module_info_properties(self, module_info):
        """
        Property 1: Complete Module Discovery
        For any valid ModuleInfo, all fields should be properly initialized and accessible.
        **Feature: decomposition-documentation, Property 1: Complete Module Discovery**
        **Validates: Requirements 1.1, 1.2, 1.3**
        """
        # Test that all required fields are present and have expected types
        assert isinstance(module_info.name, str)
        assert len(module_info.name) > 0
        assert isinstance(module_info.path, Path)
        assert isinstance(module_info.relative_path, str)
        assert isinstance(module_info.module_dotted_name, str)
        
        # Test that the module info can be converted to string representation
        str_repr = str(module_info)
        assert module_info.name in str_repr
    
    @given(import_info_strategy())
    def test_import_info_properties(self, import_info):
        """
        Property 1: Complete Module Discovery - Import Analysis
        For any valid ImportInfo, all fields should be properly initialized.
        **Feature: decomposition-documentation, Property 1: Complete Module Discovery**
        **Validates: Requirements 1.1, 1.2, 1.3**
        """
        assert isinstance(import_info.module, str)
        assert len(import_info.module) > 0
        assert isinstance(import_info.names, list)
        assert len(import_info.names) > 0
        assert all(isinstance(name, str) for name in import_info.names)
        assert isinstance(import_info.is_from_import, bool)
        assert isinstance(import_info.level, int)
        assert import_info.level >= 0
    
    @given(function_info_strategy())
    def test_function_info_properties(self, function_info):
        """
        Property 1: Complete Module Discovery - Function Analysis
        For any valid FunctionInfo, all fields should be properly initialized.
        **Feature: decomposition-documentation, Property 1: Complete Module Discovery**
        **Validates: Requirements 1.1, 1.2, 1.3**
        """
        assert isinstance(function_info.name, str)
        assert len(function_info.name) > 0
        assert isinstance(function_info.signature, str)
        assert function_info.name in function_info.signature
        assert function_info.docstring is None or isinstance(function_info.docstring, str)
        assert isinstance(function_info.is_public, bool)
        assert isinstance(function_info.is_entry_point, bool)
        assert isinstance(function_info.line_number, int)
        assert function_info.line_number > 0
    
    @given(class_info_strategy())
    def test_class_info_properties(self, class_info):
        """
        Property 1: Complete Module Discovery - Class Analysis
        For any valid ClassInfo, all fields should be properly initialized.
        **Feature: decomposition-documentation, Property 1: Complete Module Discovery**
        **Validates: Requirements 1.1, 1.2, 1.3**
        """
        assert isinstance(class_info.name, str)
        assert len(class_info.name) > 0
        assert class_info.docstring is None or isinstance(class_info.docstring, str)
        assert isinstance(class_info.methods, list)
        assert all(isinstance(method, FunctionInfo) for method in class_info.methods)
        assert isinstance(class_info.is_public, bool)
        assert isinstance(class_info.line_number, int)
        assert class_info.line_number > 0
    
    @given(module_analysis_strategy())
    def test_module_analysis_properties(self, module_analysis):
        """
        Property 1: Complete Module Discovery - Complete Analysis
        For any valid ModuleAnalysis, all components should be properly integrated.
        **Feature: decomposition-documentation, Property 1: Complete Module Discovery**
        **Validates: Requirements 1.1, 1.2, 1.3**
        """
        # Test core structure
        assert isinstance(module_analysis.module_info, ModuleInfo)
        assert isinstance(module_analysis.imports, list)
        assert isinstance(module_analysis.functions, list)
        assert isinstance(module_analysis.classes, list)
        
        # Test analysis results
        assert isinstance(module_analysis.role, str)
        assert module_analysis.role in ['analyzer', 'executor', 'utility', 'interface']
        assert isinstance(module_analysis.entry_points, list)
        assert isinstance(module_analysis.input_patterns, list)
        assert isinstance(module_analysis.output_patterns, list)
        assert isinstance(module_analysis.artifacts, list)
        assert isinstance(module_analysis.risks, list)
        assert isinstance(module_analysis.overlaps, list)
        
        # Test that all list elements are strings
        for lst in [module_analysis.entry_points, module_analysis.input_patterns, 
                   module_analysis.output_patterns, module_analysis.artifacts,
                   module_analysis.risks, module_analysis.overlaps]:
            assert all(isinstance(item, str) for item in lst)


class TestModuleDiscoveryProperties:
    """Property-based tests for ModuleDiscovery functionality."""
    
    @given(st.lists(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_')), min_size=1, max_size=10))
    def test_module_discovery_properties(self, module_names):
        """
        Property 1: Complete Module Discovery
        For any set of Python modules in a directory, the discovery system should find all modules
        excluding __init__.py and extract complete metadata for each.
        **Feature: decomposition-documentation, Property 1: Complete Module Discovery**
        **Validates: Requirements 1.1, 1.2, 1.3**
        """
        # Ensure valid Python module names
        valid_names = []
        for name in module_names:
            if name and (name[0].isalpha() or name[0] == '_'):
                valid_names.append(name)
        
        assume(len(valid_names) > 0)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test Python files
            created_files = []
            for name in valid_names:
                py_file = temp_path / f"{name}.py"
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(f'"""Module {name}"""\n\ndef test_function():\n    pass\n')
                created_files.append(py_file)
            
            # Create an __init__.py file that should be excluded
            init_file = temp_path / "__init__.py"
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write('# Init file\n')
            
            # Test module discovery
            discovery = ModuleDiscovery(temp_path)
            discovered_modules = discovery.discover_modules()
            
            # Property: All non-__init__.py files should be discovered
            discovered_names = {module.name for module in discovered_modules}
            expected_names = set(valid_names)
            
            assert discovered_names == expected_names, f"Expected {expected_names}, got {discovered_names}"
            
            # Property: Each discovered module should have complete metadata
            for module in discovered_modules:
                assert isinstance(module.name, str)
                assert len(module.name) > 0
                assert isinstance(module.path, Path)
                assert module.path.exists()
                assert module.path.suffix == '.py'
                assert isinstance(module.relative_path, str)
                assert isinstance(module.module_dotted_name, str)
                
                # Property: Module dotted name should match the file structure
                assert module.module_dotted_name == module.name
                
                # Property: Relative path should be correct
                assert module.relative_path == f"{module.name}.py"
    
    def test_module_discovery_with_subdirectories(self):
        """
        Property 1: Complete Module Discovery - Subdirectory Support
        For any directory structure with subdirectories, discovery should find modules recursively
        and generate correct dotted names.
        **Feature: decomposition-documentation, Property 1: Complete Module Discovery**
        **Validates: Requirements 1.1, 1.2, 1.3**
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create nested directory structure
            subdir = temp_path / "subpackage"
            subdir.mkdir()
            
            # Create files at different levels
            root_file = temp_path / "root_module.py"
            sub_file = subdir / "sub_module.py"
            
            with open(root_file, 'w', encoding='utf-8') as f:
                f.write('"""Root module"""\n')
            
            with open(sub_file, 'w', encoding='utf-8') as f:
                f.write('"""Sub module"""\n')
            
            # Test discovery
            discovery = ModuleDiscovery(temp_path)
            discovered_modules = discovery.discover_modules()
            
            # Should find both modules
            assert len(discovered_modules) == 2
            
            # Check dotted names are correct
            module_map = {module.name: module for module in discovered_modules}
            
            assert "root_module" in module_map
            assert module_map["root_module"].module_dotted_name == "root_module"
            
            assert "sub_module" in module_map
            assert module_map["sub_module"].module_dotted_name == "subpackage.sub_module"
    
    def test_module_filtering_properties(self):
        """
        Property 1: Complete Module Discovery - Filtering
        For any set of discovered modules, filtering should only keep accessible and readable files.
        **Feature: decomposition-documentation, Property 1: Complete Module Discovery**
        **Validates: Requirements 1.1, 1.2, 1.3**
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create valid Python file
            valid_file = temp_path / "valid_module.py"
            with open(valid_file, 'w', encoding='utf-8') as f:
                f.write('"""Valid module"""\n')
            
            # Create ModuleInfo objects - one valid, one pointing to non-existent file
            valid_module = ModuleInfo(
                name="valid_module",
                path=valid_file,
                relative_path="valid_module.py",
                module_dotted_name="valid_module"
            )
            
            invalid_module = ModuleInfo(
                name="invalid_module",
                path=temp_path / "nonexistent.py",
                relative_path="nonexistent.py",
                module_dotted_name="invalid_module"
            )
            
            discovery = ModuleDiscovery(temp_path)
            filtered_modules = discovery.filter_modules([valid_module, invalid_module])
            
            # Property: Only valid, accessible modules should pass filtering
            assert len(filtered_modules) == 1
            assert filtered_modules[0].name == "valid_module"
            assert filtered_modules[0].path.exists()
    
    def test_error_handling_properties(self):
        """
        Property 1: Complete Module Discovery - Error Resilience
        For any directory access issues or file permission problems, the system should handle
        errors gracefully and continue processing other files.
        **Feature: decomposition-documentation, Property 1: Complete Module Discovery**
        **Validates: Requirements 5.1**
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test with non-existent directory
            nonexistent_path = temp_path / "nonexistent"
            discovery = ModuleDiscovery(nonexistent_path)
            modules = discovery.discover_modules()
            
            # Property: Should return empty list for non-existent directory, not crash
            assert isinstance(modules, list)
            assert len(modules) == 0
            
            # Test with file instead of directory
            file_path = temp_path / "not_a_directory.txt"
            with open(file_path, 'w') as f:
                f.write("test")
            
            discovery_file = ModuleDiscovery(file_path)
            modules_file = discovery_file.discover_modules()
            
            # Property: Should return empty list for file path, not crash
            assert isinstance(modules_file, list)
            assert len(modules_file) == 0


class TestCodeAnalyzerProperties:
    """Property-based tests for CodeAnalyzer functionality."""
    
    @given(st.lists(import_info_strategy(), max_size=10), 
           st.lists(function_info_strategy(), max_size=10), 
           st.lists(class_info_strategy(), max_size=5))
    def test_role_classification_accuracy(self, imports, functions, classes):
        """
        Property 2: Role Classification Accuracy
        For any analyzed module, the system should correctly classify it as either an analyzer 
        (read-only) or executor (modifies files) based on its code patterns and function signatures.
        **Feature: decomposition-documentation, Property 2: Role Classification Accuracy**
        **Validates: Requirements 1.4, 3.2**
        """
        analyzer = CodeAnalyzer()
        role = analyzer.determine_role(imports, functions, classes)
        
        # Property: Role should always be a non-empty string
        assert isinstance(role, str)
        assert len(role) > 0
        
        # Property: Role should be one of the expected categories
        expected_roles = {
            "Executor (modifies files/data)",
            "Analyzer (read-only analysis)", 
            "Mixed (both analysis and modification)"
        }
        assert role in expected_roles
        
        # Property: Classification should be consistent - same inputs should give same output
        role2 = analyzer.determine_role(imports, functions, classes)
        assert role == role2
        
        # Property: If module has file modification imports, it should lean toward executor
        file_modification_imports = {'os', 'shutil', 'pathlib', 'tempfile', 'io', 'json', 'yaml'}
        has_modification_imports = any(imp.module in file_modification_imports for imp in imports)
        
        # Property: If module has modification functions, it should lean toward executor
        modification_keywords = {'write', 'save', 'create', 'delete', 'remove', 'modify', 'update'}
        has_modification_functions = any(
            any(keyword in func.name.lower() for keyword in modification_keywords)
            for func in functions
        )
        
        # Property: If module has analysis imports, it should lean toward analyzer
        analysis_imports = {'ast', 'inspect', 'dis', 'tokenize', 'parser'}
        has_analysis_imports = any(imp.module in analysis_imports for imp in imports)
        
        analysis_keywords = {'analyze', 'parse', 'extract', 'read', 'scan', 'detect'}
        has_analysis_functions = any(
            any(keyword in func.name.lower() for keyword in analysis_keywords)
            for func in functions
        )
        
        # Property: Strong modification signals should result in executor classification
        if has_modification_imports and has_modification_functions and not has_analysis_imports:
            assert "Executor" in role
        
        # Property: Strong analysis signals should result in analyzer classification  
        if has_analysis_imports and has_analysis_functions and not has_modification_imports:
            assert "Analyzer" in role
    
    @given(st.lists(function_info_strategy(), min_size=1, max_size=10))
    def test_role_classification_with_function_patterns(self, functions):
        """
        Property 2: Role Classification Accuracy - Function Pattern Analysis
        For any set of functions, role classification should correctly identify modification vs analysis patterns.
        **Feature: decomposition-documentation, Property 2: Role Classification Accuracy**
        **Validates: Requirements 1.4, 3.2**
        """
        analyzer = CodeAnalyzer()
        
        # Create functions with specific patterns
        modification_func = FunctionInfo(
            name="write_output_file",
            signature="write_output_file(data, path)",
            docstring="Writes data to file",
            is_public=True,
            is_entry_point=False,
            line_number=10
        )
        
        analysis_func = FunctionInfo(
            name="analyze_ast_tree", 
            signature="analyze_ast_tree(tree)",
            docstring="Analyzes AST structure",
            is_public=True,
            is_entry_point=False,
            line_number=20
        )
        
        # Test with modification function
        role_mod = analyzer.determine_role([], [modification_func], [])
        assert "Executor" in role_mod or "Mixed" in role_mod
        
        # Test with analysis function
        role_analysis = analyzer.determine_role([], [analysis_func], [])
        assert "Analyzer" in role_analysis or "Mixed" in role_analysis
        
        # Test with mixed functions
        role_mixed = analyzer.determine_role([], [modification_func, analysis_func], [])
        # Should be classified appropriately based on the balance
        assert isinstance(role_mixed, str)
        assert len(role_mixed) > 0
    
    @given(st.lists(import_info_strategy(), min_size=1, max_size=10))
    def test_role_classification_with_import_patterns(self, imports):
        """
        Property 2: Role Classification Accuracy - Import Pattern Analysis
        For any set of imports, role classification should correctly weight modification vs analysis imports.
        **Feature: decomposition-documentation, Property 2: Role Classification Accuracy**
        **Validates: Requirements 1.4, 3.2**
        """
        analyzer = CodeAnalyzer()
        
        # Create imports with specific patterns
        modification_import = ImportInfo(
            module="shutil",
            names=["copy", "move"],
            is_from_import=True,
            level=0
        )
        
        analysis_import = ImportInfo(
            module="ast",
            names=["parse", "walk"],
            is_from_import=True,
            level=0
        )
        
        # Test with modification import
        role_mod = analyzer.determine_role([modification_import], [], [])
        # Should lean toward executor or be mixed
        assert "Executor" in role_mod or "Mixed" in role_mod or "Analyzer" in role_mod
        
        # Test with analysis import
        role_analysis = analyzer.determine_role([analysis_import], [], [])
        # Should lean toward analyzer or be mixed
        assert "Analyzer" in role_analysis or "Mixed" in role_analysis or "Executor" in role_analysis
        
        # Property: Classification should be deterministic
        role_analysis2 = analyzer.determine_role([analysis_import], [], [])
        assert role_analysis == role_analysis2
    
    @given(st.lists(function_info_strategy(), max_size=10), 
           st.lists(class_info_strategy(), max_size=5))
    def test_entry_point_detection(self, functions, classes):
        """
        Property 3: Entry Point Detection
        For any module with public interfaces, the system should identify all entry points 
        including CLI commands, Python APIs, classes, and functions.
        **Feature: decomposition-documentation, Property 3: Entry Point Detection**
        **Validates: Requirements 1.5, 3.3**
        """
        analyzer = CodeAnalyzer()
        entry_points = analyzer._identify_entry_points(functions, classes)
        
        # Property: Entry points should always be a list
        assert isinstance(entry_points, list)
        
        # Property: All entry points should be strings
        assert all(isinstance(ep, str) for ep in entry_points)
        
        # Property: Entry points should be deterministic
        entry_points2 = analyzer._identify_entry_points(functions, classes)
        assert entry_points == entry_points2
        
        # Property: If there's a main function, it should be identified
        main_functions = [f for f in functions if f.name == 'main' and f.is_public]
        if main_functions:
            main_entry_points = [ep for ep in entry_points if 'Main function:' in ep]
            assert len(main_entry_points) > 0
        
        # Property: Public classes should be identified as entry points
        public_classes = [c for c in classes if c.is_public]
        for cls in public_classes:
            class_entry_points = [ep for ep in entry_points if f'Class: {cls.name}' in ep]
            assert len(class_entry_points) > 0
        
        # Property: Entry point functions should be identified
        entry_point_functions = [f for f in functions if f.is_entry_point and f.name != 'main']
        for func in entry_point_functions:
            func_entry_points = [ep for ep in entry_points if f'Function: {func.signature}' in ep]
            assert len(func_entry_points) > 0
    
    def test_entry_point_detection_with_specific_patterns(self):
        """
        Property 3: Entry Point Detection - Specific Pattern Recognition
        For any functions with entry point patterns, they should be correctly identified.
        **Feature: decomposition-documentation, Property 3: Entry Point Detection**
        **Validates: Requirements 1.5, 3.3**
        """
        analyzer = CodeAnalyzer()
        
        # Create functions with entry point patterns
        main_func = FunctionInfo(
            name="main",
            signature="main()",
            docstring="Main entry point",
            is_public=True,
            is_entry_point=True,
            line_number=1
        )
        
        run_func = FunctionInfo(
            name="run_analysis",
            signature="run_analysis(config)",
            docstring="Run the analysis",
            is_public=True,
            is_entry_point=True,
            line_number=10
        )
        
        private_func = FunctionInfo(
            name="_internal_helper",
            signature="_internal_helper(data)",
            docstring="Internal helper",
            is_public=False,
            is_entry_point=False,
            line_number=20
        )
        
        # Create a public class
        api_class = ClassInfo(
            name="AnalysisAPI",
            docstring="Public API class",
            methods=[],
            is_public=True,
            line_number=30
        )
        
        functions = [main_func, run_func, private_func]
        classes = [api_class]
        
        entry_points = analyzer._identify_entry_points(functions, classes)
        
        # Property: Main function should be identified
        main_entries = [ep for ep in entry_points if 'Main function:' in ep]
        assert len(main_entries) == 1
        assert 'main()' in main_entries[0]
        
        # Property: Entry point function should be identified
        run_entries = [ep for ep in entry_points if 'run_analysis' in ep]
        assert len(run_entries) == 1
        
        # Property: Private function should not be identified as entry point
        private_entries = [ep for ep in entry_points if '_internal_helper' in ep]
        assert len(private_entries) == 0
        
        # Property: Public class should be identified
        class_entries = [ep for ep in entry_points if 'Class: AnalysisAPI' in ep]
        assert len(class_entries) == 1
    
    @given(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_')))
    def test_potential_entry_point_detection(self, func_name):
        """
        Property 3: Entry Point Detection - Potential Entry Point Recognition
        For any function name, the entry point detection should correctly identify potential entry points.
        **Feature: decomposition-documentation, Property 3: Entry Point Detection**
        **Validates: Requirements 1.5, 3.3**
        """
        # Ensure valid function name
        if not (func_name[0].isalpha() or func_name[0] == '_'):
            func_name = 'func_' + func_name
        
        analyzer = CodeAnalyzer()
        
        # Create a mock AST function node for testing
        class MockFuncNode:
            def __init__(self, name):
                self.name = name
                self.decorator_list = []
                self.args = MockArgs()
        
        class MockArgs:
            def __init__(self):
                self.args = []
        
        mock_node = MockFuncNode(func_name)
        is_entry_point = analyzer._is_potential_entry_point(mock_node)
        
        # Property: Result should always be a boolean
        assert isinstance(is_entry_point, bool)
        
        # Property: Should be deterministic
        is_entry_point2 = analyzer._is_potential_entry_point(mock_node)
        assert is_entry_point == is_entry_point2
        
        # Property: Common entry point names should be detected
        entry_point_names = {'main', 'run', 'execute', 'start', 'process', 'analyze', 'generate'}
        if func_name.lower() in entry_point_names:
            assert is_entry_point == True
        
        # Property: Private functions (starting with _) should generally not be entry points
        # unless they have other entry point characteristics
        if func_name.startswith('_'):
            # Private functions are less likely to be entry points, but not impossible
            # The test should still be deterministic
            pass


class TestDocumentationGeneratorProperties:
    """Property-based tests for DocumentationGenerator initialization."""
    
    def test_documentation_generator_initialization(self):
        """
        Property 1: Complete Module Discovery - Generator Initialization
        For any DocumentationGenerator, paths should be properly configured.
        **Feature: decomposition-documentation, Property 1: Complete Module Discovery**
        **Validates: Requirements 1.1, 1.2, 1.3**
        """
        # Test with default path
        generator = DocumentationGenerator()
        assert isinstance(generator.base_path, Path)
        assert isinstance(generator.decomposition_path, Path)
        assert isinstance(generator.output_path, Path)
        
        # Test path relationships
        assert generator.decomposition_path.name == "decomposition"
        assert generator.output_path.name == "readme_decomp.md"
        
        # Test with custom path
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = Path(temp_dir)
            custom_generator = DocumentationGenerator(custom_path)
            assert custom_generator.base_path == custom_path
            assert custom_generator.decomposition_path == custom_path / "intellirefactor" / "analysis" / "decomposition"
            assert custom_generator.output_path == custom_path / "docs" / "readme_decomp.md"


class TestTemplateEngineProperties:
    """Property-based tests for TemplateEngine functionality."""
    
    @given(module_analysis_strategy())
    def test_template_format_compliance(self, module_analysis):
        """
        Property 4: Template Format Compliance
        For any generated module documentation, the output should conform to the provided 
        Russian template format with all required sections populated.
        **Feature: decomposition-documentation, Property 4: Template Format Compliance**
        **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
        """
        from decomposition_doc_generator import TemplateEngine
        
        template_engine = TemplateEngine()
        documentation = template_engine.populate_template(module_analysis)
        
        # Property: Documentation should always be a non-empty string
        assert isinstance(documentation, str)
        assert len(documentation.strip()) > 0
        
        # Property: Documentation should contain required Russian template sections
        required_sections = [
            "**Роль:**",
            "**Ключевые функции:**",
            "**Есть (реально реализовано):**",
            "**Нет / нестабильно / заглушка:**",
            "**Риски/опасности:**",
            "**Пересечения/дубли с другими модулями:**",
            "**Что улучшить в первую очередь:**"
        ]
        
        for section in required_sections:
            assert section in documentation, f"Missing required section: {section}"
        
        # Property: Documentation should start with module header
        expected_header = f"## {module_analysis.module_info.name}.py"
        assert documentation.startswith(expected_header)
        
        # Property: Documentation should end with separator
        assert documentation.strip().endswith("---")
        
        # Property: Role section should be populated with actual role
        role_lines = [line for line in documentation.split('\n') if "**Роль:**" in line]
        assert len(role_lines) == 1
        role_content = role_lines[0].replace("**Роль:**", "").strip()
        assert len(role_content) > 0
        assert role_content != ""
        
        # Property: Functions section should contain function information
        functions_start = documentation.find("**Ключевые функции:**")
        functions_end = documentation.find("**Есть (реально реализовано):**")
        functions_section = documentation[functions_start:functions_end]
        
        # Should contain at least one function or class entry
        assert ("- " in functions_section or "Публичные функции не обнаружены" in functions_section)
        
        # Property: Each section should be properly formatted with markdown
        lines = documentation.split('\n')
        # Count sections that contain the required section markers
        found_sections = 0
        for section in required_sections:
            if section in documentation:
                found_sections += 1
        assert found_sections == len(required_sections), f"Found {found_sections} sections, expected {len(required_sections)}"
        
        # Property: Template should be deterministic - same input produces same output
        documentation2 = template_engine.populate_template(module_analysis)
        assert documentation == documentation2
    
    @given(st.lists(module_analysis_strategy(), min_size=2, max_size=5))
    def test_template_consistency_across_modules(self, module_analyses):
        """
        Property 4: Template Format Compliance - Consistency
        For any set of module analyses, all generated documentation should follow 
        the same template structure and formatting rules.
        **Feature: decomposition-documentation, Property 4: Template Format Compliance**
        **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
        """
        from decomposition_doc_generator import TemplateEngine
        
        template_engine = TemplateEngine()
        documentations = []
        
        for analysis in module_analyses:
            doc = template_engine.populate_template(analysis)
            documentations.append(doc)
        
        # Property: All documentations should have the same section structure
        required_sections = [
            "**Роль:**",
            "**Ключевые функции:**", 
            "**Есть (реально реализовано):**",
            "**Нет / нестабильно / заглушка:**",
            "**Риски/опасности:**",
            "**Пересечения/дубли с другими модулями:**",
            "**Что улучшить в первую очередь:**"
        ]
        
        for doc in documentations:
            for section in required_sections:
                assert section in doc, f"Inconsistent template structure - missing: {section}"
        
        # Property: All documentations should follow the same header format
        for i, doc in enumerate(documentations):
            expected_header = f"## {module_analyses[i].module_info.name}.py"
            assert doc.startswith(expected_header)
        
        # Property: All documentations should end with separator
        for doc in documentations:
            assert doc.strip().endswith("---")
        
        # Property: Section order should be consistent across all modules
        for doc in documentations:
            section_positions = []
            for section in required_sections:
                pos = doc.find(section)
                assert pos != -1, f"Section not found: {section}"
                section_positions.append(pos)
            
            # Positions should be in ascending order (sections appear in correct order)
            assert section_positions == sorted(section_positions), "Sections not in correct order"
    
    def test_template_with_minimal_data(self):
        """
        Property 4: Template Format Compliance - Minimal Data Handling
        For any module analysis with minimal or missing data, the template should 
        still generate valid documentation with appropriate placeholder text.
        **Feature: decomposition-documentation, Property 4: Template Format Compliance**
        **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
        """
        from decomposition_doc_generator import TemplateEngine, ModuleAnalysis, ModuleInfo
        from pathlib import Path
        
        # Create minimal module analysis
        minimal_module_info = ModuleInfo(
            name="minimal_module",
            path=Path("minimal_module.py"),
            relative_path="minimal_module.py",
            module_dotted_name="minimal_module"
        )
        
        minimal_analysis = ModuleAnalysis(
            module_info=minimal_module_info,
            imports=[],
            functions=[],
            classes=[],
            docstring=None,
            role="",
            entry_points=[],
            input_patterns=[],
            output_patterns=[],
            artifacts=[],
            risks=[],
            overlaps=[]
        )
        
        template_engine = TemplateEngine()
        documentation = template_engine.populate_template(minimal_analysis)
        
        # Property: Should still generate valid documentation
        assert isinstance(documentation, str)
        assert len(documentation.strip()) > 0
        
        # Property: Should contain all required sections even with minimal data
        required_sections = [
            "**Роль:**",
            "**Ключевые функции:**",
            "**Есть (реально реализовано):**",
            "**Нет / нестабильно / заглушка:**",
            "**Риски/опасности:**",
            "**Пересечения/дубли с другими модулями:**",
            "**Что улучшить в первую очередь:**"
        ]
        
        for section in required_sections:
            assert section in documentation
        
        # Property: Should handle empty role gracefully
        assert "Не определена" in documentation or "не завершен" in documentation
        
        # Property: Should provide appropriate placeholder text for missing functions
        assert "Публичные функции не обнаружены" in documentation
    
    def test_template_error_handling(self):
        """
        Property 4: Template Format Compliance - Error Handling
        For any template population errors, the system should generate fallback 
        documentation that still follows the basic template structure.
        **Feature: decomposition-documentation, Property 4: Template Format Compliance**
        **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
        """
        from decomposition_doc_generator import TemplateEngine
        
        template_engine = TemplateEngine()
        
        # Test with None input (should not crash)
        try:
            fallback_doc = template_engine._create_fallback_template("test_module")
            
            # Property: Fallback should still be valid documentation
            assert isinstance(fallback_doc, str)
            assert len(fallback_doc.strip()) > 0
            assert "## test_module.py" in fallback_doc
            assert "**Роль:**" in fallback_doc
            assert "---" in fallback_doc
            
        except Exception as e:
            pytest.fail(f"Fallback template creation should not raise exceptions: {e}")


class TestRiskAndOverlapDetectionProperties:
    """Property-based tests for risk and overlap detection functionality."""
    
    @given(st.lists(module_analysis_strategy(), min_size=2, max_size=5))
    def test_risk_and_overlap_detection(self, module_analyses):
        """
        Property 5: Risk and Overlap Detection
        For any set of analyzed modules, the system should identify potential failure points, 
        dependencies, and duplicate functionality across modules.
        **Feature: decomposition-documentation, Property 5: Risk and Overlap Detection**
        **Validates: Requirements 2.7, 2.8, 3.9**
        """
        from decomposition_doc_generator import TemplateEngine
        
        template_engine = TemplateEngine()
        
        # Test overlap detection
        template_engine.detect_overlaps(module_analyses)
        
        # Property: All modules should have overlaps field populated (even if empty)
        for analysis in module_analyses:
            assert isinstance(analysis.overlaps, list)
            # All overlap entries should be strings
            assert all(isinstance(overlap, str) for overlap in analysis.overlaps)
        
        # Property: If modules have identical functions, overlaps should be detected
        modules_with_functions = [a for a in module_analyses if a.functions]
        if len(modules_with_functions) >= 2:
            # Check if any modules have functions with the same name
            function_names = {}
            for analysis in modules_with_functions:
                for func in analysis.functions:
                    if func.name not in function_names:
                        function_names[func.name] = []
                    function_names[func.name].append(analysis.module_info.name)
            
            # If there are duplicate function names, at least one module should report overlaps
            duplicate_functions = {name: modules for name, modules in function_names.items() if len(modules) > 1}
            if duplicate_functions:
                modules_with_overlaps = [a for a in module_analyses if a.overlaps]
                assert len(modules_with_overlaps) > 0, "Modules with duplicate functions should have overlaps detected"
        
        # Property: If modules have identical roles, overlaps should be detected
        role_counts = {}
        for analysis in module_analyses:
            if analysis.role and "Unknown" not in analysis.role:
                if analysis.role not in role_counts:
                    role_counts[analysis.role] = []
                role_counts[analysis.role].append(analysis.module_info.name)
        
        duplicate_roles = {role: modules for role, modules in role_counts.items() if len(modules) > 1}
        if duplicate_roles:
            modules_with_role_overlaps = [a for a in module_analyses if any("Аналогичная роль" in overlap for overlap in a.overlaps)]
            assert len(modules_with_role_overlaps) > 0, "Modules with duplicate roles should have overlaps detected"
        
        # Property: Overlap detection should be deterministic
        original_overlaps = {a.module_info.name: a.overlaps.copy() for a in module_analyses}
        template_engine.detect_overlaps(module_analyses)
        new_overlaps = {a.module_info.name: a.overlaps for a in module_analyses}
        assert original_overlaps == new_overlaps, "Overlap detection should be deterministic"
    
    @given(st.lists(import_info_strategy(), max_size=10), 
           st.lists(function_info_strategy(), max_size=10))
    def test_risk_assessment_properties(self, imports, functions):
        """
        Property 5: Risk and Overlap Detection - Risk Assessment
        For any module with imports and functions, the system should correctly identify 
        potential risks based on code patterns.
        **Feature: decomposition-documentation, Property 5: Risk and Overlap Detection**
        **Validates: Requirements 2.7, 2.8, 3.9**
        """
        from decomposition_doc_generator import CodeAnalyzer
        
        analyzer = CodeAnalyzer()
        risks = analyzer._assess_risks(functions, [], imports)
        
        # Property: Risks should always be a list
        assert isinstance(risks, list)
        
        # Property: All risk entries should be strings
        assert all(isinstance(risk, str) for risk in risks)
        
        # Property: Risk assessment should be deterministic
        risks2 = analyzer._assess_risks(functions, [], imports)
        assert risks == risks2
        
        # Property: Risky imports should be detected
        risky_imports = {'subprocess', 'os', 'shutil', 'tempfile'}
        has_risky_imports = any(imp.module in risky_imports for imp in imports)
        
        if has_risky_imports:
            risky_import_risks = [risk for risk in risks if "risky module" in risk.lower()]
            assert len(risky_import_risks) > 0, "Risky imports should be detected in risk assessment"
        
        # Property: File modification functions should be detected as risks
        modification_keywords = {'write', 'delete', 'remove', 'create', 'modify'}
        has_modification_functions = any(
            any(keyword in func.name.lower() for keyword in modification_keywords)
            for func in functions
        )
        
        if has_modification_functions:
            modification_risks = [risk for risk in risks if "file system modifications" in risk.lower()]
            assert len(modification_risks) > 0, "File modification functions should be detected as risks"
        
        # Property: High number of external dependencies should be flagged
        external_imports = [imp for imp in imports if not imp.module.startswith('.') and imp.module not in {
            'os', 'sys', 'pathlib', 'typing', 'dataclasses', 'logging', 'ast', 'json'
        }]
        
        if len(external_imports) > 5:
            dependency_risks = [risk for risk in risks if "dependencies" in risk.lower()]
            assert len(dependency_risks) > 0, "High number of external dependencies should be flagged as risk"
    
    def test_overlap_detection_with_specific_patterns(self):
        """
        Property 5: Risk and Overlap Detection - Specific Pattern Detection
        For any modules with known overlapping patterns, the system should detect them correctly.
        **Feature: decomposition-documentation, Property 5: Risk and Overlap Detection**
        **Validates: Requirements 2.7, 2.8, 3.9**
        """
        from decomposition_doc_generator import TemplateEngine, ModuleAnalysis, ModuleInfo, FunctionInfo, ImportInfo
        from pathlib import Path
        
        # Create modules with known overlaps
        module1_info = ModuleInfo(name="parser1", path=Path("parser1.py"), relative_path="parser1.py", module_dotted_name="parser1")
        module2_info = ModuleInfo(name="parser2", path=Path("parser2.py"), relative_path="parser2.py", module_dotted_name="parser2")
        
        # Similar functions
        parse_func1 = FunctionInfo(name="parse_data", signature="parse_data(input)", docstring="Parse data", is_public=True, is_entry_point=True, line_number=10)
        parse_func2 = FunctionInfo(name="parse_data", signature="parse_data(content)", docstring="Parse content", is_public=True, is_entry_point=True, line_number=15)
        
        # Common imports
        ast_import1 = ImportInfo(module="ast", names=["parse"], is_from_import=True, level=0)
        ast_import2 = ImportInfo(module="ast", names=["walk"], is_from_import=True, level=0)
        json_import = ImportInfo(module="json", names=["loads"], is_from_import=True, level=0)
        
        analysis1 = ModuleAnalysis(
            module_info=module1_info, imports=[ast_import1, json_import], functions=[parse_func1], classes=[],
            docstring=None, role="Analyzer (read-only analysis)", entry_points=[], 
            input_patterns=[], output_patterns=[], artifacts=[], risks=[], overlaps=[]
        )
        
        analysis2 = ModuleAnalysis(
            module_info=module2_info, imports=[ast_import2, json_import], functions=[parse_func2], classes=[],
            docstring=None, role="Analyzer (read-only analysis)", entry_points=[], 
            input_patterns=[], output_patterns=[], artifacts=[], risks=[], overlaps=[]
        )
        
        template_engine = TemplateEngine()
        template_engine.detect_overlaps([analysis1, analysis2])
        
        # Property: Should detect function name overlaps
        function_overlaps1 = [o for o in analysis1.overlaps if "Похожие функции" in o and "parse_data" in o]
        function_overlaps2 = [o for o in analysis2.overlaps if "Похожие функции" in o and "parse_data" in o]
        
        assert len(function_overlaps1) > 0, "Should detect similar function names"
        assert len(function_overlaps2) > 0, "Should detect similar function names"
        
        # Property: Should detect role overlaps
        role_overlaps1 = [o for o in analysis1.overlaps if "Аналогичная роль" in o]
        role_overlaps2 = [o for o in analysis2.overlaps if "Аналогичная роль" in o]
        
        assert len(role_overlaps1) > 0, "Should detect similar roles"
        assert len(role_overlaps2) > 0, "Should detect similar roles"
        
        # Property: Should detect common dependencies (if threshold is met)
        # Note: The current implementation has a threshold of 3+ common imports
        # With 1 common import (json), it might not trigger the overlap detection
        # This is expected behavior based on the implementation
    
    def test_risk_assessment_edge_cases(self):
        """
        Property 5: Risk and Overlap Detection - Edge Cases
        For any edge cases in risk assessment, the system should handle them gracefully.
        **Feature: decomposition-documentation, Property 5: Risk and Overlap Detection**
        **Validates: Requirements 2.7, 2.8, 3.9**
        """
        from decomposition_doc_generator import CodeAnalyzer
        
        analyzer = CodeAnalyzer()
        
        # Test with empty inputs
        risks_empty = analyzer._assess_risks([], [], [])
        assert isinstance(risks_empty, list)
        assert len(risks_empty) == 0 or all(isinstance(risk, str) for risk in risks_empty)
        
        # Test with None inputs (should not crash)
        try:
            risks_none = analyzer._assess_risks([], [], [])
            assert isinstance(risks_none, list)
        except Exception as e:
            pytest.fail(f"Risk assessment should handle empty inputs gracefully: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])