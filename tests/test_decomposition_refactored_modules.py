"""
Smoke tests for refactored decomposition modules.

Tests basic functionality of modules extracted from DecompositionAnalyzer god class.
"""

import ast
import tempfile
from pathlib import Path

import pytest

from intellirefactor.analysis.decomposition import (
    DecompositionAnalyzer,
    FileOperations,
    SafeExactEvaluator,
    UnifiedSymbolGenerator,
    WrapperPatcher,
    ImportUpdater,
    UnifiedAliasValidator,
    StatisticsGenerator,
    DecompositionConfig,
    FunctionalBlock,
    SimilarityCluster,
    CanonicalizationPlan,
    ProjectFunctionalMap,
    RecommendationType,
    RiskLevel,
    EffortClass,
)
from intellirefactor.analysis.decomposition import ast_utils


class TestAstUtils:
    """Test AST utility functions."""

    def test_find_def_node(self):
        """Test finding function definition nodes."""
        code = """
def foo():
    pass

def bar():
    pass
"""
        tree = ast.parse(code)
        node = ast_utils.find_def_node(tree, "foo")
        assert isinstance(node, ast.FunctionDef)
        assert node.name == "foo"

    def test_get_decorator_name(self):
        """Test extracting decorator names."""
        code = "@staticmethod\ndef foo(): pass"
        tree = ast.parse(code)
        func = tree.body[0]
        dec_name = ast_utils.get_decorator_name(func.decorator_list[0])
        assert dec_name == "staticmethod"

    def test_collect_free_names(self):
        """Test collecting free variable names."""
        code = """
def foo():
    x = 1
    return x + y
"""
        tree = ast.parse(code)
        func = tree.body[0]
        free_names = ast_utils.collect_free_names(func)
        assert "y" in free_names
        assert "x" not in free_names  # x is local


class TestFileOperations:
    """Test FileOperations class."""

    def test_initialization(self):
        """Test FileOperations initialization."""
        config = DecompositionConfig.default()
        file_ops = FileOperations(config, None)
        assert file_ops.config == config

    def test_read_write_text(self):
        """Test reading and writing text files."""
        config = DecompositionConfig.default()
        file_ops = FileOperations(config, None)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_path = Path(f.name)
            f.write("test content")
        
        try:
            content = file_ops.read_text(temp_path)
            assert content == "test content"
            
            file_ops.write_text(temp_path, "new content")
            content = file_ops.read_text(temp_path)
            assert content == "new content"
        finally:
            temp_path.unlink()

    def test_parse_file(self):
        """Test parsing Python files."""
        config = DecompositionConfig.default()
        file_ops = FileOperations(config, None)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            temp_path = Path(f.name)
            f.write("def foo(): pass")
        
        try:
            tree = file_ops.parse_file(temp_path)
            assert isinstance(tree, ast.Module)
            assert len(tree.body) == 1
            assert isinstance(tree.body[0], ast.FunctionDef)
        finally:
            temp_path.unlink()


class TestSafeExactEvaluator:
    """Test SafeExactEvaluator class."""

    def test_initialization(self):
        """Test SafeExactEvaluator initialization."""
        config = DecompositionConfig.default()
        file_ops = FileOperations(config, None)
        evaluator = SafeExactEvaluator(file_ops, ast_utils, set())
        assert evaluator.file_ops == file_ops


class TestUnifiedSymbolGenerator:
    """Test UnifiedSymbolGenerator class."""

    def test_initialization(self):
        """Test UnifiedSymbolGenerator initialization."""
        config = DecompositionConfig.default()
        file_ops = FileOperations(config, None)
        generator = UnifiedSymbolGenerator(file_ops, ast_utils, "[MARKER]")
        assert generator.file_ops == file_ops
        # Note: _WRAPPER_MARKER is stored internally, not as public attribute


class TestWrapperPatcher:
    """Test WrapperPatcher class."""

    def test_initialization(self):
        """Test WrapperPatcher initialization."""
        patcher = WrapperPatcher("[MARKER]", set(), ast_utils)
        assert patcher._WRAPPER_MARKER == "[MARKER]"

    def test_module_dotted_from_target_module(self):
        """Test module path conversion."""
        patcher = WrapperPatcher("[MARKER]", set(), ast_utils)
        result = patcher._module_dotted_from_target_module("path/to/module.py")
        assert result == "path.to.module"


class TestImportUpdater:
    """Test ImportUpdater class."""

    def test_initialization(self):
        """Test ImportUpdater initialization."""
        config = DecompositionConfig.default()
        file_ops = FileOperations(config, None)
        updater = ImportUpdater(file_ops, None)
        assert updater.file_ops == file_ops

    def test_segment_has_comment(self):
        """Test comment detection in code segments."""
        config = DecompositionConfig.default()
        file_ops = FileOperations(config, None)
        updater = ImportUpdater(file_ops, None)
        
        assert updater._segment_has_comment("# comment")
        assert updater._segment_has_comment("x = 1  # comment")
        assert not updater._segment_has_comment("x = 1")


class TestUnifiedAliasValidator:
    """Test UnifiedAliasValidator class."""

    def test_initialization(self):
        """Test UnifiedAliasValidator initialization."""
        validator = UnifiedAliasValidator()
        assert validator is not None

    def test_validate_no_collision(self):
        """Test validation with no collisions."""
        validator = UnifiedAliasValidator()
        code = """
from module import foo as __ir_foo
from module import bar as __ir_bar
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            temp_path = Path(f.name)
        
        try:
            # Should not raise
            validator.validate_unified_import_aliases(file_path=temp_path, code=code)
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestStatisticsGenerator:
    """Test StatisticsGenerator class."""

    def test_initialization(self):
        """Test StatisticsGenerator initialization."""
        generator = StatisticsGenerator()
        assert generator is not None

    def test_generate_statistics_empty(self):
        """Test statistics generation with empty data."""
        generator = StatisticsGenerator()
        stats = generator.generate_statistics(None, [], [])
        assert stats == {}

    def test_calculate_cluster_benefit(self):
        """Test cluster benefit calculation."""
        generator = StatisticsGenerator()
        
        # Create a mock cluster
        cluster = SimilarityCluster(
            id="test",
            category="test",
            subcategory="test",
            blocks=["b1", "b2"],
            avg_similarity=0.9,
            recommendation=RecommendationType.MERGE,
            risk_level=RiskLevel.LOW,
            effort_class=EffortClass.S,
        )
        
        benefit = generator.calculate_cluster_benefit(cluster)
        assert benefit > 0
        assert isinstance(benefit, float)


class TestDecompositionAnalyzer:
    """Test DecompositionAnalyzer integration."""

    def test_initialization(self):
        """Test DecompositionAnalyzer initialization."""
        analyzer = DecompositionAnalyzer()
        assert analyzer.config is not None
        assert analyzer.file_ops is not None
        assert analyzer.safe_exact_eval is not None
        assert analyzer.unified_gen is not None
        assert analyzer.wrapper_patcher is not None
        assert analyzer.import_updater is not None
        assert analyzer.alias_validator is not None
        assert analyzer.stats_gen is not None

    def test_get_functional_map_empty(self):
        """Test getting functional map when empty."""
        analyzer = DecompositionAnalyzer()
        fm = analyzer.get_functional_map()
        assert fm is None

    def test_get_clusters_empty(self):
        """Test getting clusters when empty."""
        analyzer = DecompositionAnalyzer()
        clusters = analyzer.get_clusters()
        assert clusters == []

    def test_get_plans_empty(self):
        """Test getting plans when empty."""
        analyzer = DecompositionAnalyzer()
        plans = analyzer.get_plans()
        assert plans == []


class TestBackwardCompatibility:
    """Test backward compatibility of refactored code."""

    def test_all_modules_importable(self):
        """Test that all refactored modules can be imported."""
        from intellirefactor.analysis.decomposition import (
            DecompositionAnalyzer,
            FileOperations,
            SafeExactEvaluator,
            UnifiedSymbolGenerator,
            WrapperPatcher,
            ImportUpdater,
            UnifiedAliasValidator,
            StatisticsGenerator,
        )
        
        # All imports should succeed
        assert DecompositionAnalyzer is not None
        assert FileOperations is not None
        assert SafeExactEvaluator is not None
        assert UnifiedSymbolGenerator is not None
        assert WrapperPatcher is not None
        assert ImportUpdater is not None
        assert UnifiedAliasValidator is not None
        assert StatisticsGenerator is not None

    def test_decomposition_analyzer_has_all_methods(self):
        """Test that DecompositionAnalyzer still has all expected methods."""
        analyzer = DecompositionAnalyzer()
        
        # Public API methods
        assert hasattr(analyzer, 'analyze_project')
        assert hasattr(analyzer, 'get_functional_map')
        assert hasattr(analyzer, 'get_clusters')
        assert hasattr(analyzer, 'get_plans')
        assert hasattr(analyzer, 'get_cluster_details')
        assert hasattr(analyzer, 'get_top_opportunities')
        assert hasattr(analyzer, 'export_results')
        
        # Delegated methods (should still exist)
        assert hasattr(analyzer, '_evaluate_safe_exact')
        assert hasattr(analyzer, '_apply_wrapper_patch')
        assert hasattr(analyzer, '_generate_statistics')
        assert hasattr(analyzer, '_generate_recommendations')
        assert hasattr(analyzer, '_calculate_cluster_benefit')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
