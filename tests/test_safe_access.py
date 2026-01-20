"""
Tests for safe_access module.
"""

import pytest
from unittest.mock import Mock

from intellirefactor.analysis.expert.safe_access import (
    safe_get_nested,
    safe_analyzer_call,
    SafeAnalyzerRegistry,
)


class TestSafeGetNested:
    """Tests for safe_get_nested function."""
    
    def test_simple_key(self):
        """Test getting a simple key."""
        data = {'a': 1}
        assert safe_get_nested(data, 'a') == 1
    
    def test_nested_keys(self):
        """Test getting nested keys."""
        data = {'a': {'b': {'c': 42}}}
        assert safe_get_nested(data, 'a', 'b', 'c') == 42
    
    def test_missing_key_returns_default(self):
        """Test that missing key returns default."""
        data = {'a': 1}
        assert safe_get_nested(data, 'b', default=0) == 0
    
    def test_missing_nested_key_returns_default(self):
        """Test that missing nested key returns default."""
        data = {'a': {'b': 1}}
        assert safe_get_nested(data, 'a', 'c', 'd', default='missing') == 'missing'
    
    def test_default_is_none(self):
        """Test that default is None when not specified."""
        data = {'a': 1}
        assert safe_get_nested(data, 'b') is None
    
    def test_non_dict_intermediate(self):
        """Test handling non-dict intermediate values."""
        data = {'a': 'string_value'}
        assert safe_get_nested(data, 'a', 'b', default=0) == 0
    
    def test_empty_dict(self):
        """Test with empty dict."""
        assert safe_get_nested({}, 'a', default='empty') == 'empty'
    
    def test_real_world_pattern(self):
        """Test real-world nested pattern from expert_analyzer."""
        data = {
            'call_graph': {
                'call_graph': {
                    'total_relationships': 12
                }
            }
        }
        result = safe_get_nested(data, 'call_graph', 'call_graph', 'total_relationships', default=0)
        assert result == 12


class TestSafeAnalyzerCall:
    """Tests for safe_analyzer_call function."""
    
    def test_successful_call(self):
        """Test successful analyzer call."""
        analyzer = Mock()
        analyzer.analyze.return_value = {'result': 'data'}
        
        analyzers = {'test': analyzer}
        result = safe_analyzer_call(analyzers, 'test', 'analyze', 'arg1', key='value')
        
        assert result == {'result': 'data'}
        analyzer.analyze.assert_called_once_with('arg1', key='value')
    
    def test_missing_analyzer_returns_fallback(self):
        """Test that missing analyzer returns fallback."""
        analyzers = {}
        result = safe_analyzer_call(analyzers, 'missing', 'method', fallback='default')
        assert result == 'default'
    
    def test_missing_method_returns_fallback(self):
        """Test that missing method returns fallback."""
        analyzer = Mock(spec=[])  # No methods
        analyzers = {'test': analyzer}
        
        result = safe_analyzer_call(analyzers, 'test', 'nonexistent', fallback=None)
        assert result is None
    
    def test_exception_returns_fallback(self):
        """Test that exception returns fallback."""
        analyzer = Mock()
        analyzer.analyze.side_effect = ValueError("Test error")
        
        analyzers = {'test': analyzer}
        result = safe_analyzer_call(analyzers, 'test', 'analyze', fallback='error_fallback')
        
        assert result == 'error_fallback'
    
    def test_on_error_callback(self):
        """Test on_error callback is called on exception."""
        analyzer = Mock()
        analyzer.analyze.side_effect = ValueError("Test error")
        
        error_handler = Mock()
        analyzers = {'test': analyzer}
        
        safe_analyzer_call(
            analyzers, 'test', 'analyze',
            fallback=None,
            on_error=error_handler
        )
        
        error_handler.assert_called_once()
        assert isinstance(error_handler.call_args[0][0], ValueError)


class TestSafeAnalyzerRegistry:
    """Tests for SafeAnalyzerRegistry class."""
    
    def test_get_existing_analyzer(self):
        """Test getting an existing analyzer."""
        mock_analyzer = Mock()
        registry = SafeAnalyzerRegistry({'test': mock_analyzer})
        
        assert registry.get('test') is mock_analyzer
    
    def test_get_missing_analyzer(self):
        """Test getting a missing analyzer returns None."""
        registry = SafeAnalyzerRegistry({})
        assert registry.get('missing') is None
    
    def test_missing_keys_tracking(self):
        """Test that missing keys are tracked."""
        registry = SafeAnalyzerRegistry({'a': Mock()})
        
        registry.get('missing1')
        registry.get('missing2')
        registry.get('a')  # This exists
        
        missing = registry.get_missing_keys()
        assert 'missing1' in missing
        assert 'missing2' in missing
        assert 'a' not in missing
    
    def test_access_stats(self):
        """Test access statistics tracking."""
        registry = SafeAnalyzerRegistry({'a': Mock(), 'b': Mock()})
        
        registry.get('a')
        registry.get('a')
        registry.get('b')
        
        stats = registry.get_access_stats()
        assert stats['a'] == 2
        assert stats['b'] == 1
    
    def test_call_method(self):
        """Test calling analyzer method through registry."""
        analyzer = Mock()
        analyzer.process.return_value = 'result'
        
        registry = SafeAnalyzerRegistry({'test': analyzer})
        result = registry.call('test', 'process', 'arg1', kwarg='value')
        
        assert result == 'result'
        analyzer.process.assert_called_once_with('arg1', kwarg='value')
    
    def test_contains(self):
        """Test __contains__ method."""
        registry = SafeAnalyzerRegistry({'a': Mock()})
        
        assert 'a' in registry
        assert 'b' not in registry
    
    def test_getitem_existing(self):
        """Test __getitem__ for existing key."""
        mock_analyzer = Mock()
        registry = SafeAnalyzerRegistry({'a': mock_analyzer})
        
        assert registry['a'] is mock_analyzer
    
    def test_getitem_missing_raises(self):
        """Test __getitem__ raises KeyError for missing key."""
        registry = SafeAnalyzerRegistry({})
        
        with pytest.raises(KeyError):
            _ = registry['missing']