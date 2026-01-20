"""
Утилиты безопасного доступа к словарям и анализаторам.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AnalyzerAccessError(Exception):
    """Ошибка доступа к анализатору."""
    pass


def safe_get_nested(
    data: Dict[str, Any],
    *keys: str,
    default: T = None
) -> T:
    """
    Безопасное получение вложенного значения из словаря.
    
    Args:
        data: Исходный словарь
        *keys: Путь ключей (например, 'call_graph', 'total_relationships')
        default: Значение по умолчанию
        
    Returns:
        Значение или default
        
    Example:
        >>> safe_get_nested(data, 'call_graph', 'summary', 'total', default=0)
    """
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def safe_analyzer_call(
    analyzers: Dict[str, Any],
    analyzer_key: str,
    method_name: str,
    *args,
    fallback: T = None,
    on_error: Optional[Callable[[Exception], None]] = None,
    **kwargs
) -> T:
    """
    Безопасный вызов метода анализатора.
    
    Args:
        analyzers: Словарь анализаторов
        analyzer_key: Ключ анализатора
        method_name: Имя метода
        *args: Позиционные аргументы
        fallback: Значение при ошибке
        on_error: Callback при ошибке
        **kwargs: Именованные аргументы
        
    Returns:
        Результат вызова или fallback
    """
    analyzer = analyzers.get(analyzer_key)
    if analyzer is None:
        logger.warning(f"Analyzer '{analyzer_key}' not found in registry")
        return fallback
    
    method = getattr(analyzer, method_name, None)
    if method is None:
        logger.warning(f"Method '{method_name}' not found in analyzer '{analyzer_key}'")
        return fallback
    
    if not callable(method):
        logger.warning(f"'{analyzer_key}.{method_name}' is not callable")
        return fallback
    
    try:
        return method(*args, **kwargs)
    except Exception as e:
        logger.error(f"Analyzer '{analyzer_key}.{method_name}' failed: {e}")
        if on_error:
            on_error(e)
        return fallback


class SafeAnalyzerRegistry:
    """
    Обертка для безопасного доступа к анализаторам.
    
    Преимущества:
    - Логирование отсутствующих анализаторов
    - Единообразная обработка ошибок
    - Отслеживание использования
    """
    
    def __init__(self, analyzers: Dict[str, Any]):
        self._analyzers = analyzers
        self._access_log: Dict[str, int] = {}
        self._missing_keys: set = set()
    
    def get(self, key: str) -> Optional[Any]:
        """Получить анализатор по ключу."""
        self._access_log[key] = self._access_log.get(key, 0) + 1
        
        analyzer = self._analyzers.get(key)
        if analyzer is None:
            self._missing_keys.add(key)
            logger.debug(f"Analyzer '{key}' not found")
        return analyzer
    
    def call(
        self,
        key: str,
        method: str,
        *args,
        fallback: T = None,
        **kwargs
    ) -> T:
        """Вызвать метод анализатора."""
        return safe_analyzer_call(
            self._analyzers,
            key,
            method,
            *args,
            fallback=fallback,
            **kwargs
        )
    
    def get_missing_keys(self) -> set:
        """Получить список запрошенных, но отсутствующих ключей."""
        return self._missing_keys.copy()
    
    def get_access_stats(self) -> Dict[str, int]:
        """Получить статистику обращений."""
        return self._access_log.copy()
    
    def __contains__(self, key: str) -> bool:
        return key in self._analyzers
    
    def __getitem__(self, key: str) -> Any:
        """Для обратной совместимости, но с логированием."""
        analyzer = self.get(key)
        if analyzer is None:
            raise KeyError(f"Analyzer '{key}' not found")
        return analyzer