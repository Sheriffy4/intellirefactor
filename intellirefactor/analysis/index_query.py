"""
Backward-compatible shim for index_query.

This module has been moved to:
  intellirefactor.analysis.index.query

Maintained for backward compatibility.
"""

from intellirefactor.analysis.index.query import *  # noqa

# For explicit imports that might be used
try:
    from intellirefactor.analysis.index.query import IndexQuery
except ImportError:
    pass