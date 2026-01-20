"""
Backward-compatible shim for index_schema.

This module has been moved to:
  intellirefactor.analysis.index.schema

Maintained for backward compatibility.
"""

from intellirefactor.analysis.index.schema import *  # noqa

# For explicit imports that might be used
try:
    from intellirefactor.analysis.index.schema import IndexSchema
except ImportError:
    pass