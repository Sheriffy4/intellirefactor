"""
REMOVED: intellirefactor.analysis.models

All analysis contract models were migrated to:
  intellirefactor.analysis.foundation.models

This module intentionally raises to force cleanup of legacy imports.
"""

raise ImportError(
    "intellirefactor.analysis.models has been removed. "
    "Use intellirefactor.analysis.foundation.models instead."
)
