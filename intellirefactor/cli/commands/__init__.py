"""
CLI command handlers.

Organized by functional domain:
- analysis.py: Analysis, visualization, and documentation commands
- config.py: Configuration, template, and status commands
- refactoring.py: Refactoring opportunities and operations
- deduplication.py: Index and duplicate detection commands (future)
- quality.py: Unused code, audit, and smell detection commands (future)
"""

# Re-export analysis commands for backward compatibility
from .analysis import (
    cmd_analyze,
    cmd_visualize,
    cmd_docs,
    format_analysis_result,
)

# Re-export config commands for backward compatibility
from .config import (
    cmd_status,
    cmd_config,
    cmd_template,
)

# Re-export refactoring commands for backward compatibility
from .refactoring import (
    cmd_opportunities,
    cmd_refactor,
    cmd_apply,
    format_refactoring_result,
)

# Re-export knowledge commands for backward compatibility
from .knowledge import (
    cmd_knowledge,
    cmd_report,
)

# Re-export deduplication commands for backward compatibility
from .deduplication import (
    cmd_index,
    cmd_duplicates,
    get_index_db_path,
    format_index_build_result,
    format_index_status,
)

# Re-export quality commands for backward compatibility
from .quality import (
    cmd_unused,
    cmd_audit,
    cmd_smells,
    format_audit_results,
)

__all__ = [
    # Analysis commands
    'cmd_analyze',
    'cmd_visualize',
    'cmd_docs',
    'format_analysis_result',
    # Config commands
    'cmd_status',
    'cmd_config',
    'cmd_template',
    # Refactoring commands
    'cmd_opportunities',
    'cmd_refactor',
    'cmd_apply',
    'format_refactoring_result',
    # Knowledge commands
    'cmd_knowledge',
    'cmd_report',
    # Deduplication commands
    'cmd_index',
    'cmd_duplicates',
    'get_index_db_path',
    'format_index_build_result',
    'format_index_status',
    # Quality commands
    'cmd_unused',
    'cmd_audit',
    'cmd_smells',
    'format_audit_results',
]

