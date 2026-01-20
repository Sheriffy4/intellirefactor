"""
Standardized stub utilities for IntelliRefactor.

Provides consistent way to mark unimplemented functionality across the codebase.
This ensures transparency about what features are planned vs implemented.

Key Features:
1. Unified NotImplementedError formatting
2. Automatic backlog generation capability  
3. Consistent messaging across all stubs
4. Easy identification of planned features
"""

from typing import Optional


def not_implemented(
    feature: str, 
    *, 
    owner: str, 
    hint: str = "",
    planned_step: Optional[str] = None
) -> NotImplementedError:
    """
    Create a standardized NotImplementedError with consistent formatting.
    
    Args:
        feature: Name of the unimplemented feature/function
        owner: Class/module that owns this feature
        hint: Additional context or guidance
        planned_step: Reference to roadmap step (e.g., "Step 4.B")
        
    Returns:
        NotImplementedError with standardized message
        
    Example:
        >>> raise not_implemented(
        ...     "analyze_project", 
        ...     owner="ProjectAnalyzer",
        ...     hint="See intellirefactor/analysis/project_analyzer.py",
        ...     planned_step="Step 4.B"
        ... )
    """
    msg_parts = [f"{owner}: '{feature}' is not implemented yet."]
    
    if planned_step:
        msg_parts.append(f"Planned in {planned_step}.")
    
    if hint:
        msg_parts.append(hint)
        
    message = " ".join(msg_parts)
    return NotImplementedError(message)


def stub_audit_entry(
    module: str,
    symbol: str, 
    feature: str,
    owner: str,
    planned_step: Optional[str] = None,
    hint: str = ""
) -> dict:
    """
    Create a standardized entry for stub audit/backlog.
    
    Args:
        module: Module path (e.g., "intellirefactor.analysis.project_analyzer")
        symbol: Function/class name
        feature: Feature description
        owner: Owning component
        planned_step: Roadmap step reference
        hint: Additional context
        
    Returns:
        Dictionary representing backlog entry
    """
    return {
        "module": module,
        "symbol": symbol,
        "feature": feature,
        "owner": owner,
        "planned_step": planned_step,
        "hint": hint,
        "status": "not_implemented"
    }
