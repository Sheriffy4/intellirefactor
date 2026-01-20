"""
Standardized expert analysis report generator.

Generates consistent JSON and Markdown reports following the agreed schema.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from intellirefactor.analysis.foundation.models import Severity


@dataclass
class AnalysisFinding:
    """Standardized finding structure for expert analysis."""
    id: str
    type: str  # e.g., "external_call_site", "public_api", "dependency"
    severity: Severity
    confidence: float  # 0.0 to 1.0
    file: str
    line: Optional[int] = None
    symbol: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisError:
    """Standardized error structure for reporting issues."""
    file: str
    stage: str  # "read", "parse", "analyze"
    error_type: str
    message: str


@dataclass
class AnalysisStats:
    """Analysis statistics."""
    files_scanned: int = 0
    python_files_analyzed: int = 0
    parse_errors: int = 0
    findings: int = 0


@dataclass
class ExpertReport:
    """Standardized expert analysis report structure."""
    tool: str = "intellirefactor"
    tool_version: str = "0.1.0"
    analysis_type: str = "expert"
    project_root: str = ""
    started_at: str = ""
    duration_seconds: float = 0.0
    stats: AnalysisStats = field(default_factory=AnalysisStats)
    findings: List[AnalysisFinding] = field(default_factory=list)
    errors: List[AnalysisError] = field(default_factory=list)


class ExpertReportGenerator:
    """Generates standardized expert analysis reports."""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.start_time = datetime.utcnow()
        self.report = ExpertReport(
            project_root=project_root,
            started_at=self.start_time.isoformat() + "Z"
        )
    
    def add_finding(self, finding: AnalysisFinding) -> None:
        """Add a finding to the report."""
        self.report.findings.append(finding)
        self.report.stats.findings = len(self.report.findings)
    
    def add_error(self, error: AnalysisError) -> None:
        """Add an error to the report."""
        self.report.errors.append(error)
        self.report.stats.parse_errors = len(self.report.errors)
    
    def finalize(self, duration_seconds: float) -> None:
        """Finalize the report with timing information."""
        self.report.duration_seconds = duration_seconds
    
    def to_json(self) -> str:
        """Generate JSON report string."""
        payload = asdict(self.report)

        def to_jsonable(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_jsonable(x) for x in obj]
            if isinstance(obj, Enum):
                return obj.value
            return obj

        return json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False)
    
    def to_markdown(self) -> str:
        """Generate human-readable Markdown report."""
        lines = [
            "# IntelliRefactor Expert Analysis Report",
            "",
            f"**Project:** {self.report.project_root}",
            f"**Analysis Type:** {self.report.analysis_type}",
            f"**Duration:** {self.report.duration_seconds:.2f} seconds",
            f"**Started:** {self.report.started_at}",
            "",
            "## Statistics",
            f"- Files scanned: {self.report.stats.files_scanned}",
            f"- Python files analyzed: {self.report.stats.python_files_analyzed}",
            f"- Parse errors: {self.report.stats.parse_errors}",
            f"- Findings: {self.report.stats.findings}",
            ""
        ]
        
        if self.report.errors:
            lines.extend([
                "## Errors",
                ""
            ])
            for error in self.report.errors:
                lines.append(f"- **{error.file}** ({error.stage}): {error.message}")
            lines.append("")
        
        if self.report.findings:
            lines.extend([
                "## Findings",
                ""
            ])
            
            # Group findings by type
            findings_by_type = {}
            for finding in self.report.findings:
                if finding.type not in findings_by_type:
                    findings_by_type[finding.type] = []
                findings_by_type[finding.type].append(finding)
            
            for finding_type, findings in findings_by_type.items():
                lines.append(f"### {finding_type.replace('_', ' ').title()}")
                lines.append("")
                for finding in findings:
                    location = f"{finding.file}"
                    if finding.line:
                        location += f":{finding.line}"
                    
                    lines.append(f"- **{finding.symbol or 'N/A'}** ({location})")
                    if finding.evidence:
                        for key, value in finding.evidence.items():
                            lines.append(f"  - {key}: {value}")
                    lines.append("")
        
        return "\n".join(lines)
    
    def save_reports(self, output_dir: Path) -> Dict[str, str]:
        """
        Save both JSON and Markdown reports.
        
        Args:
            output_dir: Directory to save reports
            
        Returns:
            Dictionary mapping format to file path
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_path = output_dir / f"expert_report_{timestamp}.json"
        json_path.write_text(self.to_json(), encoding="utf-8")
        
        # Save Markdown report
        md_path = output_dir / f"expert_report_{timestamp}.md"
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        
        return {
            "json": str(json_path),
            "markdown": str(md_path)
        }


# Utility functions for creating standard findings
def create_external_call_site_finding(
    file_path: str,
    line_number: int,
    symbol: str,
    snippet: str,
    notes: str = ""
) -> AnalysisFinding:
    """Create a standardized external call site finding."""
    return AnalysisFinding(
        id=f"EXTERNAL_CALLSITE_{hash(file_path + str(line_number)) % 10000:04d}",
        type="external_call_site",
        severity=Severity.INFO,
        confidence=0.7,
        file=file_path,
        line=line_number,
        symbol=symbol,
        evidence={
            "snippet": snippet,
            "notes": notes
        }
    )


def create_public_api_finding(
    file_path: str,
    symbol: str,
    signature: str,
    is_public: bool = True
) -> AnalysisFinding:
    """Create a standardized public API finding."""
    return AnalysisFinding(
        id=f"PUBLIC_API_{hash(file_path + symbol) % 10000:04d}",
        type="public_api",
        severity=Severity.INFO if is_public else Severity.LOW,
        confidence=0.9,
        file=file_path,
        symbol=symbol,
        evidence={
            "signature": signature,
            "visibility": "public" if is_public else "private"
        }
    )


def create_dependency_finding(
    file_path: str,
    imported_module: str,
    import_type: str,  # "internal", "external", "stdlib"
    line_number: Optional[int] = None
) -> AnalysisFinding:
    """Create a standardized dependency finding."""
    return AnalysisFinding(
        id=f"DEPENDENCY_{hash(file_path + imported_module) % 10000:04d}",
        type="dependency",
        severity=Severity.INFO,
        confidence=0.95,
        file=file_path,
        line=line_number,
        evidence={
            "imported_module": imported_module,
            "import_type": import_type
        }
    )


def create_analysis_error(
    file_path: str,
    stage: str,
    error_type: str,
    message: str
) -> AnalysisError:
    """Create a standardized analysis error."""
    return AnalysisError(
        file=file_path,
        stage=stage,
        error_type=error_type,
        message=message
    )