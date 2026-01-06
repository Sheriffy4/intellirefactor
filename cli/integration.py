"""
Integration utilities for connecting CLI with existing IntelliRefactor systems.

Provides seamless integration with KnowledgeManager, orchestration components,
and existing configuration systems while maintaining backward compatibility.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime

from intellirefactor.config import IntelliRefactorConfig
from intellirefactor.knowledge.knowledge_manager import KnowledgeManager
from intellirefactor.orchestration.global_refactoring_orchestrator import GlobalRefactoringOrchestrator
from intellirefactor.orchestration.refactoring_reporter import RefactoringReporter
from intellirefactor.orchestration.refactoring_validator import RefactoringValidator
from intellirefactor.analysis.project_analyzer import ProjectAnalyzer
from intellirefactor.analysis.metrics_analyzer import MetricsAnalyzer
from intellirefactor.refactoring.intelligent_refactoring_system import IntelligentRefactoringSystem
from intellirefactor.safety.safety_manager import SafetyManager
from intellirefactor.plugins.plugin_manager import PluginManager


logger = logging.getLogger(__name__)


class CLIIntegrationManager:
    """Manages integration between CLI and existing IntelliRefactor systems."""
    
    def __init__(self, config: Optional[IntelliRefactorConfig] = None):
        """Initialize the integration manager."""
        self.config = config or IntelliRefactorConfig()
        self._knowledge_manager: Optional[KnowledgeManager] = None
        self._orchestrator: Optional[GlobalRefactoringOrchestrator] = None
        self._reporter: Optional[RefactoringReporter] = None
        self._validator: Optional[RefactoringValidator] = None
        self._project_analyzer: Optional[ProjectAnalyzer] = None
        self._metrics_analyzer: Optional[MetricsAnalyzer] = None
        self._refactoring_system: Optional[IntelligentRefactoringSystem] = None
        self._safety_manager: Optional[SafetyManager] = None
        self._plugin_manager: Optional[PluginManager] = None
    
    @property
    def knowledge_manager(self) -> KnowledgeManager:
        """Get or create knowledge manager instance."""
        if self._knowledge_manager is None:
            self._knowledge_manager = KnowledgeManager(self.config)
        return self._knowledge_manager
    
    @property
    def orchestrator(self) -> GlobalRefactoringOrchestrator:
        """Get or create orchestrator instance."""
        if self._orchestrator is None:
            self._orchestrator = GlobalRefactoringOrchestrator(
                config=self.config
            )
        return self._orchestrator
    
    @property
    def reporter(self) -> RefactoringReporter:
        """Get or create reporter instance."""
        if self._reporter is None:
            self._reporter = RefactoringReporter(self.config)
        return self._reporter
    
    @property
    def validator(self) -> RefactoringValidator:
        """Get or create validator instance."""
        if self._validator is None:
            self._validator = RefactoringValidator(self.config)
        return self._validator
    
    @property
    def project_analyzer(self) -> ProjectAnalyzer:
        """Get or create project analyzer instance."""
        if self._project_analyzer is None:
            self._project_analyzer = ProjectAnalyzer(self.config)
        return self._project_analyzer
    
    @property
    def metrics_analyzer(self) -> MetricsAnalyzer:
        """Get or create metrics analyzer instance."""
        if self._metrics_analyzer is None:
            self._metrics_analyzer = MetricsAnalyzer(self.config)
        return self._metrics_analyzer
    
    @property
    def refactoring_system(self) -> IntelligentRefactoringSystem:
        """Get or create intelligent refactoring system instance."""
        if self._refactoring_system is None:
            self._refactoring_system = IntelligentRefactoringSystem(
                config=self.config
            )
        return self._refactoring_system
    
    @property
    def safety_manager(self) -> SafetyManager:
        """Get or create safety manager instance."""
        if self._safety_manager is None:
            self._safety_manager = SafetyManager(self.config)
        return self._safety_manager
    
    @property
    def plugin_manager(self) -> PluginManager:
        """Get or create plugin manager instance."""
        if self._plugin_manager is None:
            self._plugin_manager = PluginManager(self.config.plugin_settings)
        return self._plugin_manager
    
    def initialize_project(self, project_path: Path) -> Dict[str, Any]:
        """Initialize project for analysis with all integrated systems."""
        logger.info(f"Initializing project: {project_path}")
        
        # Validate project path
        if not project_path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")
        
        if not project_path.is_dir():
            raise ValueError(f"Project path must be a directory: {project_path}")
        
        # Initialize knowledge base for project
        knowledge_status = self.knowledge_manager.initialize_project_knowledge(str(project_path))
        
        # Load plugins
        self.plugin_manager.load_all_plugins()
        
        # Initialize safety systems
        self.safety_manager.initialize_project(str(project_path))
        
        return {
            'project_path': str(project_path),
            'knowledge_status': knowledge_status,
            'plugins_loaded': len(self.plugin_manager.get_loaded_plugins()),
            'safety_initialized': True,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_comprehensive_analysis(self, project_path: Path, 
                                 include_metrics: bool = True,
                                 include_opportunities: bool = True,
                                 include_safety_check: bool = True) -> Dict[str, Any]:
        """Run comprehensive analysis using all integrated systems."""
        logger.info(f"Running comprehensive analysis for: {project_path}")
        
        results = {
            'project_path': str(project_path),
            'timestamp': datetime.now().isoformat(),
            'analysis_components': []
        }
        
        # Project structure analysis
        if include_metrics:
            logger.info("Running project analysis...")
            project_analysis = self.project_analyzer.analyze_project(str(project_path))
            results['project_analysis'] = project_analysis
            results['analysis_components'].append('project_structure')
        
        # Metrics analysis
        if include_metrics:
            logger.info("Running metrics analysis...")
            metrics = self.metrics_analyzer.analyze_project_metrics(str(project_path))
            results['metrics'] = metrics
            results['analysis_components'].append('metrics')
        
        # Refactoring opportunities
        if include_opportunities:
            logger.info("Identifying refactoring opportunities...")
            # Extract data from GenericAnalysisResult if it exists
            project_analysis_data = {}
            if 'project_analysis' in results:
                project_analysis = results['project_analysis']
                if hasattr(project_analysis, 'data') and project_analysis.data:
                    project_analysis_data = project_analysis.data
                elif hasattr(project_analysis, '__dict__'):
                    # Fallback: convert the object to dict
                    project_analysis_data = {
                        k: v for k, v in project_analysis.__dict__.items() 
                        if not k.startswith('_')
                    }
            
            opportunities = self.refactoring_system.identify_opportunities(project_analysis_data)
            results['opportunities'] = opportunities
            results['analysis_components'].append('opportunities')
        
        # Safety analysis
        if include_safety_check:
            logger.info("Running safety analysis...")
            safety_report = self.safety_manager.analyze_project_safety(str(project_path))
            results['safety_report'] = safety_report
            results['analysis_components'].append('safety')
        
        # Knowledge integration
        logger.info("Integrating with knowledge base...")
        # Use get_recommendations method instead of non-existent get_project_insights
        try:
            knowledge_insights = self.knowledge_manager.get_recommendations(str(project_path))
        except Exception as e:
            logger.warning(f"Knowledge integration failed: {e}")
            knowledge_insights = []
        results['knowledge_insights'] = knowledge_insights
        results['analysis_components'].append('knowledge')
        
        return results
    
    def execute_refactoring_plan(self, project_path: Path, plan: Dict[str, Any],
                               dry_run: bool = True, 
                               create_backup: bool = True) -> Dict[str, Any]:
        """Execute refactoring plan using orchestrator."""
        logger.info(f"Executing refactoring plan for: {project_path}")
        
        # Safety check before execution
        if not dry_run:
            safety_check = self.safety_manager.pre_refactoring_check(str(project_path), plan)
            if not safety_check['safe_to_proceed']:
                raise ValueError(f"Safety check failed: {safety_check['issues']}")
        
        # Create backup if requested
        backup_info = None
        if create_backup and not dry_run:
            backup_info = self.safety_manager.create_backup(str(project_path))
        
        # Execute through orchestrator
        execution_result = self.orchestrator.execute_refactoring_plan(
            project_path=str(project_path),
            plan=plan,
            dry_run=dry_run
        )
        
        # Validate results
        if not dry_run:
            validation_result = self.validator.validate_refactoring_result(
                str(project_path), 
                execution_result
            )
            execution_result['validation'] = validation_result
        
        # Update knowledge base
        self.knowledge_manager.record_refactoring_execution(
            str(project_path), 
            plan, 
            execution_result
        )
        
        return {
            'execution_result': execution_result,
            'backup_info': backup_info,
            'dry_run': dry_run,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_comprehensive_report(self, project_path: Path, 
                                    analysis_results: Dict[str, Any],
                                    output_format: str = 'markdown') -> str:
        """Generate comprehensive report using reporter."""
        logger.info(f"Generating comprehensive report for: {project_path}")
        
        return self.reporter.generate_comprehensive_report(
            project_path=str(project_path),
            analysis_results=analysis_results,
            format=output_format
        )
    
    def migrate_legacy_data(self, legacy_data_path: Path, 
                          target_format: str = 'modern') -> Dict[str, Any]:
        """Migrate legacy analysis data to modern format."""
        logger.info(f"Migrating legacy data from: {legacy_data_path}")
        
        if not legacy_data_path.exists():
            raise ValueError(f"Legacy data path does not exist: {legacy_data_path}")
        
        migration_results = {
            'source_path': str(legacy_data_path),
            'target_format': target_format,
            'migrated_items': [],
            'errors': [],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Load legacy data
            if legacy_data_path.suffix == '.json':
                with open(legacy_data_path, 'r') as f:
                    legacy_data = json.load(f)
            else:
                raise ValueError(f"Unsupported legacy data format: {legacy_data_path.suffix}")
            
            # Migrate different types of data
            if 'analysis_results' in legacy_data:
                migrated_analysis = self._migrate_analysis_results(legacy_data['analysis_results'])
                migration_results['migrated_items'].append({
                    'type': 'analysis_results',
                    'count': len(migrated_analysis),
                    'data': migrated_analysis
                })
            
            if 'refactoring_opportunities' in legacy_data:
                migrated_opportunities = self._migrate_refactoring_opportunities(
                    legacy_data['refactoring_opportunities']
                )
                migration_results['migrated_items'].append({
                    'type': 'refactoring_opportunities',
                    'count': len(migrated_opportunities),
                    'data': migrated_opportunities
                })
            
            if 'knowledge_items' in legacy_data:
                migrated_knowledge = self._migrate_knowledge_items(legacy_data['knowledge_items'])
                migration_results['migrated_items'].append({
                    'type': 'knowledge_items',
                    'count': len(migrated_knowledge),
                    'data': migrated_knowledge
                })
            
        except Exception as e:
            migration_results['errors'].append({
                'error': str(e),
                'type': type(e).__name__
            })
            logger.error(f"Migration error: {e}")
        
        return migration_results
    
    def _migrate_analysis_results(self, legacy_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Migrate legacy analysis results to modern format."""
        migrated = []
        for result in legacy_results:
            # Convert legacy format to modern format
            modern_result = {
                'id': result.get('id', f"migrated_{len(migrated)}"),
                'type': result.get('type', 'unknown'),
                'description': result.get('description', ''),
                'severity': result.get('severity', 'medium'),
                'file_path': result.get('file_path', ''),
                'line_number': result.get('line_number', 0),
                'metadata': result.get('metadata', {}),
                'migrated_from': 'legacy',
                'migration_timestamp': datetime.now().isoformat()
            }
            migrated.append(modern_result)
        return migrated
    
    def _migrate_refactoring_opportunities(self, legacy_opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Migrate legacy refactoring opportunities to modern format."""
        migrated = []
        for opportunity in legacy_opportunities:
            modern_opportunity = {
                'id': opportunity.get('id', f"migrated_opp_{len(migrated)}"),
                'title': opportunity.get('title', 'Migrated Opportunity'),
                'description': opportunity.get('description', ''),
                'type': opportunity.get('type', 'general'),
                'priority': opportunity.get('priority', 'medium'),
                'effort_estimate': opportunity.get('effort_estimate', 'unknown'),
                'impact_estimate': opportunity.get('impact_estimate', 'unknown'),
                'file_paths': opportunity.get('file_paths', []),
                'metadata': opportunity.get('metadata', {}),
                'migrated_from': 'legacy',
                'migration_timestamp': datetime.now().isoformat()
            }
            migrated.append(modern_opportunity)
        return migrated
    
    def _migrate_knowledge_items(self, legacy_knowledge: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Migrate legacy knowledge items to modern format."""
        migrated = []
        for item in legacy_knowledge:
            modern_item = {
                'id': item.get('id', f"migrated_knowledge_{len(migrated)}"),
                'type': item.get('type', 'general'),
                'content': item.get('content', ''),
                'tags': item.get('tags', []),
                'confidence': item.get('confidence', 0.5),
                'source': item.get('source', 'legacy_migration'),
                'metadata': item.get('metadata', {}),
                'migrated_from': 'legacy',
                'migration_timestamp': datetime.now().isoformat()
            }
            migrated.append(modern_item)
        return migrated
    
    def check_system_compatibility(self) -> Dict[str, Any]:
        """Check compatibility with existing systems."""
        compatibility_report = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'overall_status': 'compatible',
            'issues': []
        }
        
        # Check each component
        components = [
            ('knowledge_manager', self.knowledge_manager),
            ('orchestrator', self.orchestrator),
            ('reporter', self.reporter),
            ('validator', self.validator),
            ('project_analyzer', self.project_analyzer),
            ('metrics_analyzer', self.metrics_analyzer),
            ('refactoring_system', self.refactoring_system),
            ('safety_manager', self.safety_manager),
            ('plugin_manager', self.plugin_manager)
        ]
        
        for name, component in components:
            try:
                # Basic compatibility check
                if hasattr(component, 'check_compatibility'):
                    status = component.check_compatibility()
                else:
                    status = {'compatible': True, 'version': 'unknown'}
                
                compatibility_report['components'][name] = status
                
                if not status.get('compatible', True):
                    compatibility_report['overall_status'] = 'incompatible'
                    compatibility_report['issues'].append(f"{name}: {status.get('issue', 'Unknown compatibility issue')}")
                    
            except Exception as e:
                compatibility_report['components'][name] = {
                    'compatible': False,
                    'error': str(e)
                }
                compatibility_report['overall_status'] = 'error'
                compatibility_report['issues'].append(f"{name}: {str(e)}")
        
        return compatibility_report
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'safety_level': self.config.safety_level.value if hasattr(self.config, 'safety_level') else 'unknown',
                'plugins_enabled': getattr(self.config, 'plugins_enabled', True),
                'knowledge_base_path': getattr(self.config, 'knowledge_base_path', 'unknown')
            },
            'components': {
                'knowledge_manager': self._knowledge_manager is not None,
                'orchestrator': self._orchestrator is not None,
                'reporter': self._reporter is not None,
                'validator': self._validator is not None,
                'project_analyzer': self._project_analyzer is not None,
                'metrics_analyzer': self._metrics_analyzer is not None,
                'refactoring_system': self._refactoring_system is not None,
                'safety_manager': self._safety_manager is not None,
                'plugin_manager': self._plugin_manager is not None
            },
            'plugins': {
                'loaded': len(self.plugin_manager.get_loaded_plugins()) if self._plugin_manager else 0,
                'available': len(self.plugin_manager.get_available_plugins()) if self._plugin_manager else 0
            }
        }