"""
Example demonstrating how to use the IntelliRefactor plugin system with custom hooks.

This example shows how to:
1. Load and configure plugins
2. Use the hook system for extensibility
3. Coordinate multiple plugins working together
4. Handle plugin lifecycle and error management
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add the parent directory to the path so we can import the plugins
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from intellirefactor.plugins.plugin_manager import PluginManager
from intellirefactor.plugins.hook_system import HookSystem, HookType, HookPriority
from intellirefactor.config import PluginConfig
from intellirefactor.plugins.examples.example_custom_rules_plugin import CustomRulesPlugin
from intellirefactor.plugins.examples.example_refactoring_plugin import ExampleRefactoringPlugin
from intellirefactor.plugins.examples.example_knowledge_plugin import ExampleKnowledgePlugin

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PluginUsageExample:
    """
    Example class demonstrating comprehensive plugin usage.
    """
    
    def __init__(self):
        self.hook_system = HookSystem()
        self.plugin_manager = None
        self.plugins = {}
    
    def setup_plugins(self) -> bool:
        """Set up and initialize all example plugins."""
        try:
            logger.info("Setting up example plugins...")
            
            # Create plugin configurations
            custom_rules_config = {
                'max_parameters': 4,
                'max_complexity': 8,
                'naming_conventions': {
                    'class_pattern': r'^[A-Z][a-zA-Z0-9]*$',
                    'function_pattern': r'^[a-z_][a-z0-9_]*$',
                    'constant_pattern': r'^[A-Z_][A-Z0-9_]*$'
                }
            }
            
            refactoring_config = {
                'min_string_length': 8,
                'enable_fstring_conversion': True,
                'enable_boolean_simplification': True,
                'min_duplicate_lines': 3
            }
            
            knowledge_config = {
                'knowledge_db_path': 'example_knowledge.json',
                'min_confidence': 0.6,
                'max_knowledge_items': 500,
                'enable_pattern_learning': True
            }
            
            # Initialize plugins
            self.plugins['custom_rules'] = CustomRulesPlugin(custom_rules_config)
            self.plugins['refactoring'] = ExampleRefactoringPlugin(refactoring_config)
            self.plugins['knowledge'] = ExampleKnowledgePlugin(knowledge_config)
            
            # Inject hook system into plugins
            for plugin in self.plugins.values():
                plugin._hook_system = self.hook_system
            
            # Initialize all plugins
            for name, plugin in self.plugins.items():
                if not plugin.initialize():
                    logger.error(f"Failed to initialize plugin: {name}")
                    return False
                logger.info(f"Initialized plugin: {name}")
            
            # Register additional coordination hooks
            self._register_coordination_hooks()
            
            logger.info("All plugins initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up plugins: {e}")
            return False
    
    def _register_coordination_hooks(self) -> None:
        """Register hooks that coordinate between plugins."""
        # Hook to coordinate analysis between plugins
        self.hook_system.register_hook(
            hook_type=HookType.POST_PROJECT_ANALYSIS,
            callback=self._coordinate_analysis_results,
            name="coordinate_analysis",
            priority=HookPriority.LOW,  # Run after individual plugin analysis
            plugin_name="coordinator",
            description="Coordinate analysis results between plugins"
        )
        
        # Hook to share knowledge between plugins
        self.hook_system.register_hook(
            hook_type=HookType.POST_KNOWLEDGE_EXTRACTION,
            callback=self._share_knowledge,
            name="share_knowledge",
            priority=HookPriority.LOW,
            plugin_name="coordinator",
            description="Share knowledge between plugins"
        )
        
        # Hook to validate refactoring opportunities across plugins
        self.hook_system.register_hook(
            hook_type=HookType.POST_OPPORTUNITY_DETECTION,
            callback=self._validate_opportunities,
            name="validate_opportunities",
            priority=HookPriority.HIGH,
            plugin_name="coordinator",
            description="Validate refactoring opportunities across plugins"
        )
    
    def _coordinate_analysis_results(self, project_path: str, analysis_results: Dict[str, Any]) -> None:
        """Coordinate analysis results between plugins."""
        logger.info("Coordinating analysis results between plugins")
        
        # Merge custom analysis results
        merged_issues = []
        merged_metrics = {}
        
        for file_path, file_results in analysis_results.get('files', {}).items():
            # Collect issues from all plugins
            if 'custom_rules' in file_results:
                custom_issues = file_results['custom_rules'].get('issues', [])
                merged_issues.extend(custom_issues)
            
            # Merge metrics
            if 'metrics' in file_results:
                for metric_name, metric_value in file_results['metrics'].items():
                    if metric_name not in merged_metrics:
                        merged_metrics[metric_name] = []
                    merged_metrics[metric_name].append(metric_value)
        
        # Add coordinated results
        analysis_results['coordinated'] = {
            'total_issues': len(merged_issues),
            'issues_by_severity': self._group_issues_by_severity(merged_issues),
            'average_metrics': {name: sum(values) / len(values) for name, values in merged_metrics.items()},
            'plugin_coordination': True
        }
    
    def _group_issues_by_severity(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group issues by severity level."""
        severity_counts = {'error': 0, 'warning': 0, 'info': 0}
        
        for issue in issues:
            severity = issue.get('severity', 'info')
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        return severity_counts
    
    def _share_knowledge(self, analysis_results: Dict[str, Any], 
                        refactoring_results: List[Dict[str, Any]], 
                        knowledge_items: List[Dict[str, Any]]) -> None:
        """Share knowledge between plugins."""
        logger.info("Sharing knowledge between plugins")
        
        # Get knowledge plugin
        knowledge_plugin = self.plugins.get('knowledge')
        if not knowledge_plugin:
            return
        
        # Generate recommendations based on current analysis
        recommendations = self.hook_system.execute_custom_hooks(
            'generate_recommendations', 
            analysis_results.get('project_path', ''), 
            analysis_results
        )
        
        # Share recommendations with other plugins through context
        if recommendations:
            self.hook_system.set_context('shared_recommendations', recommendations[0] if recommendations else {})
    
    def _validate_opportunities(self, analysis_results: Dict[str, Any], 
                               opportunities: List[Dict[str, Any]]) -> None:
        """Validate refactoring opportunities across plugins."""
        logger.info(f"Validating {len(opportunities)} refactoring opportunities")
        
        # Get shared recommendations
        recommendations = self.hook_system.get_context('shared_recommendations', {})
        risk_warnings = recommendations.get('risk_warnings', [])
        
        # Mark risky opportunities
        for opportunity in opportunities:
            opportunity_type = opportunity.get('type', '')
            
            # Check against known risky patterns
            for warning in risk_warnings:
                if warning.get('type') == opportunity_type:
                    opportunity['risk_level'] = warning.get('risk_level', 'medium')
                    opportunity['risk_reason'] = warning.get('warning', 'Historical low success rate')
                    logger.warning(f"Marked opportunity as risky: {opportunity_type}")
    
    def analyze_project(self, project_path: str) -> Dict[str, Any]:
        """Analyze a project using all plugins."""
        logger.info(f"Analyzing project: {project_path}")
        
        # Prepare analysis context
        context = {
            'project_path': project_path,
            'start_time': 'now',
            'plugins_used': list(self.plugins.keys())
        }
        
        # Execute pre-analysis hooks
        self.hook_system.execute_hooks(HookType.PRE_PROJECT_ANALYSIS, project_path, context)
        
        # Run analysis with each plugin
        analysis_results = {
            'project_path': project_path,
            'files': {},
            'plugins': {}
        }
        
        # Custom rules analysis
        if 'custom_rules' in self.plugins:
            custom_results = self.plugins['custom_rules'].analyze_project(project_path, context)
            analysis_results['plugins']['custom_rules'] = custom_results
            
            # Merge file-level results
            for file_path, file_result in custom_results.get('files', {}).items():
                if file_path not in analysis_results['files']:
                    analysis_results['files'][file_path] = {}
                analysis_results['files'][file_path].update(file_result)
        
        # Execute post-analysis hooks
        self.hook_system.execute_hooks(HookType.POST_PROJECT_ANALYSIS, project_path, analysis_results)
        
        logger.info(f"Analysis complete. Analyzed {len(analysis_results['files'])} files")
        return analysis_results
    
    def identify_refactoring_opportunities(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify refactoring opportunities using all plugins."""
        logger.info("Identifying refactoring opportunities")
        
        # Execute pre-opportunity detection hooks
        self.hook_system.execute_hooks(HookType.PRE_OPPORTUNITY_DETECTION, analysis_results)
        
        all_opportunities = []
        
        # Get opportunities from refactoring plugin
        if 'refactoring' in self.plugins:
            opportunities = self.plugins['refactoring'].identify_opportunities(analysis_results)
            all_opportunities.extend(opportunities)
            logger.info(f"Found {len(opportunities)} opportunities from refactoring plugin")
        
        # Execute post-opportunity detection hooks (includes validation)
        self.hook_system.execute_hooks(HookType.POST_OPPORTUNITY_DETECTION, analysis_results, all_opportunities)
        
        logger.info(f"Total opportunities identified: {len(all_opportunities)}")
        return all_opportunities
    
    def apply_refactoring(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a refactoring using the appropriate plugin."""
        logger.info(f"Applying refactoring: {opportunity.get('type', 'unknown')}")
        
        context = {'opportunity': opportunity}
        
        # Execute pre-refactoring hooks
        self.hook_system.execute_hooks(HookType.PRE_REFACTORING, opportunity, context)
        
        # Apply refactoring using the refactoring plugin
        result = {'success': False, 'error': 'No suitable plugin found'}
        
        if 'refactoring' in self.plugins:
            result = self.plugins['refactoring'].apply_refactoring(opportunity, context)
        
        # Execute post-refactoring hooks (includes learning)
        self.hook_system.execute_hooks(HookType.POST_REFACTORING, opportunity, context, result)
        
        logger.info(f"Refactoring result: {'success' if result.get('success') else 'failed'}")
        return result
    
    def extract_knowledge(self, analysis_results: Dict[str, Any], 
                         refactoring_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract knowledge using the knowledge plugin."""
        logger.info("Extracting knowledge from results")
        
        knowledge_items = []
        
        if 'knowledge' in self.plugins:
            knowledge_items = self.plugins['knowledge'].extract_knowledge(
                analysis_results, refactoring_results
            )
        
        logger.info(f"Extracted {len(knowledge_items)} knowledge items")
        return knowledge_items
    
    def get_recommendations(self, project_path: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations from the knowledge plugin."""
        logger.info("Getting recommendations")
        
        recommendations = {}
        
        if 'knowledge' in self.plugins:
            rec_results = self.hook_system.execute_custom_hooks(
                'generate_recommendations', project_path, analysis_results
            )
            if rec_results:
                recommendations = rec_results[0] or {}
        
        return recommendations
    
    def run_complete_workflow(self, project_path: str) -> Dict[str, Any]:
        """Run a complete analysis and refactoring workflow."""
        logger.info(f"Running complete workflow for: {project_path}")
        
        workflow_results = {
            'project_path': project_path,
            'analysis': {},
            'opportunities': [],
            'refactoring_results': [],
            'knowledge': [],
            'recommendations': {},
            'summary': {}
        }
        
        try:
            # Step 1: Analyze project
            analysis_results = self.analyze_project(project_path)
            workflow_results['analysis'] = analysis_results
            
            # Step 2: Identify opportunities
            opportunities = self.identify_refactoring_opportunities(analysis_results)
            workflow_results['opportunities'] = opportunities
            
            # Step 3: Apply selected refactorings (for demo, apply first 2)
            refactoring_results = []
            for opportunity in opportunities[:2]:  # Limit for demo
                result = self.apply_refactoring(opportunity)
                refactoring_results.append(result)
            workflow_results['refactoring_results'] = refactoring_results
            
            # Step 4: Extract knowledge
            knowledge_items = self.extract_knowledge(analysis_results, refactoring_results)
            workflow_results['knowledge'] = knowledge_items
            
            # Step 5: Get recommendations
            recommendations = self.get_recommendations(project_path, analysis_results)
            workflow_results['recommendations'] = recommendations
            
            # Step 6: Generate summary
            workflow_results['summary'] = self._generate_workflow_summary(workflow_results)
            
            logger.info("Complete workflow finished successfully")
            
        except Exception as e:
            logger.error(f"Error in workflow: {e}")
            workflow_results['error'] = str(e)
        
        return workflow_results
    
    def _generate_workflow_summary(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the workflow results."""
        analysis = workflow_results.get('analysis', {})
        opportunities = workflow_results.get('opportunities', [])
        refactoring_results = workflow_results.get('refactoring_results', [])
        knowledge = workflow_results.get('knowledge', [])
        
        successful_refactorings = sum(1 for r in refactoring_results if r.get('success', False))
        
        return {
            'files_analyzed': len(analysis.get('files', {})),
            'total_issues': analysis.get('coordinated', {}).get('total_issues', 0),
            'opportunities_found': len(opportunities),
            'refactorings_attempted': len(refactoring_results),
            'refactorings_successful': successful_refactorings,
            'success_rate': successful_refactorings / len(refactoring_results) if refactoring_results else 0,
            'knowledge_items_extracted': len(knowledge),
            'plugins_used': list(self.plugins.keys()),
            'hooks_executed': len(self.hook_system.registry.list_hooks())
        }
    
    def cleanup(self) -> None:
        """Clean up all plugins and resources."""
        logger.info("Cleaning up plugins and resources")
        
        # Cleanup all plugins
        for name, plugin in self.plugins.items():
            try:
                plugin.cleanup()
                logger.info(f"Cleaned up plugin: {name}")
            except Exception as e:
                logger.error(f"Error cleaning up plugin {name}: {e}")
        
        # Cleanup hook system
        self.hook_system.cleanup()
        
        logger.info("Cleanup complete")


def main():
    """Main function demonstrating plugin usage."""
    # Create example instance
    example = PluginUsageExample()
    
    try:
        # Setup plugins
        if not example.setup_plugins():
            logger.error("Failed to setup plugins")
            return
        
        # Get hook system status
        hook_status = example.hook_system.get_hook_status()
        logger.info(f"Hook system status: {hook_status}")
        
        # For demonstration, we'll analyze the current directory
        current_dir = os.getcwd()
        logger.info(f"Running workflow on current directory: {current_dir}")
        
        # Run complete workflow
        results = example.run_complete_workflow(current_dir)
        
        # Print summary
        summary = results.get('summary', {})
        logger.info("=== Workflow Summary ===")
        for key, value in summary.items():
            logger.info(f"{key}: {value}")
        
        # Print some detailed results
        if results.get('opportunities'):
            logger.info("=== Refactoring Opportunities ===")
            for i, opp in enumerate(results['opportunities'][:3]):  # Show first 3
                logger.info(f"{i+1}. {opp.get('type', 'unknown')}: {opp.get('description', 'No description')}")
        
        if results.get('recommendations'):
            recs = results['recommendations']
            if recs.get('refactoring_suggestions'):
                logger.info("=== Recommendations ===")
                for rec in recs['refactoring_suggestions'][:3]:  # Show first 3
                    logger.info(f"- {rec.get('recommendation', 'No recommendation')} (confidence: {rec.get('confidence', 0):.1%})")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
    
    finally:
        # Always cleanup
        example.cleanup()


if __name__ == "__main__":
    main()