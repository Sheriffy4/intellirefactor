"""
Golden Traces Extractor for expert refactoring analysis.

Extracts real usage examples and traces from production code
to understand actual usage patterns and create realistic tests.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class GoldenTracesExtractor:
    """Extracts real usage examples and execution traces."""

    def __init__(self, project_root: str, target_module: str):
        self.project_root = Path(project_root)
        self.target_module = Path(target_module)

    def extract_golden_traces(self, module_ast: ast.Module) -> Dict[str, Any]:
        """
        Extract golden traces and real usage examples.
        
        Args:
            module_ast: Parsed AST of the target module
            
        Returns:
            Dictionary with golden traces and usage examples
        """
        logger.info("Extracting golden traces and real usage examples...")
        
        # Extract real call sites from project
        real_call_sites = self._find_real_call_sites()
        
        # Extract parameter patterns
        parameter_patterns = self._extract_parameter_patterns(real_call_sites)
        
        # Extract typical workflows
        typical_workflows = self._extract_typical_workflows(real_call_sites)
        
        # Extract data examples
        data_examples = self._extract_data_examples(real_call_sites)
        
        # Generate instrumentation plan
        instrumentation_plan = self._generate_instrumentation_plan(module_ast)
        
        return {
            'real_call_sites': real_call_sites,
            'parameter_patterns': parameter_patterns,
            'typical_workflows': typical_workflows,
            'data_examples': data_examples,
            'instrumentation_plan': instrumentation_plan
        }

    def export_detailed_golden_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export detailed golden traces data as requested by experts.
        
        Returns:
            Dictionary with real usage examples and instrumentation data
        """
        # Organize real scenarios by method
        scenarios_by_method = {}
        for call_site in traces['real_call_sites']:
            method_name = call_site['method_name']
            if method_name not in scenarios_by_method:
                scenarios_by_method[method_name] = []
            
            scenarios_by_method[method_name].append({
                'file': call_site['file'],
                'line': call_site['line'],
                'context': call_site['context'],
                'parameters': call_site['parameters'],
                'usage_pattern': call_site['usage_pattern']
            })
        
        # Create realistic test data
        realistic_test_data = self._create_realistic_test_data(traces['data_examples'])
        
        # Generate production scenarios
        production_scenarios = []
        for workflow in traces['typical_workflows']:
            scenario = {
                'name': workflow['name'],
                'description': workflow['description'],
                'steps': workflow['steps'],
                'input_examples': workflow['input_examples'],
                'expected_outputs': workflow['expected_outputs'],
                'test_implementation': self._generate_scenario_test(workflow)
            }
            production_scenarios.append(scenario)
        
        return {
            'real_usage_scenarios': scenarios_by_method,
            'realistic_test_data': realistic_test_data,
            'production_scenarios': production_scenarios,
            'instrumentation_guide': self._create_instrumentation_guide(traces['instrumentation_plan']),
            'golden_examples': self._create_golden_examples(traces),
            'recommendations': self._generate_golden_trace_recommendations(traces)
        }

    def _find_real_call_sites(self) -> List[Dict[str, Any]]:
        """Find real call sites of the target module in the project."""
        call_sites = []
        
        # Get target module name
        target_name = self.target_module.stem
        
        # Search all Python files for usage
        for py_file in self.project_root.rglob("*.py"):
            if py_file == self.target_module:
                continue
            
            try:
                call_sites.extend(self._analyze_file_for_call_sites(py_file, target_name))
            except Exception as e:
                logger.warning(f"Error analyzing {py_file}: {e}")
        
        return call_sites

    def _analyze_file_for_call_sites(self, py_file: Path, target_name: str) -> List[Dict[str, Any]]:
        """Analyze a single file for call sites."""
        call_sites = []
        
        try:
            content = py_file.read_text(encoding='utf-8')
            
            # Quick check if file references target
            if target_name not in content:
                return call_sites
            
            tree = ast.parse(content)
            
            # Find function calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    call_info = self._extract_call_info(node, py_file)
                    if call_info and self._is_target_call(call_info, target_name):
                        call_sites.append(call_info)
        
        except (OSError, SyntaxError) as e:
            logger.warning(f"Could not parse {py_file}: {e}")
        
        return call_sites

    def _extract_call_info(self, call_node: ast.Call, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract information from a function call."""
        try:
            # Get method name
            method_name = self._get_call_name(call_node)
            if not method_name:
                return None
            
            # Extract parameters
            parameters = self._extract_call_parameters(call_node)
            
            # Get context
            context = self._get_call_context(call_node)
            
            # Determine usage pattern
            usage_pattern = self._determine_usage_pattern(call_node, context)
            
            return {
                'file': str(file_path.relative_to(self.project_root)),
                'line': getattr(call_node, 'lineno', 0),
                'method_name': method_name,
                'parameters': parameters,
                'context': context,
                'usage_pattern': usage_pattern
            }
        
        except Exception as e:
            logger.warning(f"Error extracting call info: {e}")
            return None

    def _get_call_name(self, call_node: ast.Call) -> Optional[str]:
        """Get the name of the called function/method."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            if isinstance(call_node.func.value, ast.Name):
                return f"{call_node.func.value.id}.{call_node.func.attr}"
            else:
                return call_node.func.attr
        return None

    def _extract_call_parameters(self, call_node: ast.Call) -> Dict[str, Any]:
        """Extract parameters from a function call."""
        parameters = {}
        
        # Positional arguments
        for i, arg in enumerate(call_node.args):
            param_value = self._extract_parameter_value(arg)
            parameters[f'arg_{i}'] = param_value
        
        # Keyword arguments
        for keyword in call_node.keywords:
            param_value = self._extract_parameter_value(keyword.value)
            parameters[keyword.arg] = param_value
        
        return parameters

    def _extract_parameter_value(self, value_node: ast.expr) -> Any:
        """Extract the value of a parameter."""
        if isinstance(value_node, ast.Constant):
            return value_node.value
        elif isinstance(value_node, ast.Name):
            return f"<variable:{value_node.id}>"
        elif isinstance(value_node, ast.List):
            return [self._extract_parameter_value(item) for item in value_node.elts]
        elif isinstance(value_node, ast.Dict):
            result = {}
            for key, value in zip(value_node.keys, value_node.values):
                key_val = self._extract_parameter_value(key) if key else None
                val_val = self._extract_parameter_value(value)
                result[key_val] = val_val
            return result
        elif isinstance(value_node, ast.Str):  # For older Python versions
            return value_node.s
        else:
            try:
                if hasattr(ast, 'unparse'):
                    return f"<expr:{ast.unparse(value_node)}>"
                else:
                    return "<expr:complex>"
            except Exception:
                return "<expr:unknown>"

    def _get_call_context(self, call_node: ast.Call) -> str:
        """Get context information about the call."""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(call_node)
            else:
                return f"call at line {getattr(call_node, 'lineno', 0)}"
        except Exception:
            return "call context unknown"

    def _determine_usage_pattern(self, call_node: ast.Call, context: str) -> str:
        """Determine the usage pattern of the call."""
        if not context:
            return 'general_usage'
            
        context_lower = context.lower()
        
        if 'dispatch_attack' in context_lower:
            return 'attack_dispatch'
        elif 'fake' in context_lower and 'disorder' in context_lower:
            return 'fake_disorder_attack'
        elif 'split' in context_lower:
            return 'split_attack'
        elif 'sni' in context_lower:
            return 'sni_processing'
        elif 'normalize' in context_lower:
            return 'parameter_normalization'
        elif 'log' in context_lower:
            return 'logging_operation'
        else:
            return 'general_usage'

    def _is_target_call(self, call_info: Dict[str, Any], target_name: str) -> bool:
        """Check if this call is related to our target module."""
        method_name = call_info['method_name']
        
        # Direct method calls
        if target_name in method_name:
            return True
        
        # Common method names from the target domain
        target_methods = [
            'dispatch_attack', 'normalize_parameters', 'parse_sni',
            'extract_sni', 'build_recipe', 'execute_attack'
        ]
        
        return any(target_method in method_name for target_method in target_methods)

    def _extract_parameter_patterns(self, call_sites: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common parameter patterns from call sites."""
        patterns = {}
        
        # Group by method
        by_method = {}
        for call_site in call_sites:
            method = call_site['method_name']
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(call_site['parameters'])
        
        # Analyze patterns for each method
        for method, param_lists in by_method.items():
            method_patterns = self._analyze_method_parameters(param_lists)
            patterns[method] = method_patterns
        
        return patterns

    def _analyze_method_parameters(self, param_lists: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze parameter patterns for a specific method."""
        analysis = {
            'total_calls': len(param_lists),
            'parameter_frequency': {},
            'common_values': {},
            'parameter_types': {},
            'typical_combinations': []
        }
        
        # Count parameter frequency
        all_params = set()
        for params in param_lists:
            all_params.update(params.keys())
        
        for param in all_params:
            count = sum(1 for params in param_lists if param in params)
            analysis['parameter_frequency'][param] = count
        
        # Find common values
        for param in all_params:
            values = [params.get(param) for params in param_lists if param in params]
            value_counts = {}
            for value in values:
                str_value = str(value)
                value_counts[str_value] = value_counts.get(str_value, 0) + 1
            
            # Get most common values
            sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
            analysis['common_values'][param] = sorted_values[:5]  # Top 5
        
        # Find typical combinations (first few examples)
        analysis['typical_combinations'] = param_lists[:5]
        
        return analysis

    def _extract_typical_workflows(self, call_sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract typical workflows from call sites."""
        workflows = []
        
        # Group call sites by file to find workflows
        by_file = {}
        for call_site in call_sites:
            file_path = call_site['file']
            if file_path not in by_file:
                by_file[file_path] = []
            by_file[file_path].append(call_site)
        
        # Analyze each file for workflows
        for file_path, file_calls in by_file.items():
            if len(file_calls) > 1:  # Multiple calls suggest a workflow
                workflow = self._analyze_file_workflow(file_path, file_calls)
                if workflow:
                    workflows.append(workflow)
        
        return workflows

    def _analyze_file_workflow(self, file_path: str, calls: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Analyze a workflow from calls in a single file."""
        # Sort calls by line number
        sorted_calls = sorted(calls, key=lambda x: x['line'])
        
        # Extract workflow pattern
        workflow_name = self._generate_workflow_name(file_path, sorted_calls)
        
        # Extract steps
        steps = []
        for i, call in enumerate(sorted_calls):
            step = {
                'step_number': i + 1,
                'method': call['method_name'],
                'parameters': call['parameters'],
                'line': call['line']
            }
            steps.append(step)
        
        # Generate input/output examples
        input_examples = self._extract_workflow_inputs(sorted_calls)
        expected_outputs = self._extract_workflow_outputs(sorted_calls)
        
        return {
            'name': workflow_name,
            'description': f"Workflow from {file_path}",
            'file': file_path,
            'steps': steps,
            'input_examples': input_examples,
            'expected_outputs': expected_outputs
        }

    def _generate_workflow_name(self, file_path: str, calls: List[Dict[str, Any]]) -> str:
        """Generate a name for a workflow."""
        file_name = Path(file_path).stem
        
        # Look for common patterns
        method_names = [call['method_name'] for call in calls]
        
        if any('dispatch' in method for method in method_names):
            return f"{file_name}_dispatch_workflow"
        elif any('attack' in method for method in method_names):
            return f"{file_name}_attack_workflow"
        else:
            return f"{file_name}_workflow"

    def _extract_workflow_inputs(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract typical inputs for a workflow."""
        inputs = []
        
        for call in calls:
            if call['parameters']:
                # Extract meaningful parameters
                meaningful_params = {}
                for param, value in call['parameters'].items():
                    if not str(value).startswith('<variable:'):  # Skip variable references
                        meaningful_params[param] = value
                
                if meaningful_params:
                    inputs.append({
                        'method': call['method_name'],
                        'parameters': meaningful_params
                    })
        
        return inputs

    def _extract_workflow_outputs(self, calls: List[Dict[str, Any]]) -> List[str]:
        """Extract expected outputs for a workflow."""
        # This is a simplified version - in practice you'd need more sophisticated analysis
        outputs = []
        
        for call in calls:
            method = call['method_name']
            if 'dispatch' in method:
                outputs.append("Attack recipe with segments")
            elif 'normalize' in method:
                outputs.append("Normalized parameters dictionary")
            elif 'parse' in method:
                outputs.append("Parsed data structure")
            else:
                outputs.append(f"Result from {method}")
        
        return outputs

    def _extract_data_examples(self, call_sites: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract real data examples from call sites."""
        examples = {
            'task_types': set(),
            'parameter_combinations': [],
            'payload_patterns': [],
            'packet_info_examples': []
        }
        
        for call_site in call_sites:
            params = call_site['parameters']
            
            # Extract task types
            for param_name, param_value in params.items():
                if param_name and 'task' in param_name.lower() and isinstance(param_value, str):
                    examples['task_types'].add(param_value)
            
            # Extract parameter combinations
            if len(params) > 1:
                examples['parameter_combinations'].append(params)
            
            # Extract payload patterns
            for param_name, param_value in params.items():
                if param_name and 'payload' in param_name.lower() and param_value:
                    examples['payload_patterns'].append(str(param_value)[:100])  # First 100 chars
            
            # Extract packet info examples
            for param_name, param_value in params.items():
                if param_name and 'packet' in param_name.lower() and isinstance(param_value, dict):
                    examples['packet_info_examples'].append(param_value)
        
        # Convert sets to lists for JSON serialization
        examples['task_types'] = list(examples['task_types'])
        
        return examples

    def _generate_instrumentation_plan(self, module_ast: ast.Module) -> Dict[str, Any]:
        """Generate a plan for instrumenting the code to capture traces."""
        plan = {
            'instrumentation_points': [],
            'data_capture_methods': [],
            'trace_format': {},
            'setup_instructions': []
        }
        
        # Find key methods to instrument
        for node in ast.walk(module_ast):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith('_'):  # Public methods
                    instrumentation_point = {
                        'method': node.name,
                        'line': getattr(node, 'lineno', 0),
                        'capture_inputs': True,
                        'capture_outputs': True,
                        'capture_exceptions': True
                    }
                    plan['instrumentation_points'].append(instrumentation_point)
        
        # Data capture methods
        plan['data_capture_methods'] = [
            'function_entry_exit_logging',
            'parameter_value_capture',
            'return_value_capture',
            'exception_capture',
            'execution_time_measurement'
        ]
        
        # Trace format
        plan['trace_format'] = {
            'timestamp': 'ISO format',
            'method_name': 'string',
            'input_parameters': 'dict',
            'output_value': 'any',
            'execution_time_ms': 'float',
            'exception': 'string or null',
            'call_stack': 'list of strings'
        }
        
        # Setup instructions
        plan['setup_instructions'] = [
            "Add logging decorators to target methods",
            "Configure trace output format",
            "Set up trace file rotation",
            "Run representative test scenarios",
            "Collect and analyze trace data"
        ]
        
        return plan

    def _create_realistic_test_data(self, data_examples: Dict[str, Any]) -> Dict[str, Any]:
        """Create realistic test data from extracted examples."""
        test_data = {}
        
        # Create task type test data
        if data_examples['task_types']:
            test_data['task_types'] = {
                'common_tasks': list(data_examples['task_types'])[:10],
                'test_cases': [
                    {'input': task, 'expected_type': 'string'}
                    for task in list(data_examples['task_types'])[:5]
                ]
            }
        
        # Create parameter test data
        if data_examples['parameter_combinations']:
            test_data['parameters'] = {
                'realistic_combinations': data_examples['parameter_combinations'][:10],
                'test_cases': [
                    {'input': combo, 'description': f'Real parameter combination {i+1}'}
                    for i, combo in enumerate(data_examples['parameter_combinations'][:5])
                ]
            }
        
        # Create packet info test data
        if data_examples['packet_info_examples']:
            test_data['packet_info'] = {
                'real_examples': data_examples['packet_info_examples'][:10],
                'test_cases': [
                    {'input': packet, 'description': f'Real packet info {i+1}'}
                    for i, packet in enumerate(data_examples['packet_info_examples'][:5])
                ]
            }
        
        return test_data

    def _generate_scenario_test(self, workflow: Dict[str, Any]) -> str:
        """Generate test implementation for a workflow scenario."""
        lines = [
            f"def test_{workflow['name']}(self):",
            f'    """Test workflow: {workflow["description"]}"""',
            ""
        ]
        
        # Add setup
        lines.extend([
            "    # Setup",
            "    # TODO: Initialize required objects and mocks",
            ""
        ])
        
        # Add workflow steps
        lines.append("    # Execute workflow steps")
        for i, step in enumerate(workflow['steps']):
            lines.append(f"    # Step {step['step_number']}: {step['method']}")
            if step['parameters']:
                param_str = ', '.join([f'{k}={repr(v)}' for k, v in step['parameters'].items() if not str(v).startswith('<')])
                if param_str:
                    lines.append(f"    result_{i} = {step['method']}({param_str})")
                else:
                    lines.append(f"    result_{i} = {step['method']}()")
            else:
                lines.append(f"    result_{i} = {step['method']}()")
            lines.append("")
        
        # Add assertions
        lines.extend([
            "    # Verify results",
            "    # TODO: Add specific assertions based on expected outputs",
            "    assert True  # Placeholder"
        ])
        
        return '\n'.join(lines)

    def _create_instrumentation_guide(self, instrumentation_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Create a guide for instrumenting the code."""
        return {
            'overview': 'Guide for instrumenting code to capture golden traces',
            'steps': [
                'Identify key methods to instrument',
                'Add logging decorators or manual logging',
                'Configure trace output format',
                'Run production scenarios',
                'Collect and analyze traces'
            ],
            'instrumentation_points': len(instrumentation_plan['instrumentation_points']),
            'recommended_tools': [
                'Python logging module',
                'functools.wraps for decorators',
                'JSON for trace serialization',
                'Custom trace analysis scripts'
            ],
            'trace_analysis': [
                'Parameter frequency analysis',
                'Execution path analysis',
                'Performance pattern analysis',
                'Error pattern analysis'
            ]
        }

    def _create_golden_examples(self, traces: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create golden examples from traces."""
        examples = []
        
        # Create examples from real call sites
        for call_site in traces['real_call_sites'][:10]:  # Top 10 examples
            example = {
                'name': f"golden_example_{call_site['method_name']}",
                'description': f"Real usage from {call_site['file']}:{call_site['line']}",
                'method': call_site['method_name'],
                'input': call_site['parameters'],
                'context': call_site['context'],
                'usage_pattern': call_site['usage_pattern'],
                'source_location': f"{call_site['file']}:{call_site['line']}"
            }
            examples.append(example)
        
        return examples

    def _generate_golden_trace_recommendations(self, traces: Dict[str, Any]) -> List[str]:
        """Generate recommendations for using golden traces."""
        recommendations = []
        
        call_sites_count = len(traces['real_call_sites'])
        workflows_count = len(traces['typical_workflows'])
        
        if call_sites_count > 0:
            recommendations.append(f"Found {call_sites_count} real usage examples - use these for realistic testing")
        
        if workflows_count > 0:
            recommendations.append(f"Identified {workflows_count} typical workflows - implement these as integration tests")
        
        if traces['parameter_patterns']:
            recommendations.append("Use extracted parameter patterns to create realistic test data")
        
        if traces['data_examples']['task_types']:
            task_count = len(traces['data_examples']['task_types'])
            recommendations.append(f"Found {task_count} real task types - ensure all are covered in tests")
        
        # Instrumentation recommendations
        instrumentation_points = len(traces['instrumentation_plan']['instrumentation_points'])
        if instrumentation_points > 0:
            recommendations.append(f"Instrument {instrumentation_points} methods to capture production traces")
        
        recommendations.extend([
            "Run instrumented code with production workloads to capture golden traces",
            "Use golden traces to validate refactored code behavior",
            "Create regression tests based on real usage patterns"
        ])
        
        return recommendations