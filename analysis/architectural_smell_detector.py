import ast
import builtins
from typing import List, Dict, Set, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

# --- Constants ---
BUILTINS = set(dir(builtins))

# --- Models ---

@dataclass
class Evidence:
    description: str
    confidence: float
    code_snippets: List[str]
    metadata: Dict[str, Any]

class SmellType(Enum):
    GOD_CLASS = "god_class"
    LONG_METHOD = "long_method"
    HIGH_COMPLEXITY = "high_complexity"
    SRP_VIOLATION = "srp_violation"
    FEATURE_ENVY = "feature_envy"
    INAPPROPRIATE_INTIMACY = "inappropriate_intimacy"

class SmellSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class SmellThresholds:
    # God Class
    god_class_methods: int = 15
    god_class_responsibilities: int = 3
    god_class_cohesion: float = 0.5
    god_class_lines: int = 200
    # Long Method
    long_method_lines: int = 30
    long_method_statements: int = 20
    # High Complexity
    high_complexity_cyclomatic: int = 10
    high_complexity_nesting: int = 4
    # SRP Violation
    srp_responsibility_keywords: int = 3
    srp_external_dependencies: int = 5
    # Feature Envy
    feature_envy_external_ratio: float = 0.7
    feature_envy_min_accesses: int = 3
    # Inappropriate Intimacy
    intimacy_private_access: int = 3
    intimacy_coupling_ratio: float = 0.8

@dataclass
class ResponsibilityMarker:
    keyword: str
    context: str
    line_number: int
    confidence: float

@dataclass
class ArchitecturalSmell:
    smell_type: SmellType
    severity: SmellSeverity
    confidence: float
    file_path: str
    symbol_name: str
    line_start: int
    line_end: int
    metrics: Dict[str, Any]
    evidence: Evidence
    recommendations: List[str]
    decomposition_strategy: Optional[str] = None
    extraction_candidates: List[str] = field(default_factory=list)
    responsibility_markers: List[ResponsibilityMarker] = field(default_factory=list)

# --- Detector ---

class ArchitecturalSmellDetector:
    """
    Detects architectural code smells with formal criteria and evidence collection.
    """
    
    # Optimization: Define node types at class level
    _STOP_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    
    # Define nesting nodes, dynamically adding Match for Python 3.10+
    _NESTING_NODES = (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try)
    if hasattr(ast, 'Match'):
        _NESTING_NODES += (ast.Match,)
    
    def __init__(self, thresholds: Optional[SmellThresholds] = None):
        self.thresholds = thresholds or SmellThresholds()
        self.responsibility_keywords = {
            'data': ['load', 'save', 'read', 'write', 'parse', 'serialize', 'deserialize'],
            'validation': ['validate', 'check', 'verify', 'ensure', 'assert'],
            'calculation': ['calculate', 'compute', 'process', 'transform', 'convert'],
            'communication': ['send', 'receive', 'request', 'response', 'notify'],
            'formatting': ['format', 'render', 'display', 'print', 'show'],
            'logging': ['log', 'debug', 'info', 'warn', 'error'],
            'configuration': ['config', 'setting', 'option', 'parameter'],
            'caching': ['cache', 'store', 'retrieve', 'invalidate'],
            'security': ['authenticate', 'authorize', 'encrypt', 'decrypt', 'hash'],
            'workflow': ['start', 'stop', 'pause', 'resume', 'execute', 'run']
        }
    
    def _get_lines_count(self, node: ast.AST) -> int:
        """
        Safely calculate lines count. 
        Falls back to scanning children if end_lineno is missing (older Python).
        """
        start = getattr(node, "lineno", 0) or 0
        end = getattr(node, "end_lineno", None)
        
        if end is None:
            end = start
            for child in ast.walk(node):
                child_end = getattr(child, "end_lineno", None) or getattr(child, "lineno", None)
                if isinstance(child_end, int):
                    end = max(end, child_end)
                    
        return max(1, end - start + 1)

    def _get_safe_end_lineno(self, node: ast.AST) -> int:
        """
        Safely get end line number.
        Uses the same logic as _get_lines_count to ensure consistency.
        """
        end = getattr(node, "end_lineno", None)
        if end is not None:
            return end
            
        # Fallback: find max lineno among children
        max_line = getattr(node, "lineno", 0) or 0
        for child in ast.walk(node):
            child_line = getattr(child, "end_lineno", None) or getattr(child, "lineno", None)
            if isinstance(child_line, int):
                max_line = max(max_line, child_line)
        return max_line

    def _count_statements(self, node: ast.AST) -> int:
        """
        Recursively count statements, excluding those inside nested functions/classes.
        Correctly handles non-statement blocks like ExceptHandler and match_case.
        """
        count = 0
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, self._STOP_NODES):
                count += 1
                continue # Do not recurse into the body of nested functions/classes
            
            if isinstance(child, ast.stmt):
                count += 1
            
            # Recurse into children to find statements inside blocks 
            count += self._count_statements(child)
            
        return count

    def detect_smells(self, source_code: str, file_path: str) -> List[ArchitecturalSmell]:
        try:
            tree = ast.parse(source_code)
            smells = []
            
            # Iterate only top-level nodes to avoid double counting methods
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_smells = self._analyze_class(node, file_path)
                    smells.extend(class_smells)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Analyze standalone functions
                    method_smells = self._analyze_method(node, file_path, None)
                    smells.extend(method_smells)
            
            return smells
            
        except SyntaxError:
            return []
    
    def _analyze_class(self, class_node: ast.ClassDef, file_path: str) -> List[ArchitecturalSmell]:
        smells = []
        
        methods = [n for n in class_node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        attributes = self._extract_class_attributes(class_node)
        
        god_class_smell = self._detect_god_class(class_node, methods, attributes, file_path)
        if god_class_smell:
            smells.append(god_class_smell)
        
        srp_smell = self._detect_srp_violation(class_node, methods, file_path)
        if srp_smell:
            smells.append(srp_smell)
        
        intimacy_smell = self._detect_inappropriate_intimacy(class_node, methods, file_path)
        if intimacy_smell:
            smells.append(intimacy_smell)
        
        for method in methods:
            method_smells = self._analyze_method(method, file_path, class_node.name)
            smells.extend(method_smells)
        
        return smells
    
    def _analyze_method(self, method_node: Union[ast.FunctionDef, ast.AsyncFunctionDef], 
                       file_path: str, class_name: Optional[str]) -> List[ArchitecturalSmell]:
        smells = []
        
        long_method_smell = self._detect_long_method(method_node, file_path, class_name)
        if long_method_smell:
            smells.append(long_method_smell)
        
        complexity_smell = self._detect_high_complexity(method_node, file_path, class_name)
        if complexity_smell:
            smells.append(complexity_smell)
        
        if class_name:
            envy_smell = self._detect_feature_envy(method_node, file_path, class_name)
            if envy_smell:
                smells.append(envy_smell)
        
        return smells
    
    def _detect_god_class(self, class_node: ast.ClassDef, methods: List[Union[ast.FunctionDef, ast.AsyncFunctionDef]], 
                         attributes: Set[str], file_path: str) -> Optional[ArchitecturalSmell]:
        
        method_count = len(methods)
        lines = self._get_lines_count(class_node)
        
        responsibilities = self._detect_responsibilities(methods)
        unique_responsibilities = {r.keyword for r in responsibilities}
        responsibility_count = len(unique_responsibilities)
        
        cohesion = self._calculate_class_cohesion(methods, attributes)
        
        is_god_class = (
            method_count > self.thresholds.god_class_methods or
            responsibility_count > self.thresholds.god_class_responsibilities or
            cohesion < self.thresholds.god_class_cohesion or
            lines > self.thresholds.god_class_lines
        )
        
        if not is_god_class:
            return None
        
        severity_score = 0
        if method_count > self.thresholds.god_class_methods * 2: severity_score += 2
        elif method_count > self.thresholds.god_class_methods: severity_score += 1
            
        if responsibility_count > self.thresholds.god_class_responsibilities * 2: severity_score += 2
        elif responsibility_count > self.thresholds.god_class_responsibilities: severity_score += 1
            
        if cohesion < self.thresholds.god_class_cohesion / 2: severity_score += 2
        elif cohesion < self.thresholds.god_class_cohesion: severity_score += 1
        
        if severity_score >= 4: severity = SmellSeverity.CRITICAL
        elif severity_score >= 3: severity = SmellSeverity.HIGH
        elif severity_score >= 2: severity = SmellSeverity.MEDIUM
        else: severity = SmellSeverity.LOW
        
        confidence = min(1.0, (severity_score / 6) + 0.3)
        
        evidence = Evidence(
            description=f"God Class detected with {method_count} methods and {responsibility_count} responsibilities",
            confidence=confidence,
            code_snippets=[f"class {class_node.name}:"],
            metadata={
                'method_count': method_count,
                'responsibility_count': responsibility_count,
                'cohesion': cohesion,
                'lines': lines,
                'responsibilities': list(unique_responsibilities)
            }
        )
        
        recommendations = [
            "Extract cohesive groups of methods into separate classes",
            "Apply Single Responsibility Principle",
            "Use composition or delegation"
        ]
        
        decomposition_strategy = self._generate_decomposition_strategy(responsibilities)
        extraction_candidates = list(unique_responsibilities)[:3]
        
        return ArchitecturalSmell(
            smell_type=SmellType.GOD_CLASS,
            severity=severity,
            confidence=confidence,
            file_path=file_path,
            symbol_name=class_node.name,
            line_start=class_node.lineno,
            line_end=self._get_safe_end_lineno(class_node),
            metrics={
                'method_count': method_count,
                'responsibility_count': responsibility_count,
                'cohesion': cohesion,
                'lines': lines
            },
            evidence=evidence,
            recommendations=recommendations,
            decomposition_strategy=decomposition_strategy,
            extraction_candidates=extraction_candidates,
            responsibility_markers=responsibilities
        )
    
    def _detect_long_method(self, method_node: Union[ast.FunctionDef, ast.AsyncFunctionDef], 
                           file_path: str, class_name: Optional[str]) -> Optional[ArchitecturalSmell]:
        
        lines = self._get_lines_count(method_node)
        statements = self._count_statements(method_node)
        
        is_long_method = (
            lines > self.thresholds.long_method_lines or
            statements > self.thresholds.long_method_statements
        )
        
        if not is_long_method:
            return None
        
        severity_score = 0
        if lines > self.thresholds.long_method_lines * 2: severity_score += 2
        elif lines > self.thresholds.long_method_lines: severity_score += 1
            
        if statements > self.thresholds.long_method_statements * 2: severity_score += 2
        elif statements > self.thresholds.long_method_statements: severity_score += 1
        
        if severity_score >= 3: severity = SmellSeverity.HIGH
        elif severity_score >= 2: severity = SmellSeverity.MEDIUM
        else: severity = SmellSeverity.LOW
        
        confidence = min(1.0, (severity_score / 4) + 0.4)
        
        evidence = Evidence(
            description=f"Long method with {lines} lines and {statements} statements",
            confidence=confidence,
            code_snippets=[f"def {method_node.name}(...):"],
            metadata={
                'lines': lines,
                'statements': statements
            }
        )
        
        recommendations = [
            "Extract logical blocks into separate methods",
            "Apply Extract Method refactoring"
        ]
        
        symbol_name = f"{class_name}.{method_node.name}" if class_name else method_node.name
        
        return ArchitecturalSmell(
            smell_type=SmellType.LONG_METHOD,
            severity=severity,
            confidence=confidence,
            file_path=file_path,
            symbol_name=symbol_name,
            line_start=method_node.lineno,
            line_end=self._get_safe_end_lineno(method_node),
            metrics={'lines': lines, 'statements': statements},
            evidence=evidence,
            recommendations=recommendations
        )
    
    def _detect_high_complexity(self, method_node: Union[ast.FunctionDef, ast.AsyncFunctionDef], 
                               file_path: str, class_name: Optional[str]) -> Optional[ArchitecturalSmell]:
        
        complexity = self._calculate_cyclomatic_complexity(method_node)
        nesting_depth = self._calculate_nesting_depth(method_node)
        
        is_high_complexity = (
            complexity > self.thresholds.high_complexity_cyclomatic or
            nesting_depth > self.thresholds.high_complexity_nesting
        )
        
        if not is_high_complexity:
            return None
        
        severity_score = 0
        if complexity > self.thresholds.high_complexity_cyclomatic * 2: severity_score += 2
        elif complexity > self.thresholds.high_complexity_cyclomatic: severity_score += 1
            
        if nesting_depth > self.thresholds.high_complexity_nesting * 1.5: severity_score += 2
        elif nesting_depth > self.thresholds.high_complexity_nesting: severity_score += 1
        
        if severity_score >= 3: severity = SmellSeverity.HIGH
        elif severity_score >= 2: severity = SmellSeverity.MEDIUM
        else: severity = SmellSeverity.LOW
        
        confidence = min(1.0, (severity_score / 4) + 0.5)
        
        evidence = Evidence(
            description=f"High complexity: cyclomatic {complexity}, nesting {nesting_depth}",
            confidence=confidence,
            code_snippets=[f"def {method_node.name}(...):"],
            metadata={
                'cyclomatic_complexity': complexity,
                'nesting_depth': nesting_depth
            }
        )
        
        recommendations = [
            "Reduce conditional complexity",
            "Apply Guard Clauses"
        ]
        
        symbol_name = f"{class_name}.{method_node.name}" if class_name else method_node.name
        
        return ArchitecturalSmell(
            smell_type=SmellType.HIGH_COMPLEXITY,
            severity=severity,
            confidence=confidence,
            file_path=file_path,
            symbol_name=symbol_name,
            line_start=method_node.lineno,
            line_end=self._get_safe_end_lineno(method_node),
            metrics={'cyclomatic_complexity': complexity, 'nesting_depth': nesting_depth},
            evidence=evidence,
            recommendations=recommendations
        )
    
    def _detect_srp_violation(self, class_node: ast.ClassDef, methods: List[Union[ast.FunctionDef, ast.AsyncFunctionDef]], 
                             file_path: str) -> Optional[ArchitecturalSmell]:
        
        responsibilities = self._detect_responsibilities(methods)
        unique_responsibilities = {r.keyword for r in responsibilities}
        responsibility_count = len(unique_responsibilities)
        
        external_deps = self._count_external_dependencies(class_node)
        
        is_srp_violation = (
            responsibility_count > self.thresholds.srp_responsibility_keywords or
            external_deps > self.thresholds.srp_external_dependencies
        )
        
        if not is_srp_violation:
            return None
        
        # Determine severity
        if responsibility_count > self.thresholds.srp_responsibility_keywords * 2:
            severity = SmellSeverity.HIGH
        elif responsibility_count > self.thresholds.srp_responsibility_keywords:
            severity = SmellSeverity.MEDIUM
        else:
            # If we are here, it means external_deps > threshold (since is_srp_violation is True)
            severity = SmellSeverity.LOW
        
        confidence = min(1.0, (responsibility_count / 10) + 0.4)
        
        evidence = Evidence(
            description=f"SRP violation with {responsibility_count} distinct responsibilities",
            confidence=confidence,
            code_snippets=[f"class {class_node.name}:"],
            metadata={
                'responsibility_count': responsibility_count,
                'responsibilities': list(unique_responsibilities),
                'external_dependencies': external_deps
            }
        )
        
        recommendations = ["Split class based on identified responsibilities"]
        
        return ArchitecturalSmell(
            smell_type=SmellType.SRP_VIOLATION,
            severity=severity,
            confidence=confidence,
            file_path=file_path,
            symbol_name=class_node.name,
            line_start=class_node.lineno,
            line_end=self._get_safe_end_lineno(class_node),
            metrics={'responsibility_count': responsibility_count, 'external_dependencies': external_deps},
            evidence=evidence,
            recommendations=recommendations,
            responsibility_markers=responsibilities
        )
    
    def _detect_feature_envy(self, method_node: Union[ast.FunctionDef, ast.AsyncFunctionDef], 
                            file_path: str, class_name: str) -> Optional[ArchitecturalSmell]:
        
        internal_accesses = 0
        external_accesses = 0
        external_targets = set()
        
        # Prevent double counting of attribute chains (e.g. self.a.b)
        # We track processed attributes to skip inner parts of a chain
        processed_attributes = set()
        
        for node in ast.walk(method_node):
            if isinstance(node, ast.Attribute) and node not in processed_attributes:
                # Mark the entire chain as processed so we don't count self.a AND self.a.b
                curr = node
                while isinstance(curr, ast.Attribute):
                    processed_attributes.add(curr)
                    if isinstance(curr.value, ast.Attribute):
                        processed_attributes.add(curr.value)
                    curr = curr.value
                
                # Now analyze the root of this chain
                root = node.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                
                if isinstance(root, ast.Name) and root.id == 'self':
                    internal_accesses += 1
                else:
                    external_accesses += 1
                    if isinstance(root, ast.Name):
                        external_targets.add(root.id)
        
        total_accesses = internal_accesses + external_accesses
        if total_accesses < self.thresholds.feature_envy_min_accesses:
            return None
        
        external_ratio = external_accesses / total_accesses if total_accesses > 0 else 0
        
        if external_ratio <= self.thresholds.feature_envy_external_ratio:
            return None
        
        if external_ratio > 0.9:
            severity = SmellSeverity.HIGH
        elif external_ratio > 0.8:
            severity = SmellSeverity.MEDIUM
        else:
            severity = SmellSeverity.LOW
        
        confidence = min(1.0, external_ratio + 0.2)
        
        evidence = Evidence(
            description=f"Feature Envy with {external_ratio:.1%} external accesses",
            confidence=confidence,
            code_snippets=[f"def {method_node.name}(...):"],
            metadata={
                'internal_accesses': internal_accesses,
                'external_accesses': external_accesses,
                'external_ratio': external_ratio,
                'external_targets': list(external_targets)
            }
        )
        
        recommendations = ["Move method to the class it envies most"]
        
        return ArchitecturalSmell(
            smell_type=SmellType.FEATURE_ENVY,
            severity=severity,
            confidence=confidence,
            file_path=file_path,
            symbol_name=f"{class_name}.{method_node.name}",
            line_start=method_node.lineno,
            line_end=self._get_safe_end_lineno(method_node),
            metrics={'external_ratio': external_ratio},
            evidence=evidence,
            recommendations=recommendations
        )
    
    def _detect_inappropriate_intimacy(self, class_node: ast.ClassDef, methods: List[Union[ast.FunctionDef, ast.AsyncFunctionDef]], 
                                     file_path: str) -> Optional[ArchitecturalSmell]:
        
        private_accesses = 0
        total_external_attr_accesses = 0
        coupling_targets = {}
        
        for method in methods:
            for node in ast.walk(method):
                if isinstance(node, ast.Attribute):
                    # Check if it's an external access (not self)
                    is_self = False
                    if isinstance(node.value, ast.Name) and node.value.id == 'self':
                        is_self = True
                    
                    if not is_self:
                        total_external_attr_accesses += 1
                        attr_name = node.attr
                        if attr_name.startswith('_') and not attr_name.startswith('__'):
                            # Private member access
                            private_accesses += 1
                            # Try to identify target
                            target_name = "unknown"
                            if isinstance(node.value, ast.Name):
                                target_name = node.value.id
                            elif isinstance(node.value, ast.Attribute):
                                target_name = node.value.attr
                                
                            coupling_targets[target_name] = coupling_targets.get(target_name, 0) + 1

        if private_accesses <= self.thresholds.intimacy_private_access:
            return None
        
        coupling_ratio = private_accesses / total_external_attr_accesses if total_external_attr_accesses > 0 else 0
        
        if coupling_ratio <= self.thresholds.intimacy_coupling_ratio and private_accesses < self.thresholds.intimacy_private_access * 2:
             return None

        if private_accesses > self.thresholds.intimacy_private_access * 2:
            severity = SmellSeverity.HIGH
        elif private_accesses > self.thresholds.intimacy_private_access * 1.5:
            severity = SmellSeverity.MEDIUM
        else:
            severity = SmellSeverity.LOW

        confidence = min(1.0, (private_accesses / 10) + 0.3)
        
        evidence = Evidence(
            description=f"Inappropriate Intimacy with {private_accesses} private member accesses",
            confidence=confidence,
            code_snippets=[f"class {class_node.name}:"],
            metadata={
                'private_accesses': private_accesses,
                'coupling_targets': coupling_targets,
                'coupling_ratio': coupling_ratio
            }
        )
        
        recommendations = ["Use public interfaces instead of accessing private members"]
        
        return ArchitecturalSmell(
            smell_type=SmellType.INAPPROPRIATE_INTIMACY,
            severity=severity,
            confidence=confidence,
            file_path=file_path,
            symbol_name=class_node.name,
            line_start=class_node.lineno,
            line_end=self._get_safe_end_lineno(class_node),
            metrics={'private_accesses': private_accesses, 'coupling_ratio': coupling_ratio},
            evidence=evidence,
            recommendations=recommendations
        )
    
    def _extract_class_attributes(self, class_node: ast.ClassDef) -> Set[str]:
        attributes = set()
        for node in ast.walk(class_node):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, ast.AnnAssign):
                    targets = [node.target]
                else: # AugAssign
                    targets = [node.target]
                
                for target in targets:
                    if isinstance(target, ast.Attribute) and \
                       isinstance(target.value, ast.Name) and \
                       target.value.id == 'self':
                        attributes.add(target.attr)
        return attributes
    
    def _detect_responsibilities(self, methods: List[Union[ast.FunctionDef, ast.AsyncFunctionDef]]) -> List[ResponsibilityMarker]:
        responsibilities = []
        
        for method in methods:
            method_name = method.name.lower()
            found_category = False
            
            for responsibility, keywords in self.responsibility_keywords.items():
                if found_category:
                    break
                for keyword in keywords:
                    if keyword in method_name:
                        confidence = 0.8 if method_name.startswith(keyword) else 0.6
                        responsibilities.append(ResponsibilityMarker(
                            keyword=responsibility,
                            context=f"Method: {method.name}",
                            line_number=method.lineno,
                            confidence=confidence
                        ))
                        found_category = True
                        break 
        
        return responsibilities
    
    def _calculate_class_cohesion(self, methods: List[Union[ast.FunctionDef, ast.AsyncFunctionDef]], attributes: Set[str]) -> float:
        if not methods or not attributes:
            return 1.0
        
        method_attributes = {}
        for method in methods:
            method_attrs = set()
            for node in ast.walk(method):
                if isinstance(node, ast.Attribute) and \
                   isinstance(node.value, ast.Name) and \
                   node.value.id == 'self':
                    if node.attr in attributes:
                        method_attrs.add(node.attr)
            method_attributes[method.name] = method_attrs
        
        total_pairs = 0
        cohesive_pairs = 0
        
        method_names = list(method_attributes.keys())
        n = len(method_names)
        if n < 2: return 1.0

        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1
                attrs1 = method_attributes[method_names[i]]
                attrs2 = method_attributes[method_names[j]]
                if attrs1 & attrs2:
                    cohesive_pairs += 1
        
        return cohesive_pairs / total_pairs if total_pairs > 0 else 1.0
    
    def _count_external_dependencies(self, class_node: ast.ClassDef) -> int:
        external_calls = set()
        
        for node in ast.walk(class_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if not func_name.startswith('_') and func_name not in BUILTINS:
                        external_calls.add(func_name)
                elif isinstance(node.func, ast.Attribute):
                     # Check root to avoid counting self.method() or cls.method() as external
                     root = node.func.value
                     while isinstance(root, ast.Attribute):
                         root = root.value
                     
                     if isinstance(root, ast.Name) and root.id in ('self', 'cls'):
                         continue
                         
                     external_calls.add(node.func.attr)
        
        return len(external_calls)
    
    def _calculate_cyclomatic_complexity(self, method_node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
        complexity = 1
        
        for node in ast.walk(method_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif hasattr(ast, 'Match') and isinstance(node, ast.Match):
                complexity += len(node.cases)
            elif isinstance(node, ast.IfExp):
                complexity += 1
            elif isinstance(node, ast.comprehension):
                complexity += 1
                complexity += len(node.ifs)
        
        return complexity
    
    def _calculate_nesting_depth(self, method_node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
        max_depth = 0
        
        def calculate_depth(node, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            if isinstance(node, self._NESTING_NODES):
                current_depth += 1
            
            for child in ast.iter_child_nodes(node):
                calculate_depth(child, current_depth)
        
        calculate_depth(method_node)
        return max_depth
    
    def _generate_decomposition_strategy(self, responsibilities: List[ResponsibilityMarker]) -> str:
        if len(responsibilities) <= 1:
            return "Consider extracting methods into utility functions"
        
        responsibility_groups = {}
        for resp in responsibilities:
            if resp.keyword not in responsibility_groups:
                responsibility_groups[resp.keyword] = []
            responsibility_groups[resp.keyword].append(resp)
        
        strategies = []
        for resp_type, resp_list in responsibility_groups.items():
            if len(resp_list) > 1:
                strategies.append(f"Extract {resp_type} operations into a separate {resp_type.title()}Handler class")
        
        if len(strategies) > 1:
            return "Apply multi-step decomposition: " + "; ".join(strategies)
        elif strategies:
            return strategies[0]
        else:
            return "Extract cohesive method groups into separate classes"