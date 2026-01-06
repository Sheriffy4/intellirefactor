"""
Example analysis plugin for IntelliRefactor

Demonstrates how to create a custom analysis plugin that extends
the core analysis capabilities with domain-specific rules.
"""

import os
import ast
from typing import Dict, List, Any
from pathlib import Path

from ..plugin_interface import Analysis