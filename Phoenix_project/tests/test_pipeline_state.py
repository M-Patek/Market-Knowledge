"""
Tests for configuration file loading and validation (Layer 8).
"""

import pytest
import yaml
import os

# List of all critical YAML config files to validate
# Paths are relative to the repository root where pytest is typically run
CONFIG_FILES_TO_VALIDATE = [
    "Phoenix_project/config.yaml",
    "Phoenix_project/workflow_config.yaml",
    "Phoenix_project/config/symbolic_rules.yaml",
    "Phoenix_project/config/ai_clients.yaml",
]

@pytest.mark.parametrize("config_file", CONFIG_FILES_TO_VALIDATE)
def test_yaml_files_load_and_are_not_empty(config_file):
    """
    Confirms that every YAML configuration (Layer 8, Task 3)
    successfully loads and is not empty.
    """
    assert os.path.exists(config_file), f"Config file not found: {config_file}"
    
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)
    
    assert data is not None, f"Config file is empty or malformed: {config_file}"
