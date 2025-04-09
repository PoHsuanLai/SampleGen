"""
Utilities for loading and parsing configuration files for the music generation system.
"""

import os
import yaml
from typing import Dict, Any, Optional


def get_default_config_path() -> str:
    """
    Get the path to the default configuration file.
    
    Returns:
        Path to the default configuration file
    """
    config_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(config_dir, "training_config.yaml")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file, or None to use the default
        
    Returns:
        Dictionary containing the configuration
    """
    if config_path is None:
        config_path = get_default_config_path()
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {str(e)}")
        print("Using default configuration")
        return {}


def get_generator_config(config: Dict[str, Any], generator_type: str) -> Dict[str, Any]:
    """
    Get configuration for a specific generator type.
    
    Args:
        config: The loaded configuration dictionary
        generator_type: Type of generator (melody, harmony, bass, drums)
        
    Returns:
        Dictionary containing generator-specific configuration
    """
    if not config or 'generators' not in config:
        return {}
        
    generators_config = config.get('generators', {})
    default_config = config.get('default', {})
    
    # Get generator-specific config or empty dict if not found
    generator_config = generators_config.get(generator_type, {})
    
    # Merge with default config (generator config takes precedence)
    merged_config = {**default_config, **generator_config}
    
    # Get training config or empty dict
    training_config = merged_config.get('training', {})
    
    # Merge training with generator config
    merged_config.update(training_config)
    
    return merged_config


def get_style_config(config: Dict[str, Any], style_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific hip-hop style.
    
    Args:
        config: The loaded configuration dictionary
        style_name: Name of the style (modern, classic, trap, lofi)
        
    Returns:
        Dictionary containing style-specific configuration
    """
    if not config or 'styles' not in config:
        return {}
        
    styles_config = config.get('styles', {})
    
    # Get style-specific config or modern as default
    style_config = styles_config.get(style_name, styles_config.get('modern', {}))
    
    return style_config


def get_models_dir(root_dir: str, config: Dict[str, Any]) -> str:
    """
    Get the models directory from configuration.
    
    Args:
        config: The loaded configuration dictionary
        
    Returns:
        Path to the models directory
    """
    # Default to 'models' directory in the project root
    default_dir = os.path.join(root_dir, "models")
    
    if not config:
        return default_dir
    
    # Check for models_dir in config
    models_dir = config.get('models_dir')
    if not models_dir:
        return default_dir
    
    # If it's a relative path, make it absolute
    if not os.path.isabs(models_dir):
        models_dir = os.path.join(root_dir, models_dir)
    
    return models_dir 