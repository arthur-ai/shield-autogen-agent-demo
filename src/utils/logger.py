"""
Centralized logging configuration for the application.
Provides consistent logging setup across all modules.
"""

import logging
import logging.config
import yaml
from pathlib import Path
from datetime import datetime
import os

def setup_logging(
    default_path='config/logging_config.yaml',
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """
    Setup logging configuration for the entire application.
    
    Args:
        default_path (str): Path to the logging configuration file
        default_level (int): Default logging level
        env_key (str): Environment variable key for logging config
        
    Returns:
        logging.Logger: Configured logger instance
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value

    # Create logs directory if it doesn't exist
    Path('logs/tools').mkdir(parents=True, exist_ok=True)
    Path('logs/assistant').mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d')
    
    if os.path.exists(path):
        with open(path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                # Dynamically update log filenames with timestamp
                for handler in config['handlers'].values():
                    if 'filename' in handler:
                        base_path = handler['filename']
                        handler['filename'] = base_path.format(timestamp=timestamp)
                logging.config.dictConfig(config)
            except Exception as e:
                print(f'Error in Logging Configuration: {e}')
                logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        
    return logging.getLogger(__name__)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name (str): Name of the module requesting the logger
        
    Returns:
        logging.Logger: Logger instance configured for the module
    """
    return logging.getLogger(name)
