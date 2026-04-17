"""
Logging utilities for navigation system.
"""

import logging
import sys
from datetime import datetime
from typing import Optional

class NavLogger:
    """
    Custom logger for navigation system.
    """
    
    def __init__(self, config, name: str = "BeaconNav"):
        self.config = config
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.log_level.upper()))
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # File handler if needed
        if config.save_measurement_history:
            fh = logging.FileHandler('navigation.log')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            
    def debug(self, msg: str):
        self.logger.debug(msg)
        
    def info(self, msg: str):
        self.logger.info(msg)
        
    def warning(self, msg: str):
        self.logger.warning(msg)
        
    def error(self, msg: str):
        self.logger.error(msg)