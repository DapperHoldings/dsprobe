"""
Timing utilities: Timer, Rate (like ROS Rate).
"""

import time
from contextlib import contextmanager
from typing import Optional

class Timer:
    """Simple context manager for timing code blocks"""
    
    def __init__(self, name: str = "Timer", logger=None):
        self.name = name
        self.logger = logger
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.logger:
            self.logger.debug(f"{self.name}: {self.elapsed*1000:.2f} ms")
        else:
            print(f"{self.name}: {self.elapsed*1000:.2f} ms")

class Rate:
    """Maintain a constant publishing rate (like rospy.Rate)"""
    
    def __init__(self, frequency: float):
        self.period = 1.0 / frequency
        self.last_time = time.time()
        
    def sleep(self):
        now = time.time()
        elapsed = now - self.last_time
        if elapsed < self.period:
            time.sleep(self.period - elapsed)
        self.last_time = time.time()