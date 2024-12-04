import logging
import sys
from datetime import datetime
from typing import Optional
from uuid import uuid4

class RequestIdFilter(logging.Filter):
    """Filter that adds request_id to log records."""
    def __init__(self):
        super().__init__()
        self._request_id: Optional[str] = None

    @property
    def request_id(self) -> str:
        if self._request_id is None:
            self._request_id = str(uuid4())
        return self._request_id

    def reset_request_id(self) -> None:
        """Reset request_id for new request."""
        self._request_id = None

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = self.request_id
        return True

def setup_logging(module_name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Args:
        module_name: Name of the module (for log identification)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Logger instance configured according to requirements
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level.upper())

    # Clear any existing handlers
    logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level.upper())

    # Add request ID filter
    request_id_filter = RequestIdFilter()
    console_handler.addFilter(request_id_filter)

    # Create formatter with all required fields
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] [%(name)s] [%(request_id)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger

# Create a global request ID filter instance that can be used across the application
request_id_filter = RequestIdFilter()

def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        module_name: Name of the module
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(module_name) 