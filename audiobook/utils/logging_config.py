
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# Default log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def configure_logging(level=logging.INFO, log_file="app.log"):
    """
    Configure the root logger with console and file handlers.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Formatter
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    
    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File Handler (Rotating)
    # Ensure logs directory exists if path implies one, defaulting to current dir if simple filename
    log_path = os.path.abspath(log_file)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_path, 
        maxBytes=10*1024*1024, # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    logging.info(f"Logging configured. Writing to {log_path}")

def get_logger(name):
    """Get a logger instance for a module."""
    return logging.getLogger(name)
