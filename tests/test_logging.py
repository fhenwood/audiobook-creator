
import unittest
import os
import logging
import shutil
from audiobook.utils.logging_config import configure_logging, get_logger

class TestLogging(unittest.TestCase):
    
    def setUp(self):
        self.log_file = "tests/test_logs/app.log"
        if os.path.exists("tests/test_logs"):
            shutil.rmtree("tests/test_logs")
        
        # Configure logging (force re-configure)
        configure_logging(level=logging.INFO, log_file=self.log_file)
        self.logger = get_logger("test_logger")

    def test_log_creation(self):
        msg = "Test log message"
        self.logger.info(msg)
        
        # Check file exists
        self.assertTrue(os.path.exists(self.log_file), "Log file should exist")
        
        # Check content
        with open(self.log_file, 'r') as f:
            content = f.read()
            self.assertIn(msg, content)
            self.assertIn("INFO", content)
            self.assertIn("test_logger", content)

if __name__ == '__main__':
    unittest.main()
