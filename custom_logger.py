import logging
import os
import inspect
from dataclasses import dataclass

from config import Config

@dataclass
class CustomLogger:
    format: str = "%(asctime)s - %(name)s - %(levelname)s - Line: %(lineno)s - %(message)s"
    date_format: str = "%d-%m-%Y %H:%M:%S"
    log_file_name: str = "logs.log"
    logger_log_level: int = logging.ERROR
    file_handler_log_level: int = logging.ERROR
    logger_name: str = None

    def create_logger(self) -> logging.Logger:
        if not os.path.exists(self.log_file_name):
            log_dir = os.path.dirname(self.log_file_name)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
        if self.logger_name is None:
            caller_module = inspect.stack()[1].frame.f_globals["__name__"]
        else:
            caller_module = self.logger_name

        logging.basicConfig(format=self.format, datefmt=self.date_format)
        
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.setFormatter(logging.Formatter(self.format, self.date_format))

        logger = logging.getLogger(caller_module)
        logger.setLevel(self.logger_log_level)

        file_handler = logging.FileHandler(self.log_file_name)
        file_handler.setLevel(self.file_handler_log_level)
        file_handler.setFormatter(logging.Formatter(self.format, self.date_format))
        logger.addHandler(file_handler)

        return logger

logger = CustomLogger(logger_name="RoadConditionSystem",
                      logger_log_level=Config.CLI_LOG_LEVEL,
                      file_handler_log_level=Config.FILE_LOG_LEVEL
                      ).create_logger()