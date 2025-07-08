import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger():
    log_dir = os.path.join(os.path.dirname(__file__), '../logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('model_handler')
    logger.setLevel(logging.DEBUG)
    log_file = os.path.join(log_dir, 'logger.txt')
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
logger = setup_logger()
