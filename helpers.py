import logging

logger = logging.getLogger(__name__)
logger.propagate = True  # important: send logs to root logger

def my_helper_function():
    logger.info("Hello from helper function!")