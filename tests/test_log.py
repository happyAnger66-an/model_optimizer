import logging

from model_optimizer.utils.log import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

logger.debug("Hello, world!")