from ai4bmr_core.log.log import logger


def test_logger():
    logger.debug("Hello World!")
    logger.info("Hello World!")
    logger.warning("Hello World!")
    logger.error("Hello World!")
    logger.critical("Hello World!")
