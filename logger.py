import logging
import sys
from functools import partial


def wlogger(name, val, step=None, logger=None):
    if step is None:
        logger.info('{}: {}, ({})'.format(name, val, step))
    else:
        logger.info('{}: {}'.format(name, val))


logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)
log = partial(wlogger, logger=logger)
