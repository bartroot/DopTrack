import logging
import sys

from .config import Config
from .utils import log_formatter


if Config().runtime['logging']:
    logging.basicConfig(stream=sys.stdout, format=log_formatter, level=logging.DEBUG)
else:
    logging.getLogger(__name__).addHandler(logging.NullHandler())
