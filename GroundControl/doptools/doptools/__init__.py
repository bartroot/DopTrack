import os
import sys
import logging

from .config import Config
from .utils import log_format_string

if os.name == 'posix':
    import matplotlib
    matplotlib.use('Agg')


if Config().runtime['logging']:
    logging.basicConfig(stream=sys.stdout, format=log_format_string, level=logging.DEBUG)
else:
    logging.getLogger(__name__).addHandler(logging.NullHandler())
