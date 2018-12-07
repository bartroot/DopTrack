import sys
import logging
from collections import namedtuple
from functools import wraps
from time import time

Position = namedtuple('Position', 'x y z')
GeodeticPosition = namedtuple('GeodeticPosition', 'latitude longitude altitude')

log_format_string = '%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)-s'
log_formatter = logging.Formatter(log_format_string)


def inspect_loggers(logger_name):
    """"Inspect a logger including all its parents."""
    log_to_debug = logging.getLogger(logger_name)
    while log_to_debug is not None:
        print("level: %s, name: %s, handlers: %s" % (log_to_debug.level,
                                                     log_to_debug.name,
                                                     log_to_debug.handlers))
        log_to_debug = log_to_debug.parent


def timing(num=1, exit_after_run=False):
    """
    Decorator for timing functions and methods.

    Parameters
    ----------
    num : int, optional
        The number of times the function or method has to be called.
        The higher the number the more accurate the timing becomes.

    Example
    --------
    >>> @timing(num=1000)
    >>> def func():
    ...     #do something
    >>>
    >>> func()
    """
    def decorator(f):
        @wraps(f)
        def wrap(*args, **kwargs):
            tt = 0
            for _ in range(num):
                ts = time()
                result = f(*args, **kwargs)
                te = time()
                tt += te - ts
            print(f'Function {f.__name__} took on average: {tt/num} sec')
            if exit_after_run:
                sys.exit()
            return result
        return wrap
    return decorator
