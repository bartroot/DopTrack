import sys
import numpy as np
from collections import namedtuple
from functools import wraps
from time import time

Position = namedtuple('Position', 'x y z')
GeodeticPosition = namedtuple('GeodeticPosition', 'latitude longitude altitude')


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


#log_to_debug = logging.getLogger(f'{__name__}.test')
#while log_to_debug is not None:
#    print("level: %s, name: %s, handlers: %s" % (log_to_debug.level,
#                                                 log_to_debug.name,
#                                                 log_to_debug.handlers))
#    log_to_debug = log_to_debug.parent
