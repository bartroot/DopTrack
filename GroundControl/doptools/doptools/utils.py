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

