import logging
import requests
from ratelimit import limits, sleep_and_retry

from doptools.config import Config


__all__ = [
        'spacetrack_query',
        'get_latest_tle',
        'get_n_latest_tles',
        'get_tles_for_n_latest_days']


logger = logging.getLogger(__name__)


_one_minute = 20 * 60
_one_hour = 60 * 60


# If api is called more than 200 times in one hour throw RateLimitException
@limits(calls=200, period=_one_hour)
@sleep_and_retry  # If api is called more than 20 times in one minute sleep and call again
@limits(calls=20, period=_one_minute)
def spacetrack_query(query, username=None, password=None):
    """
    Send a query to the space-track.org API.

    The function sets up a requests session, logs in to the space-track
    website, and sends a get request to the API. Specifically, the request
    makes an API call to ``/basicspacedata/query/`` along with the query.
    The documentation to the space-track API can be found at
    https://www.space-track.org/documentation.

    Parameters
    ----------
    query : str
        ID of recording in the database.
    username : str, optional
        Username to the space-track website. If no username is given it
        will try and find one in the config file.
    password : str, optional
        Passowrd to the space-track website. If no password is given it
        will try and find one in the config file.


    Returns
    -------
    list(str) or None
        A list of lines of the response or None if no response was given.

    Raises
    ------
    RateLimitException
        If the API is called too many times within a certain time span.

    Warnings
    --------
    This function is rate limited to 20 calls per minute and 200 calls per hour.
    If the one minute rate limit is reached the function call will sleep and
    retry when allowed. If the one hour rate limit is reached an exception will
    be thrown.

    Examples
    --------
    >>> from doptools.tle import spacetrack_query
    >>> r = spacetrack_query('tle/format/tle/NORAD_CAT_ID/32789/orderby/EPOCH asc/limit/2')
    >>> r
    ['1 32789U 08021G   19027.10519311 +.00001069 +00000-0 +76214-4 0  9990',
     '2 32789 097.4845 080.9512 0011010 250.1418 109.8623 15.06607684585399',
     '1 32789U 08021G   19026.70669576 +.00000988 +00000-0 +70743-4 0  9994',
     '2 32789 097.4845 080.5640 0010984 251.6363 108.3670 15.06606680585336']

    """
    c = Config()
    if username is None:
        username = c.credentials['space-track.org']['user']
    if password is None:
        password = c.credentials['space-track.org']['password']

    payload = {'identity': username,
               'password': password}

    with requests.Session() as session:
        login_response = session.post('https://www.space-track.org/ajaxauth/login', data=payload)
        print('###########################')
        print(login_response.status_code)
        print('###########################')
        if login_response.status_code != 200:

            logger.warning('Login failed. TLE not downloaded.', login_response)
            return None
        else:
            api_url = 'https://www.space-track.org/basicspacedata/query/' + query
            print('###########################')
            print(session.get)
            print('###########################')
            response = session.get(api_url)
            return response.text.split('\r\n')[:-1]


def get_latest_tle(norad_id, **kwargs):
    """
    Get latest TLE of a satellite from `space-track.org`.

    Parameters
    ----------
    norad_id : int or str
        NORAD ID of the satellite.
    kwargs : optional
        Keyword arguments are passed on to the `spacetrack_query` function.

    Returns
    -------
    tuple(str)
        A tuple of two strings corresponding to the two lines in a TLE.

    See also
    --------
    spacetrack_query : Send a query to the space-track API.

    Examples
    --------
    >>> from doptools.tle import get_latest_tle
    >>> tle = get_latest_tle(32789)
    >>> tle
    ('1 32789U 08021G   19027.10519311 +.00001069 +00000-0 +76214-4 0  9990',
     '2 32789 097.4845 080.9512 0011010 250.1418 109.8623 15.06607684585399')

    """
    logger.info("Getting latest TLE: NORAD ID {norad_id}")
    query = f'class/tle_latest/ORDINAL/1/NORAD_CAT_ID/{norad_id}/format/tle'
    response = spacetrack_query(query, **kwargs)
    assert len(response) == 2
    logger.info('TLE downloaded from space-track.org')
    return tuple(response)


def get_n_latest_tles(norad_id, n, **kwargs):
    """
    Get the n latest TLE's of a satellite from `space-track.org`.

    Parameters
    ----------
    norad_id : int or str
        NORAD ID of the satellite.
    n : int
        The number of TLE's to get.
    kwargs : optional
        Keyword arguments are passed on to the `spacetrack_query` function.

    Returns
    -------
    tuple(str)
        A tuple of two strings corresponding to the two lines in a TLE.

    See also
    --------
    spacetrack_query : Send a query to the space-track API.

    Examples
    --------
    >>> from doptools.tle import get_n_latest_tles
    >>> tles = get_n_latest_tles(32789, 2)
    >>> tles
    [('1 32789U 08021G   19027.10519311 +.00001069 +00000-0 +76214-4 0  9990',
      '2 32789 097.4845 080.9512 0011010 250.1418 109.8623 15.06607684585399'),
     ('1 32789U 08021G   19026.70669576 +.00000988 +00000-0 +70743-4 0  9994',
      '2 32789 097.4845 080.5640 0010984 251.6363 108.3670 15.06606680585336')]

    """
    logger.info(f"Getting {n} latest TLE's: NORAD ID {norad_id}")
    query = f'class/tle/format/tle/NORAD_CAT_ID/{norad_id}/orderby/EPOCH desc/limit/{n}'
    response = spacetrack_query(query, **kwargs)
    assert len(response) == 2 * n
    logger.info(f"{n} TLE's downloaded from space-track.org")
    return list(zip(*[iter(response)]*2))


def get_tles_for_n_latest_days(norad_id, n, **kwargs):
    """
    Get all TLE's from the last n days of a satellite from `space-track.org`.

    Parameters
    ----------
    norad_id : int or str
        NORAD ID of the satellite.
    n : int
        Number of days. The function gets TLE's from `now - n` to `now`.
    kwargs : optional
        Keyword arguments are passed on to the `spacetrack_query` function.

    Returns
    -------
    tuple(str)
        A tuple of two strings corresponding to the two lines in a TLE.

    See also
    --------
    spacetrack_query : Send a query to the space-track API.

    Examples
    --------
    >>> from doptools.tle import get_tles_for_n_latest_days
    >>> tles = get_tles_for_n_latest_days(32789, 5)
    >>> tles
    [('1 32789U 08021G   19027.10519311 +.00001069 +00000-0 +76214-4 0  9990',
      '2 32789 097.4845 080.9512 0011010 250.1418 109.8623 15.06607684585399'),
     ('1 32789U 08021G   19026.70669576 +.00000988 +00000-0 +70743-4 0  9994',
      '2 32789 097.4845 080.5640 0010984 251.6363 108.3670 15.06606680585336'),
     ('1 32789U 08021G   19026.57386317 +.00000974 +00000-0 +69762-4 0  9992',
      '2 32789 097.4845 080.4349 0010973 252.1324 107.8707 15.06606462585313'),
     ('1 32789U 08021G   19026.10894873  .00000934  00000-0  67078-4 0  9996',
      '2 32789  97.4846  79.9832 0010949 253.9878 106.0143 15.06605837585343'),
     ('1 32789U 08021G   19025.90969979 +.00000909 +00000-0 +65416-4 0  9999',
      '2 32789 097.4846 079.7896 0010940 254.7345 105.2674 15.06605459585216'),
     ('1 32789U 08021G   19024.71420506  .00000730  00000-0  53339-4 0  9998',
      '2 32789  97.4848  78.6278 0010865 259.3299 100.6705 15.06603411585136'),
     ('1 32789U 08021G   19024.38212296 +.00000690 +00000-0 +50665-4 0  9990',
      '2 32789 097.4848 078.3051 0010843 260.6127 099.3875 15.06602909584988')]

    """
    logger.info(f"Getting {n} latest TLE's: NORAD ID {norad_id}")
    query = f'class/tle/format/tle/NORAD_CAT_ID/{norad_id}/orderby/EPOCH desc/EPOCH/>now-{n}'
    response = spacetrack_query(query, **kwargs)
    logger.info(f"{n} TLE's downloaded from space-track.org")
    return list(zip(*[iter(response)]*2))
