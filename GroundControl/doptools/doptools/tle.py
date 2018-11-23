import sys
import logging
import requests

from .config import config


logger = logging.getLogger(__name__)


def _spacetrack_query(query):
    payload = {'identity': config['space-track.org']['user'],
               'password': config['space-track.org']['password']}
    with requests.Session() as session:
        login_response = session.post('https://www.space-track.org/ajaxauth/login', data=payload)
        print(login_response.__dict__)
        if login_response.status_code != 200:
            logger.warning('Login failed. TLE not downloaded.', login_response)
            return None
        else:
            api_url = 'https://www.space-track.org/basicspacedata/query/class/' + query
            response = session.get(api_url)
            tle = response.text.split('\r\n')[:2]
            logger.info('TLE downloaded from space-track.org')
            logger.debug('TLE: {}'.format(tle))
            return tle


def get_latest_tle(norad_id):
    logger.info("Getting latest TLE: NORAD ID {}".format(norad_id))
    query_url = f'tle_latest/ORDINAL/1/NORAD_CAT_ID/{norad_id}/orderby/EPOCH ASC/format/tle'
    return _spacetrack_query(query=query_url)
