from ftplib import FTP
import logging
import sys
import os
from contextlib import contextmanager

from .config import config


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@contextmanager
def ftpconnection(server, *args, **kwargs):
    logger.info(f'Connecting to FTP server: {server} ...')
    ftp = FTP(server, *args, **kwargs)
    logger.info('Connection established.')
    ftp.login()
    logger.info('Login completed.')
    yield ftp
    ftp.quit()
    logger.info('FTP connection closed.')


# TODO Change from os.path to pathlib
def download_eopp():
    logger.info('Downloading Earth orientation parameters.')
    eopp_dir = os.path.join(config['path']['external'], 'eopp')
    local_filenames = {f.split('.')[0] for f in os.listdir(eopp_dir)}
    files = {}

    while True:
        try:
            with ftpconnection('ftp.nga.mil') as ftp:
                if not files:
                    logger.info('Searching FTP server for .eopp files.')
                    base_dir = 'pub2/gps/eopp36'
                    eopp_dirs = [d for d in ftp.nlst(base_dir) if d[-4:] == 'eopp']
                    files = [path for d in eopp_dirs for path in ftp.nlst(d)]
                    files = {os.path.basename(file).split('.')[0]: file for file in files}
                files_to_download = sorted(set(files.keys()).difference(local_filenames))

                logger.info(f'Downloading {len(files_to_download)} file(s).')

                for filename in files_to_download:
                    filepath = files[filename]
                    with open(os.path.join(eopp_dir, filename + '.txt'), 'wb') as f:
                        logger.info(f'Downloading {filepath}')
                        ftp.retrbinary(f'RETR {filepath}', f.write)
                    del files[filename]
                else:
                    logger.info('All .eopp files downloaded.')
                    break
        except TimeoutError:
            logger.info('TimeoutError: Trying to reestablish connection.')


def download_eopc04():
    logger.info('Downloading UT1 time corrections.')
    with ftpconnection('hpiers.obspm.fr') as ftp:
        ftp.cwd('iers/series/opa')
        with open(os.path.join(config['path']['external'], 'eopc04.dat'), 'wb') as f:
            logger.info(f'Downloading iers/series/opa/eopc04_IAU2000')
            ftp.retrbinary(f'RETR eopc04_IAU2000', f.write)


def download_tai_utc():
    logger.info('Downloading TAI time corrections.')
    with ftpconnection('maia.usno.navy.mil') as ftp:
        ftp.cwd('ser7')
        with open(os.path.join(config['path']['external'], 'tai-utc.dat'), 'wb') as f:
            logger.info(f'Downloading maia.usno.navy.mil/ser7/tai-utc.dat')
            ftp.retrbinary(f'RETR tai-utc.dat', f.write)


def download_external():
    download_eopp()
    download_eopc04()
    download_tai_utc()
