from ftplib import FTP
from pathlib import Path
import logging
import sys
import os
from contextlib import contextmanager

from .config import Config


logger = logging.getLogger(__name__)


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


def download_eopp():
    logger.info('Downloading Earth orientation parameters.')
    eopp_dir = Config().paths['external'] / 'eopp'
    local_filenames = {Path(file).stem for file in os.listdir(eopp_dir)}
    files = {}

    while True:
        try:
            with ftpconnection('ftp.nga.mil') as ftp:
                if not files:
                    logger.info('Searching FTP server for .eopp files.')
                    base_dir = 'pub2/gps/eopp36'
                    eopp_dirs = [d for d in ftp.nlst(base_dir) if d[-4:] == 'eopp']
                    files = [path for d in eopp_dirs for path in ftp.nlst(d)]
                    files = {Path(file).stem: file for file in files}

                files_to_download = sorted(set(files.keys()).difference(local_filenames))

                logger.info(f'Downloading {len(files_to_download)} file(s).')

                for filename in files_to_download:
                    filepath = files[filename]
                    with open(eopp_dir / f'{filename}.txt', 'wb') as f:
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
        with open(Config().paths['external'] / 'eopc04.dat', 'wb') as f:
            logger.info(f'Downloading iers/series/opa/eopc04_IAU2000')
            ftp.retrbinary(f'RETR eopc04_IAU2000', f.write)


def download_tai_utc():
    logger.info('Downloading TAI time corrections.')
    with ftpconnection('maia.usno.navy.mil') as ftp:
        ftp.cwd('ser7')
        with open(Config().paths['external'] / 'tai-utc.dat', 'wb') as f:
            logger.info(f'Downloading maia.usno.navy.mil/ser7/tai-utc.dat')
            ftp.retrbinary(f'RETR tai-utc.dat', f.write)


def download_external():
    download_eopc04()
    download_tai_utc()
    download_eopp()

