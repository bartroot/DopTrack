import os
import logging
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict

from .config import Config


logger = logging.getLogger(__name__)


levels = {'L0': '32fc',
          'L0_meta': 'yml',
          'L1A': 'npy',
          'L1A_meta': 'npy.meta',
          'L1B': 'DOP1B',
          'L2': 'rre'}


class Database:

    def __init__(self, config=None):

        config = config if config is not None else Config()
        self.paths = config.paths

    @property
    def dataids(self):
        dataid_dict = {}

        # Add dataids based on what files are in database
        for level, ext in levels.items():
            baselevel = level.split('_')[0]
            if not isinstance(self.paths[baselevel], set):
                files = self._listfiles(self.paths[baselevel])
            else:
                files = []
                for path in self.paths[baselevel]:
                    files.extend(self._listfiles(path))
            filenames_with_correct_ext = [file for file in files if file.split('.')[-1] == ext]
            dataid_dict[level] = self._get_dataids_from_filenames(filenames_with_correct_ext)

        # Add dataids based on what has been logged in the processing status file
        status_dict = defaultdict(set)
        for dataid, status in self.read_status().items():
            if status != 'success':
                status_dict[status].add(DataID(dataid))
        dataid_dict.update(status_dict)
        if len(status_dict) == 0:
            dataid_dict['L1B_failed'] = set()
        else:
            dataid_dict['L1B_failed'] = set.union(*status_dict.values())
        dataid_dict['all'] = set.union(*dataid_dict.values())

        return dataid_dict

    def filepath(self, dataid, level, meta=False):

        dataid = DataID(dataid)

        if meta and f'{level}_meta' not in levels:
            raise RuntimeError(f'Meta files are not used for data level {level}')

        if not meta:
            filename = f'{dataid}.{levels[level]}'
        else:
            filename = f"{dataid}.{levels[f'{level}_meta']}"

        if level == 'L0':
            folderpath = self._get_folder_with_file(filename, self.paths[level])
        else:
            folderpath = self.paths[level]
        filepath = folderpath / filename

        if filepath.is_file():
            return filepath
        else:
            raise FileNotFoundError(f'No such file in database: {filepath}')

    def create_folder_structure(self):

        for key, path in self.paths.items():

            if type(path) is set:
                for subpath in path:
                    try:
                        subpath.mkdir(parents=True)
                        logger.info(f"Created directory: {subpath}")
                    except FileExistsError:
                        logger.info(f"Directory already exists: {subpath}")

            elif key is not "config":
                try:
                    path.mkdir(parents=True)
                    logger.info(f"Created directory: {path}")
                except FileExistsError:
                    logger.info(f"Directory already exists: {path}")

        # Temporary output folder structure
        # TODO change this when processing scripts are integrated into doptools
        for path in [self.paths['output'] / 'L1B', self.paths['output'] / 'L1b_failed']:
            try:
                path.mkdir(parents=True)
                logger.info(f"Created directory: {path}")
            except FileExistsError:
                logger.info(f"Directory already exists: {path}")

    def update_status(self, dataid, status):

        if status not in {'success', 'empty_recording', 'pass_not_found', 'unknown_error'}:
            raise ValueError(f'Given status {status} is not a valid status')

        status_dict = self.read_status()
        status_dict[dataid] = status
        self.save_status(status_dict)

    def read_status(self):
        status_dict = {}
        filepath = self.paths['default'] / 'status.txt'
        if filepath.is_file():
            with open(filepath, 'r') as readfile:
                readfile.readline()
                for line in readfile.readlines():
                    dataid_old, status_old = line.split()
                    status_dict[dataid_old] = status_old
        return status_dict

    def save_status(self, status_dict):
        filepath = self.paths['default'] / 'status.txt'
        with open(filepath, 'w') as savefile:
            savefile.write('dataid                         error\n')
            for key in sorted(status_dict):
                savefile.write(f'{key}    {status_dict[key]}\n')

    @classmethod
    def setup(cls, config=None):

        config = config if config is not None else Config()

        for key, path in config.paths.items():
            if key == 'L0':
                paths = path if isinstance(path, set) else {path}
                for subpath in paths:
                    if not subpath.is_dir():
                        logger.error(f'Given L0 data folder does not exist: {subpath}')
            elif key in ['default', 'L1A', 'L1B', 'L2', 'external', 'output', 'logs']:
                if not path.is_dir():
                    logger.info(f'Creating folder ({key}): {path}')
                    os.makedirs(path)
                else:
                    logger.info(f'Folder already exists ({key}): {path}')

    def validate(self):
        raise NotImplementedError()

    @staticmethod
    def _listfiles(path):
        """List all files, excluding folders, in a directory"""
        return [name for name in os.listdir(path) if len(name.split('.')) != 1]

    @staticmethod
    def _get_folder_with_file(filename, folders):
        """Returns the first folder that contains file"""
        for folder in folders:
            filepath = folder / filename
            if filepath.is_file():
                return folder
        else:
            raise FileNotFoundError(f'No such file in database: {filename}')

    @staticmethod
    def _get_dataids_from_filenames(filenames):
        valid_dataids = []
        for filename in filenames:
            filestem = filename.split('.')[0]
            try:
                valid_dataids.append(DataID(filestem))
            except TypeError:
                logger.warning(f'File with incorrect dataid in database: {filename}')

        return set(valid_dataids)


class DataID(str):

    def __init__(self, string):
        self.validate()

    def validate(self):
        try:
            assert len(self.split('_')) == 3
            assert len(self.satnum) == 5
            assert len(self.strtimestamp) == 12
        except AssertionError as e:
            raise TypeError(f'Invalid dataid format {self}. {e}')

    @property
    def satname(self):
        satname, satnum, strtimestamp = self.split('_')
        return satname

    @property
    def satnum(self):
        satname, satnum, timestamp = self.split('_')
        return satnum

    @property
    def strtimestamp(self):
        satname, satnum, strtimestamp = self.split('_')
        return strtimestamp

    @property
    def timestamp(self):
        raise NotImplementedError()

    def __repr__(self):
        return f"DataID('{str(self)}')"


def read_meta(dataid, filepath=None):
    if filepath is None:
        filepath = Database().filepath(dataid, level='L0', meta=True)
    with open(filepath, 'r') as metafile:
        meta = yaml.load(metafile)
    return meta


def read_rre(dataid, filepath=None, metafilepath=None):

    if filepath is None:
        dataid = dataid[:27]
        filepath = Database().filepath(dataid, level='L1B')
    meta = read_meta(dataid, filepath=metafilepath)

    rre = dict(tca=None, fca=None, time=[], datetime=[], frequency=[])

    with open(filepath, 'r') as rrefile:
        rre['tca'] = float(rrefile.readline().split('=')[1])
        rre['fca'] = float(rrefile.readline().split('=')[1])
        next(rrefile)  # skip header line
        for line in rrefile:
            time, frequency, _ = line.split(',')
            rre['time'].append(float(time))
            rre['frequency'].append(float(frequency))

    rre['time'] = np.array(rre['time'])
    rre['frequency'] = np.array(rre['frequency'])

    start_time = meta['Sat']['Record']['time1 UTC']
    rre['tca'] = start_time + timedelta(seconds=rre['tca'])
    rre['datetime'] = np.array([start_time + timedelta(seconds=i) for i in rre['time']])

    return rre


def read_eopp(folderpath=None):
    folderpath = Database().paths['external'] if folderpath is None else folderpath
    data = dict(MJD=[], Xp=[], Yp=[])
    try:
        for filename in os.listdir(folderpath / 'eopp'):
            if os.stat(folderpath / 'eopp' / filename).st_size == 0:
                break
            with open(folderpath / 'eopp' / filename) as f:
                for _ in range(5):
                    next(f)
                line = f.readline()
                e = line.split()
                data['MJD'].append(int(e[0]))
                data['Xp'].append(float(e[1]) / 3600)
                data['Yp'].append(float(e[2]) / 3600)
    except FileNotFoundError as e:
        logger.error(e)
    df = pd.DataFrame.from_dict(data)
    df.set_index('MJD', inplace=True)
    return df


def read_eopc04(folderpath=None):
    folderpath = Database().paths['external'] if folderpath is None else folderpath
    data = dict(datetime=[], DUT1=[], LOD=[])
    try:
        with open(folderpath / 'eopc04.dat') as f:
            for _ in range(14):
                next(f)
            for line in f.readlines():
                e = line.split()
                data['datetime'].append(datetime(int(e[0]), int(e[1]), int(e[2])))
                data['DUT1'].append(float(e[6]))
                data['LOD'].append(float(e[7]))
    except FileNotFoundError as e:
        logger.error(e)
    df = pd.DataFrame.from_dict(data)
    df.set_index('datetime', inplace=True)
    return df


def read_tai_utc(folderpath=None):
    folderpath = Database().paths['external'] if folderpath is None else folderpath
    month = dict(JAN=1, FEB=2, MAR=3, APR=4, MAY=5, JUN=6,
                 JUL=7, AUG=8, SEP=9, OCT=10, NOV=11, DEC=12)
    data = dict(datetime=[], JD=[], DAT=[])
    try:
        with open(folderpath / 'tai-utc.dat') as f:
            for line in f.readlines():
                e = line.split()

                y, m, d = int(e[0]), month[e[1]], int(e[2])
                jd, dat = float(e[4]), float(e[6])
                c1, c2 = float(e[11][:5]), float(e[13][:9])

                data['datetime'].append(datetime(y, m, d))
                data['JD'].append(jd)
                data['DAT'].append(dat + (jd - 2400000.5 - c1)*c2)
    except FileNotFoundError as e:
        logger.error(e)
    df = pd.DataFrame.from_dict(data)
    df.set_index('datetime', inplace=True)
    return df


def read_nutation_coeffs():
    path = Path(__file__).parent / 'nutation_coeffs.csv'
    return pd.read_csv(path)
