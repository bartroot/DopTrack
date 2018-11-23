import os
import logging
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from .config import Config


logger = logging.getLogger(__name__)


levels = {'L0': '32fc',
          'L0_meta': 'yml',
          'L1A': 'npy',
          'L1A_meta': 'npy.meta',
          'L1B': 'DOP1B',
          'L2': 'rre'}


class Database:

    def __init__(self, config=Config()):

        self.paths = config.paths
        self.dataids = self._dataids()

    def filepath(self, dataid, level, meta=False):
        if not meta:
            filename = f'{dataid}.{levels[level]}'
        else:
            filename = f"{dataid}.{levels[f'{level}_meta']}"
        filepath = self.paths[level] / filename

        if filepath.is_file():
            return filepath
        else:
            raise FileNotFoundError(f'No such file in database: {filepath}')

    def status(self, dataid):

        L0 = dataid in set.intersection(self.dataids['L0'], self.dataids['L0_meta'])
        L1A = dataid in set.intersection(self.dataids['L1A'], self.dataids['L1A_meta'])
        L1B = dataid in self.dataids['L1B']
        L2 = dataid in self.dataids['L2']

        print(30 * '-')
        print(dataid)
        print(30 * '-')
        print(f"{'L0':3}  {L0}")
        print(f"{'L1A':3}  {L1A}")
        print(f"{'L1B':3}  {L1B}")
        print(f"{'L2':3}  {L2}")
        print(30 * '-')

    def create_log(self):

        logdict = {}

        for dataid in sorted(self.dataids['all']):
            logdict[dataid] = {'processed': {'rec_armed': None,
                                             'recorded': None,
                                             'level_1A': None,
                                             'level_1B': None,
                                             'level_2': None},
                               'files': {'rec_data': dataid in self.dataids['L0'],
                                         'rec_meta': dataid in self.dataids['L0_meta'],
                                         'level_1A': dataid in self.dataids['L1A'],
                                         'level_1B': dataid in self.dataids['L1B'],
                                         'level_2': dataid in self.dataids['L2']}}
        with open(self.paths['logs'] / 'database.log', 'w') as logfile:
            yaml.dump(logdict, stream=logfile, default_flow_style=False)

    def update_log(self, dataid, level, value):
        logdict = self.load_log()
        # TODO should be changed - processed vs files
        logdict[dataid]['processed'] = value
        with open(self.paths['logs'] / 'database.log', 'w') as logfile:
            yaml.dump(logdict, stream=logfile, default_flow_style=False)

    def load_log(self):
        with open(self.paths['logs'] / 'database.log', 'r') as logfile:
            return yaml.load(logfile)


    def _dataids(self):
        dataid_dict = {}

        for level, ext in levels.items():
            baselevel = level.split('_')[0]
            if self.paths[baselevel] is not set:
                files = self._listfiles(self.paths[baselevel])
            else:
                files = []
                for path in self.paths[baselevel]:
                    files.append(self._listfiles(path))

            dataid_dict[level] = set([file.split('.')[0] for file in files
                                      if file.split('.')[-1] == ext])
        dataid_dict['all'] = set.union(*dataid_dict.values())
        return dataid_dict

    @staticmethod
    def _listfiles(path):
        """List all files, excluding folders, in a directory"""
        try:
            return [name for name in os.listdir(path) if len(name.split('.')) != 1]
        except FileNotFoundError as e:
            logger.error(e)
            return []

#    def print_dataids(self):
#        ymls = self.get_dataids('yml')
#        rres = self.get_dataids('rre')
#        passids = {}
#        for dataid in set.union(ymls, rres):
#            satid = '_'.join(dataid.split('_')[:2])
#            if satid not in passids:
#                passids[satid] = {'yml': [], 'rre': [], 'both': []}
#            if dataid in ymls:
#                passids[satid]['yml'].append(dataid)
#            if dataid in rres:
#                passids[satid]['rre'].append(dataid)
#            if dataid in set.intersection(ymls, rres):
#                passids[satid]['both'].append(dataid)
#
#        # TODO add 32fc files to overview
#        print(39 * '-')
#        print(f"{'SatName':15}{'SatID':5}{'yml':>7}{'rre':>6}{'both':>6}")
#        print(39 * '-')
#        for key, val in passids.items():
#            satname, satid = key.split('_')[:2]
#            print(f"{satname:15}{satid:5}{len(val['yml']):7}{len(val['rre']):6}{len(val['both']):6}")
#        print(39 * '-')


def read_meta(dataid, filepath=None):
    if filepath is None:
        dataid = dataid[:27]
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


def read_eopp(folderpath=Database().paths['external']):
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


def read_eopc04(folderpath=Database().paths['external']):
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


def read_tai_utc(folderpath=Database().paths['external']):
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
