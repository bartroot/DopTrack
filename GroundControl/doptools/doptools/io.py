import logging
import sys
import os
import yaml
from datetime import datetime, timedelta
import functools
import pandas as pd

from .config import config


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def print_dataids():
    ymls = get_dataids('yml')
    rres = get_dataids('rre')
    passids = {}
    for dataid in set.union(ymls, rres):
        satid = '_'.join(dataid.split('_')[:2])
        if satid not in passids:
            passids[satid] = {'yml': [], 'rre': [], 'both': []}
        if dataid in ymls:
            passids[satid]['yml'].append(dataid)
        if dataid in rres:
            passids[satid]['rre'].append(dataid)
        if dataid in set.intersection(ymls, rres):
            passids[satid]['both'].append(dataid)
    for key, val in passids.items():
        print('==============')
        print(key)
        print('yml:', len(val['yml']))
        print('rre:', len(val['rre']))
        print('both:', len(val['both']))


def get_dataids(extensions):
    return set.intersection(*_get_dataid_sets(extensions, config['path']['recordings']))


def get_all_dataids():
    files = _listfiles(config['path']['recordings'])
    return set(map(_remove_file_extension, files))


def read_meta(dataid):
    dataid = dataid[:27]
    with open(os.path.join(config['path']['recordings'], f'{dataid}.yml'), 'r') as metafile:
        meta = yaml.load(metafile)
    return meta


def read_rre(dataid):
    meta = read_meta(dataid)
    rre = dict(tca=None, fca=None, datetime=[], frequency=[])
    path = os.path.join(config['path']['rre'], f'{dataid}.rre')
    with open(path, 'r') as rrefile:
        rre['tca'] = float(rrefile.readline().split('=')[1])
        rre['fca'] = float(rrefile.readline().split('=')[1])
        next(rrefile)  # skip header line
        for line in rrefile:
            time, frequency, _ = line.split(',')
            rre['datetime'].append(float(time))
            rre['frequency'].append(float(frequency))
    start_time = meta['Sat']['Record']['time1 UTC']
    rre['tca'] = start_time + timedelta(seconds=rre['tca'])
    rre['datetime'] = [start_time + timedelta(seconds=i) for i in rre['datetime']]
    return rre


def read_eopp():
    data = dict(MJD=[], Xp=[], Yp=[])
    for filename in os.listdir(os.path.join(config['path']['external'], 'eopp')):
        if os.stat(os.path.join(config['path']['external'], 'eopp', filename)).st_size == 0:
            break
        with open(os.path.join(config['path']['external'], 'eopp', filename)) as f:
            for _ in range(5):
                next(f)
            line = f.readline()
            e = line.split()
            data['MJD'].append(int(e[0]))
            data['Xp'].append(float(e[1])/3600)
            data['Yp'].append(float(e[2])/3600)
    df = pd.DataFrame.from_dict(data)
    df.set_index('MJD', inplace=True)
    return df


def read_eopc04():
    data = dict(datetime=[], DUT1=[], LOD=[])
    with open(os.path.join(config['path']['external'], 'eopc04.dat')) as f:
        for _ in range(14):
            next(f)
        for line in f.readlines():
            e = line.split()
            data['datetime'].append(datetime(int(e[0]), int(e[1]), int(e[2])))
            data['DUT1'].append(float(e[6]))
            data['LOD'].append(float(e[7]))
    df = pd.DataFrame.from_dict(data)
    df.set_index('datetime', inplace=True)
    return df


def read_tai_utc():
    month = dict(JAN=1, FEB=2, MAR=3, APR=4, MAY=5, JUN=6,
                 JUL=7, AUG=8, SEP=9, OCT=10, NOV=11, DEC=12)
    data = dict(datetime=[], JD=[], DAT=[])
    try:
        with open(os.path.join(config['path']['external'], 'tai-utc.dat')) as f:
            for line in f.readlines():
                e = line.split()

                y, m, d = int(e[0]), month[e[1]], int(e[2])
                jd, dat = float(e[4]), float(e[6])
                c1, c2 = float(e[11][:5]), float(e[13][:9])

                data['datetime'].append(datetime(y, m, d))
                data['JD'].append(jd)
                data['DAT'].append(dat + (jd - 2400000.5 - c1)*c2)
    except FileNotFoundError:
        logger.warning('Could not find tai-utc.dat file.')
    df = pd.DataFrame.from_dict(data)
    df.set_index('datetime', inplace=True)
    return df


def read_nutation_coeffs():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'nutation_coeffs.csv'))
    return pd.read_csv(path)


def _listfiles(path):
    """List all files, excluding folders, in a directory"""
    return (file for file in os.listdir(path)
            if os.path.isfile(os.path.join(path, file)))


def _remove_file_extension(file):
    return file.split('.')[0]


def _get_file_extension(file):
    return file.split('.')[1]


def _file_is_type(extension, file):
    return _get_file_extension(file) == extension


def _get_dataid_sets(extensions, path):
    sets = []
    for ext in extensions.split():
        files = _listfiles(path)
        files = filter(functools.partial(_file_is_type, ext), files)
        sets.append(set(map(_remove_file_extension, files)))
    return sets
