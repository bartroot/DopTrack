import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import time
from collections import defaultdict
from tqdm import tqdm
import logging

from .data import L1B, L2
from .model import SatellitePassTLE
from .io import Database
from .config import Config


logger = logging.getLogger(__name__)


class ResidualAnalysis:

    def __init__(self, dataid):

        self.dataid = dataid

        self.dataL1B = L1B.load(self.dataid)
        self.dataL2 = L2.create(self.dataL1B)
        self.dataTLE = SatellitePassTLE.from_dataid(dataid)
        assert np.array_equal(self.dataL1B.time, self.dataL2.time)
        assert np.array_equal(self.dataL1B.time, self.dataTLE.time)

        self.time = self.dataL1B.time
        self.first_residual = self.dataL2.rangerate - self.dataTLE.rangerate
        self.dtca = (self.dataL2.tca - self.dataTLE.tca).total_seconds()

    def __repr__(self):
        return f'{self.__module__}.{self.__class__.__name__}({self.dataid})'

    def plot(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        ax1.plot(self.time, self.dataL2.rangerate, label='doptrack')
        ax1.plot(self.time, self.dataTLE.rangerate, label='tle')
        ax1.set_ylabel('range rate')
        ax1.legend()
        ax1.grid()
        ax2.plot(self.time, self.first_residual)
        ax2.set_xlabel('time')
        ax2.set_ylabel('first residual')
        ax2.grid()
        fig.show()


class BulkAnalysis:

    def __init__(self):
        try:
            self.data = pd.read_csv(Config().paths['default'] / 'bulk.csv')
            self.data.set_index('dataid', inplace=True)
            self.data['tca'] = pd.to_datetime(self.data['tca'])
        except FileNotFoundError:
            self.update()

    def plot(self, key1, key2):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.data[key1], self.data[key2], '.')
        ax.set_xlabel(key1)
        ax.set_ylabel(key2)
        ax.grid()
        fig.show()

    def update(self):
        datadict = defaultdict(list)
        dataids = Database().dataids['L1B']
        for dataid in tqdm(dataids, desc='Analyzing passes:'):
            a = ResidualAnalysis(dataid)

            datadict['dataid'].append(a.dataid)
            datadict['tca'].append(a.dataTLE.tca)
            datadict['tca_time'].append(a.dataTLE.tca.time())
            datadict['fca'].append(a.dataL1B.fca)
            datadict['dtca'].append(a.dtca)
            datadict['rmse'].append(a.dataL1B.rmse)
            datadict['max_elevation'].append(a.dataTLE.max_elevation)

            if a.dataTLE.tca.time() < time(16):
                datadict['timeofday'].append('morning')
            else:
                datadict['timeofday'].append('evening')

        self.data = pd.DataFrame.from_dict(datadict)
        self.data.set_index('dataid', inplace=True)
        self.data.sort_index(axis=0, inplace=True)
        self.data.sort_index(axis=1, inplace=True)

        self.data.to_csv(Config().paths['default'] / 'bulk.csv')
