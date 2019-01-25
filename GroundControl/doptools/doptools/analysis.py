import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import time
from collections import defaultdict
from tqdm import tqdm
import logging
from scipy.optimize import curve_fit
import dill as pickle

from .data import L1B, L1C
from .model import SatellitePassTLE
from .io import Database
from . import fitting


logger = logging.getLogger(__name__)


class ResidualAnalysis:

    def __init__(self, L1B_obj):

        self.dataid = L1B_obj.dataid

        self.dataL1B = L1B_obj
        self.dataL1C = L1C.create(self.dataL1B)
        self.dataTLE = SatellitePassTLE.from_L1B(self.dataL1B)
        assert np.array_equal(self.dataL1B.time, self.dataL1C.time)
        assert np.array_equal(self.dataL1B.time, self.dataTLE.time)

        self.time = self.dataL1B.time
        self.time_sec = self.dataL1B.time_sec
        self.first_residual = self.dataL1C.rangerate - self.dataTLE.rangerate

        coeffs, covar = curve_fit(fitting.linear, self.time_sec, self.first_residual)
        self.linear_fit_slope = coeffs[0]
        self.linear_fit_intersection = coeffs[1]
        self.first_residual_fit = fitting.linear(self.time_sec, *coeffs)
        self.second_residual = self.first_residual - self.first_residual_fit

    def __repr__(self):
        return f"{self.__module__}.{self.__class__.__name__}('{self.dataid}')"

    def save(self):
        db = Database()
        path = db.paths['analysis'] / f'passes/{self.dataid}.DOPA'
        pickle.dump(self, open(path, "wb"))

    @classmethod
    def load(cls, dataid):
        db = Database()
        path = db.paths['analysis'] / f'passes/{dataid}.DOPA'
        return pickle.load(open(path, "rb"))

    def plot(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312, sharex=ax1)
        ax3 = fig.add_subplot(313, sharex=ax1)

        ax1.plot(self.time, self.dataL1C.rangerate, '.', label='doptrack')
        ax1.plot(self.time, self.dataTLE.rangerate, '.', label='tle')
        ax1.set_ylabel('')
        ax1.legend()
        ax1.grid()

        ax2.plot(self.time, self.first_residual, label='first residual')
        ax2.plot(self.time, self.first_residual_fit, label='linear fit')
        ax2.set_ylabel('range rate [m/s]')
        ax2.legend()
        ax2.grid()

        ax3.plot(self.time, self.second_residual, label='second residual')
        ax3.set_xlabel('time')
        ax3.set_ylabel('')
        ax3.legend()
        ax3.grid()

        fig.show()


class BulkAnalysis:

    def __init__(self):
        try:
            self.data = pd.read_csv(Database().paths['analysis'] / 'bulk.csv')
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
            d = L1B.load(dataid)
            try:
                a = ResidualAnalysis.load(dataid)
            except FileNotFoundError:
                a = ResidualAnalysis(d)
                a.save()

            datadict['dataid'].append(a.dataid)
            datadict['tca'].append(a.dataTLE.tca)
            datadict['tca_time'].append(a.dataTLE.tca.time())
            datadict['fca'].append(f'{a.dataL1B.fca:.2f}')
            datadict['rmse'].append(a.dataL1B.rmse)
            datadict['max_elevation'].append(max(a.dataTLE.elevation))
            datadict['linear_fit_slope'].append(a.linear_fit_slope)
            datadict['linear_fit_intersection'].append(a.linear_fit_intersection)

            if a.dataTLE.tca.time() < time(16):
                datadict['timeofday'].append('morning')
            else:
                datadict['timeofday'].append('evening')

        self.data = pd.DataFrame.from_dict(datadict)
        self.data.set_index('dataid', inplace=True)
        self.data.sort_index(axis=0, inplace=True)
        self.data.sort_index(axis=1, inplace=True)

        self.data.to_csv(Database().paths['analysis'] / 'bulk.csv')
