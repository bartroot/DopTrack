import sys
import logging
from datetime import timedelta
import autograd.numpy as np
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.optimize as optimize
from collections import Counter
from sklearn.cluster import DBSCAN

from .config import Config
from .io import Database
from .masking import area_mask, fit_mask, time_mask
from .fitting import tanh, fit_tanh, fit_residual
from .model_doptrack import Recording
from . import fitting


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=Config().runtime['log_level'])


class FrequencyData:

    def __init__(self, data):

        # Add data dictionary to instance dictionary
        self.__dict__.update(data)
        self.recording = Recording(self.dataid)

    @classmethod
    def create(cls, spectrogram):

        logger.debug(f'Extracting frequency data points from {spectrogram.dataid}')
        image = spectrogram.image

        # First extraction cycle
        image = cls.first_masking(image)
        data = cls.first_fitting_cycle(image, spectrogram.dt)
        fit_func = cls.create_fit_func(data['tanh_coeffs'],
                                       data['residual_coeffs'],
                                       data['residual_func'])
        logger.debug(f"Extraction cycle 1 completed. Found {len(data['time'])} data points.")

        # Second extraction cycle
        image = cls.second_masking(image, fit_func, data['time'][0], data['time'][-1], spectrogram.dt)
        data = cls.second_fitting_cycle(image, spectrogram.dt)
        fit_func = cls.create_fit_func(data['tanh_coeffs'],
                                       data['residual_coeffs'],
                                       data['residual_func'])
        data['fit_func'] = fit_func
        logger.debug(f"Extraction cycle 2 completed. Found {len(data['time'])} data points.")

        # Calculation of additional values
        data['tca'] = cls.calc_tca(data['time'], fit_func)
        data['fca'] = fit_func(data['tca'])
        data['dt'] = spectrogram.dt
        data['image'] = spectrogram.image
        data['dataid'] = spectrogram.dataid
        logger.debug(f"Values at closest approach calculated. tca:{data['tca']} fca:{data['fca']}")

        return cls(data)

    def save(self):

        filepath = Database().paths['L1B'] / f"{self.dataid}.DOP1B"

        start_time = self.recording.start_time

        with open(filepath, 'w+') as file:
            file.write(f"tca={self.tca}\n")
            file.write(f"fca={self.fca}\n")
            file.write(f"dt={self.dt}\n")

            file.write(f"tanh_coeffs={self.tanh_coeffs}\n")
            file.write(f"residual_func={self.residual_func.__name__}\n")
            file.write(f"residual_coeffs={self.residual_coeffs}\n")

            file.write("==============\n")

            file.write('datetime,time,frequency,power\n')
            for time, frequency, power in zip(self.time, self.frequency, self.power):
                datetime = start_time + timedelta(seconds=int(time))
                file.write(f"{datetime},{time},{frequency},{power}\n")

    @classmethod
    def load(cls, dataid):

        filepath = Database().filepath(dataid, level='L1B')

        data = {'dataid': dataid}

        with open(filepath, 'r') as file:
            data['tca'] = float(file.readline().strip('\n').split('=')[1])
            data['fca'] = float(file.readline().strip('\n').split('=')[1])
            data['dt'] = float(file.readline().strip('\n').split('=')[1])

            line = file.readline().strip('\n')
            string_coeffs = line.split('=')[1].strip('[]').split()
            data['tanh_coeffs'] = [float(coeff) for coeff in string_coeffs]

            residual_func_name = file.readline().strip('\n').split('=')[1]
            data['residual_func'] = getattr(fitting, residual_func_name)

            residual_coeffs = []
            while True:
                line = file.readline().strip('\n')
                if '=' in line and ']' in line:
                    string_coeffs = line.split('=')[1].strip('[]').split()
                    residual_coeffs.extend([float(coeff) for coeff in string_coeffs])
                    break
                elif '=' in line:
                    string_coeffs = line.split('=')[1].strip('[').split()
                    residual_coeffs.extend([float(coeff) for coeff in string_coeffs])
                elif ']' in line:
                    string_coeffs = line.strip(']').split()
                    residual_coeffs.extend([float(coeff) for coeff in string_coeffs])
                    break
                else:
                    string_coeffs = line.split()
                    residual_coeffs.extend([float(coeff) for coeff in string_coeffs])
            data['residual_coeffs'] = residual_coeffs

            file.readline()
            file.readline()

            time, frequency, power = [], [], []
            for line in file.readlines():
                time.append(float(line.split(',')[1]))
                frequency.append(float(line.split(',')[2]))
                power.append(float(line.split(',')[3]))

            data['time'] = np.array(time)
            data['frequency'] = np.array(frequency)
            data['power'] = np.array(power)
            data['fit_func'] = cls.create_fit_func(data['tanh_coeffs'],
                                                   data['residual_coeffs'],
                                                   data['residual_func'])
        return cls(data)

    def plot(self, savepath=None):
        plt.figure(figsize=(16, 9))

        try:
            xlim = (0 - 0.5, self.image.shape[1] - 0.5)
            ylim = (self.image.shape[0]*self.dt, 0)
            plt.xlim(*xlim)
            plt.ylim(*ylim)
            # TODO fix to take into account different nfft
            plt.imshow(self.image, clim=(0, 0.1), cmap='viridis', aspect='auto',
                       extent=(xlim[0],
                               xlim[1],
                               ylim[0],
                               ylim[1]))
        except AttributeError as e:
            logger.warning(f"{e}. This happens when loading data. Plotting without spectrogram.")
        markersize = 0.5 if savepath else None
        plt.scatter(self.frequency, self.time, s=markersize, color='r')

        if savepath:
            plt.savefig(savepath, format='png', dpi=300)
        else:
            plt.show()

    @staticmethod
    def create_fit_func(tanh_coeffs, residual_coeffs, residual_func):
            def fit_func(x):
                return tanh(x, *tanh_coeffs) + residual_func(x, *residual_coeffs)
            return fit_func

    @staticmethod
    def first_masking(image):
        mask1 = area_mask(image, ori='freq')
        mask2 = area_mask(image, ori='time')
        mask = np.logical_and(mask1, mask2)
        masked_image = image * mask + np.mean(image) * np.logical_not(mask)
        return masked_image

    @staticmethod
    def second_masking(image, fitfunc, start_of_signal, end_of_signal, dt):
        mask1 = fit_mask(image, fitfunc, dt)
        mask2 = time_mask(image, start_of_signal, end_of_signal, dt)
        mask = np.logical_and(mask1, mask2)
        masked_image = image * mask
        return masked_image

    @classmethod
    def first_fitting_cycle(cls, image, dt):

        # Extract initial data points
        time, frequency, power = cls.get_rowmaxes_as_points(image, dt)

        # Repeatedly fit tangent curve and filter data points
        sideband = 600
        widths = sideband * np.array([2.5, 1.5, 0.5])

        for width in widths:
            tanh_coeffs = fit_tanh(time, frequency, dt)
            tanh_fit = tanh(time, *tanh_coeffs)
            time, frequency, power = cls.remove_width_outliers(time, frequency, power, tanh_fit, width)
        tanh_fit = tanh(time, *tanh_coeffs)
        residual = frequency - tanh_fit

        # Find clusters of data points
        data = np.dstack((time, residual))[0]
        clustering = DBSCAN(eps=25, min_samples=5).fit(data)
        data = {'time': time,
                'frequency': frequency,
                'power': power,
                'residual': residual,
                'labels': clustering.labels_}

        # Remove clusters with too few points
        filtered_labels = cls.labels_of_clusters_with_n_or_more_points(data['labels'], n=10)
        data = cls.filter_clusters_by_label(data, filtered_labels)

        # Calculate initial residual fit
        residual_func, residual_coeffs = fit_residual(data['time'], data['residual'])

        # Remove cluster where mean of residual of residual is too high
        residual_of_residual = data['residual'] - residual_func(data['time'], *residual_coeffs)
        final_labels = cls.labels_of_clusters_with_low_mean(residual_of_residual, data['labels'], mean=15)
        data = cls.filter_clusters_by_label(data, final_labels)

        # Calculate final residual fit and add fits to final data
        residual_func, residual_coeffs = fit_residual(time, residual)
        data['tanh_coeffs'] = tanh_coeffs
        data['residual_func'] = residual_func
        data['residual_coeffs'] = residual_coeffs

        return data

    @classmethod
    def second_fitting_cycle(cls, image, dt):

        # Extract initial data points and remove points with too low power
        time, frequency, power = cls.get_rowmaxes_as_points(image, dt)
        treshold = 0.05
        time = time[power > treshold]
        frequency = frequency[power > treshold]
        power = power[power > treshold]

        # Compute initial fit
        tanh_coeffs = fit_tanh(time, frequency, dt)
        tanh_fit = tanh(time, *tanh_coeffs)
        residual = frequency - tanh_fit

        # Find clusters of data points
        data = np.dstack((time, residual))[0]
        clustering = DBSCAN(eps=25, min_samples=5).fit(data)
        data = {'time': time,
                'frequency': frequency,
                'power': power,
                'residual': residual,
                'labels': clustering.labels_}

        # Remove clusters with too few points
        filtered_labels = cls.labels_of_clusters_with_n_or_more_points(data['labels'], n=10)
        data = cls.filter_clusters_by_label(data, filtered_labels)

        # Calculate initial residual fit
        residual_func, residual_coeffs = fit_residual(data['time'], data['residual'])

        # Remove cluster where mean of residual of residual is too high
        residual_of_residual = data['residual'] - residual_func(data['time'], *residual_coeffs)
        final_labels = cls.labels_of_clusters_with_low_mean(residual_of_residual, data['labels'], mean=15)
        data = cls.filter_clusters_by_label(data, final_labels)

        # Calculate final residual fit and add fits to final data
        residual_func, residual_coeffs = fit_residual(time, residual)
        data['tanh_coeffs'] = tanh_coeffs
        data['residual_func'] = residual_func
        data['residual_coeffs'] = residual_coeffs

        return data

    @staticmethod
    def calc_tca(time, fitfunc):

        deriv2 = egrad(egrad(fitfunc))(time)
        lower_bound = time[deriv2.argmin()]
        upper_bound = time[deriv2.argmax()]

        try:
            tca = optimize.brentq(egrad(egrad(fitfunc)), lower_bound, upper_bound)
        except ValueError:
            logger.error("Root finding failed!")
            tca = 0
        return tca

    @staticmethod
    def labels_of_clusters_with_n_or_more_points(array_of_labels, n):
        counter = Counter(array_of_labels)
        counter.pop(-1, None)
        new_labels = []
        for key, count in counter.most_common():
            if count < n:
                break
            else:
                new_labels.append(key)
        return new_labels

    @staticmethod
    def labels_of_clusters_with_low_mean(values, array_of_labels, mean):
        new_labels = []
        for label in set(array_of_labels):
            if abs(np.mean(values[array_of_labels == label])) < mean:
                new_labels.append(label)
        return new_labels

    @staticmethod
    def filter_clusters_by_label(data, labels_to_filter):
        bools = np.isin(data['labels'], labels_to_filter)
        for key in data.keys():
            data[key] = data[key][bools]
        return data

    @staticmethod
    def get_rowmaxes_as_points(image, dt):

        noisefilter = signal.gaussian(int(14 / dt) + 1, 2.5)
        noisefilter = noisefilter / np.sum(noisefilter)

        mean = image.mean()

        power = np.zeros(len(image))
        frequency = np.zeros(len(image))

        for i, row in enumerate(image):
            row_norm = row - mean
            # Set all values less than 5 std from mean to zero
            row_norm[row_norm < 5 * np.std(row_norm)] = 0
            # Apply gaussian filter
            row_norm = np.convolve(row_norm, noisefilter, mode='same')

            power[i] = np.max(row_norm)
            frequency[i] = np.argmax(row_norm)

        time = np.arange(len(frequency)) * dt + 0.5 * dt

        # Filter out data points where freq is zero
        bools = frequency != 0
        time = time[bools]
        frequency = frequency[bools]
        power = power[bools]

        return time, frequency, power

    @staticmethod
    def remove_width_outliers(time, frequency, power, mean, band):
        bools = (frequency > mean - band) & (frequency < mean + band)
        new_frequency = frequency[bools]
        new_time = time[bools]
        new_power = power[bools]
        return new_time, new_frequency, new_power
