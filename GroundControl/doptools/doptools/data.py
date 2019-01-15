import os
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fftpack import fft, fftshift
from tqdm import tqdm
import logging
import multiprocessing

from .io import Database, read_meta
from .extraction import extract_frequency_data, create_fit_func
from . import fitting


logger = logging.getLogger(__name__)


class EmptyRecordingError(Exception):
    pass


class Recording:
    """
    Recording from DopTrack.

    Parameters
    ----------
    dataid : str
        ID of recording in the database.
    spectrogram : (N, M) numpy.ndarray
        An array containing the values of the spectrogram in dB.
    freq_lims : (float, float) tuple
        The minimum and maximum frequency values of the spectrogram.
    time_lims : (datetime.datetime, datetime.datetime) tuple
        The end and start time of recording. The order is reversed since it is
        convention to flip the y-axis in a spectrogram.

    """

    def __init__(self, dataid):
        self.dataid = dataid

        self.meta = read_meta(dataid)
        self.duration = int(self.meta['Sat']['Predict']['Length of pass'])
        self.start_time = self.meta['Sat']['Record']['time1 UTC']
        self.stop_time = self.start_time + timedelta(seconds=self.duration)
        self.sample_freq = int(self.meta['Sat']['Record']['sample_rate'])
        self.tuning_freq = int(self.meta['Sat']['State']['Tuning Frequency'])

    def data(self, dt):
        """Returns an iterable of the raw data cut into chunks of  size dt*sample_rate."""
        database = Database()
        filepath = database.filepath(self.dataid, level='L0')

        if os.stat(filepath).st_size < 10e7:  # Check if file is more than 10 mb
            raise EmptyRecordingError(f'L0 data file for {self.dataid} is empty or too small')

        samples = int(self.sample_freq * self.duration)
        timesteps = int(self.duration / dt)
        cutoff = int(2 * samples / timesteps)
        with open(filepath, 'r') as file:
            for i in range(timesteps):
                if i == 0:
                    data = np.fromfile(file, dtype=np.float32, count=cutoff)
                    if not data.any():
                        raise EmptyRecordingError(f'First chunck of L0 data file contains all zeros')
                    else:
                        yield data
                else:
                    yield np.fromfile(file, dtype=np.float32, count=cutoff)


class L1A:
    """
    L1A data (spectrogram) of a DopTrack recording.

    Parameters
    ----------
    dataid : str
        ID of recording in the database.
    spectrogram : (N, M) numpy.ndarray
        An array containing the values of the spectrogram in dB.
    freq_lims : (float, float) tuple
        The minimum and maximum frequency values of the spectrogram.
    time_lims : (datetime.datetime, datetime.datetime) tuple
        The end and start time of recording. The order is reversen since it is
        convention to flip the y-axis in a spectrogram.


    Example
    --------
    >>> from doptools.doptrack import Spectrogram
    >>> s = Spectrogram('Delfi-C3_32789_201602121133')
    >>> s.plot()
    """

    def __init__(self, dataid, spectrogram, freq_lims, time_lims, dt):
        self.dataid = dataid
        self.recording = Recording(dataid)
        self.spectrogram = spectrogram
        self.spectrogram_decibel = self._to_decibel(spectrogram)
        self.freq_lims = freq_lims
        self.time_lims = time_lims
        self.dt = dt
        self.dfreq = (freq_lims[1] - freq_lims[0]) / spectrogram.shape[1]

    @classmethod
    def create(cls, dataid, bounds=(12000, 24000), nfft=250_000, dt=1):
        """
        Create a spectrogram from a 32fc file.

        Parameters
        ----------
        dataid : str
            ID of recording in the database.
        bounds : (float or int, float or int) tuple, optional
            Contains the lower and upper bounds of the desired frequency range
            relative to the tuning frequency. Default is suitable for Delfi-C3.
        nfft : int
            The number of frequency bins to use in FFT. The FFT is fastest
            if nfft is a power of two.
            The default is suitable for 250 kHz bandwidth
        dt : int or float
            The timestep used during the FFT. Specifies the time resolution
            of the spectrogram.

        Returns
        -------
        Spectrogram
            A spectrogram object for the given recording.

        Warnings
        --------
        The dt parameter should be chosen carefully. "Nice" values like 1,
        0.5, or 0.2 should work, but "ugly" values like 0.236687 will
        most likely raise an array broadcast exception.
        """

        logger.info(f"Creating spectrogram for {dataid}")

        # Determine the limits of the spectrogram.
        recording = Recording(dataid)
        freq_lims = (recording.tuning_freq - recording.sample_freq/2,
                     recording.tuning_freq + recording.sample_freq/2)
        time_lims = (recording.stop_time, recording.start_time)

        # Create array with all frequency values.
        bandwidth = np.linspace(freq_lims[0],
                                freq_lims[1],
                                nfft)

        # Calculate the bounds of zoom
#        lower = tuning_freq + bounds[0]
#        upper = tuning_freq + bounds[1]
        # TODO decide on how bounds are determined
        estimated_signal_width = 7000
        estimated_signal_freq = 145_888_300
        lower = estimated_signal_freq - estimated_signal_width
        upper = estimated_signal_freq + estimated_signal_width

        # Create a mask for desired frequency values.
        mask, = np.nonzero((lower <= bandwidth) & (bandwidth <= upper))
        bandwidth = bandwidth[mask]
        freq_lims = (bandwidth[0], bandwidth[-1])

        # Read data, cut data, and perform FFT for each timestep.
        timesteps = int(recording.duration / dt)
        # TODO fix spec width so it is not hardcoded
        spectrogram = np.zeros((timesteps, 14000))

        for i, raw_data in tqdm(enumerate(recording.data(dt)), total=timesteps):

            signal = np.zeros(int(len(raw_data)/2), dtype=np.complex)
            signal.real = raw_data[::2]
            signal.imag = -raw_data[1::2]
            row = cls._construct_spectrum(signal, nfft)
            # TODO does not currently use bounds other than as a flag

            if bounds:
                row = row[mask]
            spectrogram[i] = row

        return cls(dataid, spectrogram, freq_lims, time_lims, dt)

    @classmethod
    def load(cls, dataid):
        """
        Load a spectrogram from .npy and .npy.meta files.

        Parameters
        ----------
        dataid : str
            ID of recording in the database.

        Returns
        -------
        Spectrogram
            A spectrogram object for the given recording.
        """

        logger.info(f"Loading spectrogram for {dataid}")

        db = Database()
        datafilepath = db.filepath(dataid, level='L1A')
        metafilepath = db.filepath(dataid, level='L1A', meta=True)
        with open(metafilepath, 'r') as file:
            xlim1 = float(file.readline().strip('\n').split('=')[1])
            xlim2 = float(file.readline().strip('\n').split('=')[1])
            ylim1 = datetime.strptime(file.readline().strip('\n').split('=')[1],
                                      '%Y-%m-%d %H:%M:%S.%f')
            ylim2 = datetime.strptime(file.readline().strip('\n').split('=')[1],
                                      '%Y-%m-%d %H:%M:%S.%f')
        freq_lims = (xlim1, xlim2)
        time_lims = (ylim1, ylim2)
        spectrogram = np.load(datafilepath)
        dt = (time_lims[0] - time_lims[1]).total_seconds() / spectrogram.shape[0]

        return cls(dataid, spectrogram, freq_lims, time_lims, dt)

    def save(self, filename=None):
        """
        Save the spectrogram data as .npy and .npy.meta files.

        Parameters
        ----------
        dataid : str
            ID of recording in the database.
        """

        logger.info(f"Saving spectrogram for {self.dataid}")

        folderpath = Database().paths['L1A']

        if filename is None:
            filename = self.dataid

        with open(folderpath / f'{filename}.npy.meta', 'w+') as file:
            file.write(f'xlim_lower={self.freq_lims[0]}\n')
            file.write(f'xlim_upper={self.freq_lims[1]}\n')
            file.write(f'ylim_lower={self.time_lims[0]}\n')
            file.write(f'ylim_upper={self.time_lims[1]}')

        filepath = folderpath / f'{filename}.npy'
        np.save(filepath, self.spectrogram)

        logger.info(f'Sepctrogram saved to {filepath}')

    def plot(self, savepath=None, cmap='viridis', clim=None, decibel=False, **kwargs):
        """
        Plot the spectrogram.

        Parameters
        ----------
        bounds : (float, float) tuple, optional
            Contains the lower and upper bounds of the desired frequency range
            relative to the tuning frequency.
        cmap : str, optional
            Specifies the color map.
        clim : (float, float) tuple, optional
            Specifies the limits of the color map.
        kwargs : optional
            Keyword arguments are passed on to matplotlib.pyplot.imshow.
        """

        # Convert datetime values to floats.
        num_time_lims = tuple(e for e in map(mdates.date2num, self.time_lims))

        fig, ax = plt.subplots(figsize=(16, 9))

        # Create new axis for colorbar.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        # Plot spectrogram with defined limits.
        spectrogram = self.spectrogram_decibel if decibel else self.spectrogram
        clim = (0, spectrogram.mean() + 4*spectrogram.std()) if not clim else clim
        im = ax.imshow(spectrogram,
                       extent=(self.freq_lims[0],
                               self.freq_lims[1],
                               num_time_lims[0],
                               num_time_lims[1]),
                       aspect="auto", cmap=cmap, clim=clim, **kwargs)

        # Format time axis and plot colorbar.
        self._axis_format_totime(ax.yaxis)
        fig.colorbar(im, cax=cax, orientation='vertical')

        if savepath:
            fig.savefig(savepath, format='png', dpi=150)
            plt.close(fig)
        else:
            fig.show()

    @staticmethod
    def _construct_spectrum(signal, nfft):
        """
        Calculate the spectrum of a given input signal.

        Parameters
        ----------
        signal : (N,) numpy.ndarray
            Array containing complex time series data of a signal

        Returns
        -------
        (N,) numpy.ndarray
            Array containing the spectrum of the signal over the full bandwidth.
        """
        spectrum = fft(signal, nfft)
        spectrum = fftshift(spectrum)
        spectrum = abs(spectrum)

        return spectrum

    @staticmethod
    def _to_decibel(spectrogram):
        """ Calculate relative power level in dB of spectrum. """
        return 10*np.log10(spectrogram / spectrogram.mean())

    @staticmethod
    def _axis_format_totime(axis):
        """Format normal axis to time axis for plotting."""
        date_format = mdates.DateFormatter('%H:%M:%S')
        axis.set_major_formatter(date_format)
        axis.set_major_locator(mdates.MinuteLocator(interval=2))


class L1B:

    def __init__(self, data):

        # Add data dictionary to instance dictionary
        self.__dict__.update(data)
        self.recording = Recording(self.dataid)

    @classmethod
    def create(cls, L1A_data, plot=False):

        logger.info(f"Extracting frequency data for {L1A_data.dataid}")

        data = extract_frequency_data(L1A_data.spectrogram, L1A_data.dt, plot=plot)
        data['dt'] = L1A_data.dt
        data['spectrogram'] = L1A_data.spectrogram
        data['dataid'] = L1A_data.dataid

        return cls(data)

    @classmethod
    def load(cls, dataid):

        logger.info(f"Loading frequency data for {dataid}")

        filepath = Database().filepath(dataid, level='L1B')

        data = {'dataid': dataid}

        with open(filepath, 'r') as file:
            data['tca'] = float(file.readline().strip('\n').split('=')[1])
            data['fca'] = float(file.readline().strip('\n').split('=')[1])
            data['dt'] = float(file.readline().strip('\n').split('=')[1])
            data['rms'] = float(file.readline().strip('\n').split('=')[1])

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
            data['fit_func'] = create_fit_func(data['tanh_coeffs'],
                                               data['residual_coeffs'],
                                               data['residual_func'])
        return cls(data)

    def save(self):

        logger.info(f"Saving frequency data for {self.dataid}")

        filepath = Database().paths['L1B'] / f"{self.dataid}.DOP1B"

        start_time = self.recording.start_time

        with open(filepath, 'w+') as file:
            file.write(f"tca={self.tca}\n")
            file.write(f"fca={self.fca}\n")
            file.write(f"dt={self.dt}\n")
            file.write(f"rms={self.rms}\n")

            file.write(f"tanh_coeffs={self.tanh_coeffs}\n")
            file.write(f"residual_func={self.residual_func.__name__}\n")
            file.write(f"residual_coeffs={self.residual_coeffs}\n")

            file.write("==============\n")

            file.write('datetime,time,frequency,power\n')
            for time, frequency, power in zip(self.time, self.frequency, self.power):
                datetime = start_time + timedelta(seconds=int(time))
                file.write(f"{datetime},{time:.2f},{frequency:.2f},{power:.6f}\n")

    def plot(self, fit_func=True, savepath=None, cmap='viridis', clim=None):
        fig, ax = plt.subplots(figsize=(16, 9))

        try:
            xlim = (0 - 0.5, self.spectrogram.shape[1] - 0.5)
            ylim = (self.spectrogram.shape[0]*self.dt, 0)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            clim = (0, self.spectrogram.mean() + 4*self.spectrogram.std()) if not clim else clim
            # TODO fix to take into account different nfft
            ax.imshow(self.spectrogram, clim=clim, cmap=cmap, aspect='auto',
                      extent=(xlim[0],
                              xlim[1],
                              ylim[0],
                              ylim[1]))
        except AttributeError as e:
            logger.warning(f"{e}. This happens when loading data. Plotting without spectrogram.")
        markersize = 0.5 if savepath else None
        ax.scatter(self.frequency, self.time, s=markersize, color='r')
        if fit_func:
            times = np.linspace(self.time[0], self.time[-1], 100)
            ax.plot(self.fit_func(times), times, 'k')

        if savepath:
            fig.savefig(savepath, format='png', dpi=300)
            plt.close(fig)
        else:
            fig.show()


class L2:

    def __init__(self):
        raise NotImplementedError
