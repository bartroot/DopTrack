"""Data model of DopTrack

This module contains classes for each main type of data used internally in
the DopTrack project.

"""
import os
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fftpack import fft, fftshift
import scipy.optimize as optimize
import scipy.constants as constants
from tqdm import tqdm
import logging
import uncertainties

from .recording import Recording
from .model import SatellitePassTLE
from .io import Database
from .extraction import extract_frequency_data, create_fit_func
from . import fitting


__all__ = ['EmptyRecordingError', 'L0', 'L1A', 'L1B', 'L1C']


logger = logging.getLogger(__name__)


class EmptyRecordingError(Exception):
    """Raised when the data file (L0) of a recording is empty."""
    pass


class L0:
    """
    Recorded complex binary data from DopTrack.

    Parameters
    ----------
    dataid : str
        ID of recording in the database.
    recording : recording.Recording
        An object representing a DopTrack radio recording.
    freq_lims : (float, float) tuple
        The minimum and maximum frequency values of the spectrogram.
    time_lims : (datetime.datetime, datetime.datetime) tuple
        The end and start time of recording. The order is reversed since it is
        convention to flip the y-axis in a spectrogram.

    Notes
    -----
    There is no technical reason to keep the L0 data as a class instead of
    simply a function. L0 is implemented as a class to be consistent with
    all the other data levels also being implemented as classes.

    """
    def __init__(self, dataid):
        self.dataid = dataid
        self.recording = Recording(dataid)

    def data(self, dt):
        """Returns an iterable of the raw data cut into chunks of  size dt*sample_rate."""
        database = Database()
        filepath = database.filepath(self.dataid, level='L0')

        if os.stat(filepath).st_size < 10e7:  # Check if file is more than 10 mb
            raise EmptyRecordingError(f'L0 data file for {self.dataid} is empty or too small')

        samples = int(self.recording.sample_freq * self.recording.duration)
        timesteps = int(self.recording.duration / dt)
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
    recording : data.L0
        Recording data object.
    spectrogram : (N, M) numpy.ndarray
        An array containing the values of the spectrogram in power.
    spectrogram_decibel : (N, M) numpy.ndarray
        An array containing the values of the spectrogram in power level (dB).
    freq_lims : (float, float) tuple
        The minimum and maximum frequency values of the spectrogram.
    time_lims : (datetime.datetime, datetime.datetime) tuple
        The end and start time of recording. The order is reversen since it is
        convention to flip the y-axis in a spectrogram.
    dt : float
        Timestep of spectrogram.
    dfreq : float
        Frequency step of spectrogram.

    Examples
    --------
    >>> from doptools.data import L1A
    >>> s = L1A.create('Delfi-C3_32789_201602121133')
    >>> s.plot()

    """

    def __init__(self, dataid, spectrogram, freq_lims, time_lims, dt):
        self.dataid = dataid
        self.recording = Recording(dataid)
        self.spectrogram = spectrogram
        self.freq_lims = freq_lims
        self.time_lims = time_lims
        self.dt = dt
        self.dfreq = (freq_lims[1] - freq_lims[0]) / spectrogram.shape[1]

    @property
    def spectrogram_decibel(self):
        """(N, M) numpy.ndarray: Spectrogram in power level (dB)."""
        return self._to_decibel(self.spectrogram)

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
        nfft : int, optional
            The number of frequency bins to use in FFT. The FFT is fastest
            if nfft is a power of two.
            The default is suitable for 250 kHz bandwidth
        dt : int or float, optional
            The timestep used during the FFT. Specifies the time resolution
            of the spectrogram.

        Returns
        -------
        data.L1A
            A spectrogram object.

        Warnings
        --------
        The dt parameter should be chosen carefully. "Nice" values like 1,
        0.5, or 0.2 should work, but "ugly" values like 0.236687 will
        most likely raise an array broadcast exception.

        Several values are stil hardcoded so the script will give incorrect
        results if nfft!=250_000 etc.

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

        for i, raw_data in tqdm(enumerate(L0(dataid).data(dt)), total=timesteps):

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
        data.L1A
            A spectrogram object.

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
        nfft : int
            The number of frequency bins to use when performing the Fast Fourier Transform.

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
        """ Calculate power level in dB of spectrogram relative to the median."""
        return 10*np.log10(spectrogram / np.median(spectrogram))

    @staticmethod
    def _axis_format_totime(axis):
        """Format normal axis to time axis for plotting."""
        date_format = mdates.DateFormatter('%H:%M:%S')
        axis.set_major_formatter(date_format)
        axis.set_major_locator(mdates.MinuteLocator(interval=2))


class L1B:
    """
    L1B data (time-frequency data points) of a DopTrack recording.

    Parameters
    ----------
    dataid : str
        ID of recording in the database.
    recording : data.L0
        Recording data object.
    time : (datetime.datetime,) numpy.ndarray
        Time in datetime for each data point.
    time_sec : (float,) numpy.ndarray
        Time in seconds from start of recording for each data point.
    frequency : (float,) numpy.ndarray
    power : (float,) numpy.ndarray
    tca : float
        Estimated time of closest approach of satellite as datetime object.
    tca_sec : float
        Estimated time of closest approach of satellite in seconds from start of recording.
    fca : float
        Estimated frequency at closest approach of satellite.
    rmse : float
        Root-mean-square error between data points and tanh fit function.
    tanh_coeffs : (float,) numpy.ndarray
        Array of coefficients of the fitted tanh function.
    residual_coeffs : (float,) numpy.ndarray
        Array of coefficients of the fitted residual function.
    residual_func : func
        The function used in residual fitting.
    fit_func : func
        The final fitting function combining both the tanh function and the residual function.

    Examples
    --------
    >>> from doptools.data import L1B
    >>> s = L1B.create('Delfi-C3_32789_201602121133')
    >>> s.plot()

    """

    def __init__(self, data):
        self.dataid = data['dataid']
        self.recording = Recording(self.dataid)

        # Series data
        self.time = data['time']
        self.time_sec = data['time_sec']
        self.frequency = data['frequency']
        self.power = data['power']

        # Single value data
        self.tca = data['tca']
        self.tca_sec = data['tca_sec']
        self.fca = data['fca']
        self.rmse = data['rmse']

        # Fitting data
        self.tanh_coeffs = data['tanh_coeffs']
        self.residual_coeffs = data['residual_coeffs']
        self.residual_func = data['residual_func']
        self.fit_func = data['fit_func']

    @classmethod
    def create(cls, L1A_object, plot=False):
        """
        Create time-frequency data from spectrogram.

        Parameters
        ----------
        L1A_object : data.L1A
            L1A (spectrogram) object.
        plot : bool, optional
            If True plot figures for each step of the data point extraction process.

        Returns
        -------
        data.L1B
            Time-frequency data object.

        """

        logger.info(f"Extracting frequency data for {L1A_object.dataid}")

        rec = Recording(L1A_object.dataid)
        satpass = SatellitePassTLE.from_recording(L1A_object.dataid, num=10_000)
        tca_sec = (satpass.tca - rec.start_time).total_seconds()

        data = extract_frequency_data(
                L1A_object.spectrogram,
                L1A_object.dt,
                tca_sec,
                plot=plot)

        data['dataid'] = L1A_object.dataid

        data['time_sec'] = data['time']
        data['time'] = np.array([rec.start_time + timedelta(seconds=t) for t in data['time_sec']])

        data['tca'] = satpass.tca
        data['tca_sec'] = tca_sec
        data['fca'] = cls.estimate_fca(data['time_sec'], data['frequency'], data['tca_sec'])

        return cls(data)

    @classmethod
    def load(cls, dataid):
        """
        Load saved time-frequency data from database.

        Parameters
        ----------
        dataid : str
            ID of recording in the database.

        Returns
        -------
        data.L1B
            Time-frequency data object.

        """

        logger.info(f"Loading frequency data for {dataid}")

        rec = Recording(dataid)

        filepath = Database().filepath(dataid, level='L1B')
        data = {'dataid': dataid}

        with open(filepath, 'r') as file:
            data['tca'] = datetime.strptime(file.readline().strip('\n').split('=')[1], "%Y-%m-%d %H:%M:%S.%f")
            data['tca_sec'] = float(file.readline().strip('\n').split('=')[1])
            assert (data['tca'] - rec.start_time).total_seconds() == data['tca_sec']

            data['fca'] = uncertainties.ufloat_fromstr(file.readline().strip('\n').split('=')[1])
            data['rmse'] = float(file.readline().strip('\n').split('=')[1])

            line = file.readline().strip('\n')
            string_coeffs = line.split('=')[1].strip('[]').split()
            data['tanh_coeffs'] = np.array([float(coeff) for coeff in string_coeffs])

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
            data['residual_coeffs'] = np.array(residual_coeffs)

            data['fit_func'] = create_fit_func(
                    data['tanh_coeffs'],
                    data['residual_func'],
                    data['residual_coeffs'])

            file.readline()
            file.readline()

            time, time_sec, frequency, power = [], [], [], []
            for line in file.readlines():
                time.append(datetime.strptime(line.split(',')[0], "%Y-%m-%d %H:%M:%S.%f"))
                time_sec.append(float(line.split(',')[1]))
                frequency.append(float(line.split(',')[2]))
                power.append(float(line.split(',')[3]))

            # time_sec set to time to conform with create method
            data['time'] = np.array(time)
            data['time_sec'] = np.array(time_sec)
            data['frequency'] = np.array(frequency)
            data['power'] = np.array(power)

        return cls(data)

    def save(self):
        """
        Save the time-frequency data to database.

        Parameters
        ----------
        dataid : str
            ID of recording in the database.

        """

        logger.info(f"Saving frequency data for {self.dataid}")

        filepath = Database().paths['L1B'] / f"{self.dataid}.DOP1B"

        with open(filepath, 'w+') as file:
            file.write(f"tcac={self.tca}\n")
            file.write(f"tca_sec={self.tca_sec}\n")
            file.write(f"fca={repr(self.fca)}\n")
            file.write(f"rmse={self.rmse}\n")

            file.write(f"tanh_coeffs={self.tanh_coeffs}\n")

            file.write(f"residual_func={self.residual_func.__name__}\n")
            file.write(f"residual_coeffs={self.residual_coeffs}\n")

            file.write(f"="*20 + "\n")

            file.write('time,time_sec,frequency,power\n')
            for time, time_sec, frequency, power in zip(
                    self.time,
                    self.time_sec,
                    self.frequency,
                    self.power):
                file.write(f"{time},{time_sec:.2f},{frequency:.2f},{power:.6f}\n")

    def plot(self, fit_func=True, savepath=None, L1A=None, cmap='viridis', clim=None, **kwargs):
        """
        Plot the time-frequency data.

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

        fig, ax = plt.subplots(figsize=(16, 9))

        if L1A is not None:
            xlim = (0 - 0.5, L1A.spectrogram.shape[1] - 0.5)
            ylim = (L1A.spectrogram.shape[0]*L1A.dt, 0)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            clim = (0, L1A.spectrogram.mean() + 4*L1A.spectrogram.std()) if not clim else clim
            # TODO fix to take into account different nfft
            ax.imshow(
                    L1A.spectrogram,
                    clim=clim,
                    cmap=cmap,
                    aspect='auto',
                    extent=(xlim[0],
                            xlim[1],
                            ylim[0],
                            ylim[1]),
                    **kwargs)

        markersize = 0.5 if savepath else None
        ax.scatter(self.frequency, self.time_sec, s=markersize, color='r')
        if fit_func:
            times = np.linspace(self.time_sec[0], self.time_sec[-1], 100)
            ax.plot(self.fit_func(times), times, 'k')

        if savepath:
            fig.savefig(savepath, format='png', dpi=300)
            plt.close(fig)
        else:
            fig.show()

    @staticmethod
    def estimate_fca(time_sec, frequency, tca_sec):

        bools = abs(time_sec - tca_sec) < 10

        coeffs, covariance = optimize.curve_fit(
                    fitting.linear,
                    time_sec[bools],
                    frequency[bools],
                    method='lm')

        err = np.sqrt(np.diag(covariance))
        a = uncertainties.ufloat(coeffs[0], err[0])
        b = uncertainties.ufloat(coeffs[1], err[1])

        return a*tca_sec + b


class L1C:
    """
    L1C data (rangerate) of a DopTrack recording.

    Parameters
    ----------
    dataid : str
        ID of recording in the database.
    recording : data.L0
        Recording data object.
    time : (datetime.datetime,) numpy.ndarray
        Time in datetime for each data point.
    time_sec : (float,) numpy.ndarray
        Time in seconds from start of recording for each data point.
    rangerate : (float,) numpy.ndarray

    """

    def __init__(self, dataid, time, time_sec, rangerate):
        self.dataid = dataid
        self.recording = Recording(dataid)
        self.time = time
        self.time_sec = time_sec
        self.rangerate = rangerate

    @classmethod
    def create(cls, L1B_obj):

        dataid = L1B_obj.dataid

        time = L1B_obj.time
        time_sec = L1B_obj.time_sec
        frequency = L1B_obj.frequency + Recording(dataid).tuning_freq
        fca = L1B_obj.fca + Recording(dataid).tuning_freq

        # TODO might want to propogate uncertainties further down the data stream. Remove .n if so.
        rangerate = cls.rangerate_model(frequency, fca.n)

        return cls(dataid, time, time_sec, rangerate)

    @classmethod
    def load(cls, dataid):

        logger.info(f"Loading frequency data for {dataid}")

        filepath = Database().filepath(dataid, level='L1C')
        with open(filepath, 'r') as file:
            file.readline()
            time, time_sec, rangerate = [], [], []
            for line in file.readlines():
                time.append(datetime.strptime(line.split(',')[0], "%Y-%m-%d %H:%M:%S.%f"))
                time_sec.append(float(line.split(',')[1]))
                rangerate.append(float(line.split(',')[2]))

        return cls(dataid, np.array(time), np.array(time_sec), np.array(rangerate))

    def save(self):
        logger.info(f"Saving rangerate data for {self.dataid}")

        filepath = Database().paths['L1C'] / f"{self.dataid}.DOP1C"

        with open(filepath, 'w+') as file:

            file.write('time,time_sec,frequency,power\n')
            for time, time_sec, rangerate in zip(
                    self.time,
                    self.time_sec,
                    self.rangerate):
                file.write(f"{time},{time_sec:.2f},{rangerate:.2f}\n")

    @staticmethod
    def rangerate_model(frequency, carrier_frequency):
        return (1 - (frequency / carrier_frequency)) * constants.c


class L2A:
    def __init__(self):
        raise NotImplementedError
