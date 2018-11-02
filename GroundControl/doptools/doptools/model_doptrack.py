import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.constants as constants
from scipy.fftpack import fft, fftshift
from tqdm import tqdm

from .io import Database, read_meta, read_rre
from .groundstation import DopTrackStation


#class Metadata:
#
#    def __init__(self, metadata):
#        self.__dict__.update(metadata)
#
#    def read(cls, dataid):
#        with open(os.path.join(DATA_DIR, f'{dataid}.yml'), 'r') as metafile:
#            metadata = yaml.load(metafile)
#        return cls(metadata)
#
#    def create(cls):
#        raise NotImplementedError


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
        database = Database()
        samples = int(self.sample_freq * self.duration)
        timesteps = int(self.duration / dt)
        cutoff = int(2 * samples / timesteps)
        with open(database.get_filepath(self.dataid, '32fc'), 'r') as file:
            for i in range(timesteps):
                yield np.fromfile(file, dtype=np.float32, count=cutoff)


class SatellitePassRecorded:
    """
    Spectrogram of a DopTrack recording.

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

    """
    def __init__(self, dataid):
        self.station = DopTrackStation
        self.dataid = dataid
        self.recording = Recording(dataid)

        rre = read_rre(self.dataid)
        self.time = rre['datetime']
        self.tca = rre['tca']
        self.frequency = np.array(rre['frequency'])
        self.fca = rre['fca']
        self.rangerate = self._rangerate_model(self.frequency, self.fca)

    @staticmethod
    def _rangerate_model(frequency, fca):
        return (1 - (frequency/fca)) * constants.c


class Spectrogram:
    """
    Spectrogram of a DopTrack recording.

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
        self.image = spectrogram
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
        0.5, or 0.2 should work, but non-"nice" values like 0.236687 will
        most likely raise an array broadcast exception.
        """

        # Determine the limits of the spectrogram.
        recording = Recording(dataid)
        freq_lims = (recording.tuning_freq - recording.sample_freq/2,
                     recording.tuning_freq + recording.sample_freq/2)
        time_lims = (recording.stop_time, recording.start_time)

        # Read data, cut data, and perform FFT for each timestep.
        timesteps = int(recording.duration / dt)
        spectrogram = np.zeros((timesteps, nfft))
        for i, raw_data in tqdm(enumerate(recording.data(dt)), total=timesteps):
            signal = np.zeros(int(len(raw_data)/2), dtype=np.complex)
            signal.real = raw_data[::2]
            signal.imag = -raw_data[1::2]
            spectrogram[i] = cls._construct_spectrum(signal, nfft)

        # Zoom spectrogram if bounds are given.
        if bounds:
            spectrogram, freq_lims = cls._zoom(spectrogram,
                                               freq_lims,
                                               recording.tuning_freq,
                                               bounds)

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
        datafilepath, metafilepath = Database().get_filepath(dataid, 'npy')
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
        folderpath = Database().paths['spectrograms']

        if filename is None:
            filename = f'{self.dataid}_{int(self.dt * 10)}'

        with open(folderpath / f'{filename}.npy.meta', 'w+') as file:
            file.write(f'xlim_lower={self.freq_lims[0]}\n')
            file.write(f'xlim_upper={self.freq_lims[1]}\n')
            file.write(f'ylim_lower={self.time_lims[0]}\n')
            file.write(f'ylim_upper={self.time_lims[1]}')

        np.save(folderpath / f'{filename}.npy', self.image)

    def plot(self, cmap='jet', clim=(-75, -50), **kwargs):
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

        fig, ax = plt.subplots()

        # Create new axis for colorbar.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        # Plot spectrogram with defined limits.
        im = ax.imshow(self.image,
                       extent=(self.freq_lims[0],
                               self.freq_lims[1],
                               num_time_lims[0],
                               num_time_lims[1]),
                       aspect="auto", cmap=cmap, clim=clim, **kwargs)

        # Format time axis and plot colorbar.
        self._axis_format_totime(ax.yaxis)
        fig.colorbar(im, cax=cax, orientation='vertical')

        fig.show()

    @staticmethod
    def _zoom(spectrogram, freq_lims, tuning_freq, bounds):
        """
        Create a zoomed version of a spectrogram.

        Parameters
        ----------
        spectrogram : (N, M) numpy.ndarray
            An array containing the values of the spectrogram in dB.
        freq_lims : (float, float) tuple
            The minimum and maximum frequency values of the spectrogram.
        tuning_freq : float
            The tuning frequency of the recorded signal.
        bounds : (float, float) tuple, optional
            The lower and upper bounds of the desired frequency range
            relative to the tuning frequency.

        Returns
        -------
        spectrogram : (N, M) numpy.ndarray
            An array containing the values of the zoomed spectrogram in dB.
        freq_lims : (float, float) tuple
            The new minimum and maximum frequency values of the zoomed spectrogram.
        """

        # Create array with all frequency values.
        bandwidth = np.linspace(freq_lims[0],
                                freq_lims[1],
                                spectrogram.shape[1])
        lower = tuning_freq + bounds[0]
        upper = tuning_freq + bounds[1]

        estimated_signal_width = 7000
        estimated_signal_freq = 145_888_300

        lower = estimated_signal_freq - estimated_signal_width
        upper = estimated_signal_freq + estimated_signal_width

        # Create a mask for desired frequency values.
        mask, = np.nonzero((lower <= bandwidth) & (bandwidth <= upper))
        bandwidth = bandwidth[mask]

        # Apply mask to reduce spectrogram down to desired size.
        spectrogram = (spectrogram.transpose()[mask]).transpose()
        freq_lims = (bandwidth[0], bandwidth[-1])

        return spectrogram, freq_lims

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
        # Perform FFT.
        spectrum = fft(signal, nfft)
        spectrum = fftshift(spectrum)

        spectrum = abs(spectrum)

#        plt.figure()
#        plt.plot(range(len(spectrum)), spectrum)
#
#        plt.figure()
#        plt.plot(range(len(spectrum1)), spectrum1)
#
#        plt.show()
#        sys.exit()

        # Calculate relative power level in dB of spectrum.
        # TODO should be fixed to dimensionless values but requires changes in drre
#        spectrum = 10*np.log10(2*abs(spectrum)/)
#        spectrum = 10*np.log10(2*abs(spectrum)/len(signal))

        return spectrum

    @staticmethod
    def _axis_format_totime(axis):
        """Format normal axis to time axis for plotting."""
        date_format = mdates.DateFormatter('%H:%M:%S')
        axis.set_major_formatter(date_format)
        axis.set_major_locator(mdates.MinuteLocator(interval=2))
