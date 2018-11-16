import numpy as np
from datetime import timedelta
from sgp4.model import Satellite
from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs84

from .coordconv import teme2ecef
from .io import read_meta, read_rre
from .coordconv import geodetic2ecef, ecef2geodetic
from .utils import GeodeticPosition


radius_earth = wgs84.radiusearthkm * 1000

geodetic = GeodeticPosition(51.9989, 4.3733585, 95)
ecef = geodetic2ecef(*geodetic)
DopTrackStation = {'geodetic': geodetic, 'ecef': ecef}


#class GroundStation:
#    """
#    This class creates an object representing a 'station'
#    from which satellites are observed.
#    """
#
#    def __init__(self, geo):
#        self.position_geodetic = GeodeticPosition(*geo)
#        self.position = geodetic2ecef(*self.position_geodetic)
#
#
#    def satellite_inview(self, sat_pos_ecef):
#        """Checks if satellite is in view of station.
#
#        Args:
#            sat_pos_ecef: position of satellite
#
#        Returns:
#            bool: True if satellite is in view. False if not.
#        """
#        # TODO Make this part more readable
#        sat_pos_geo = ecef2geodetic(*sat_pos_ecef)
#
#        inner_self_self = np.inner(self.position, self.position)
#        inner_sat_sat = np.inner(sat_pos_ecef, sat_pos_ecef)
#        inner_self_sat = np.inner(self.position, sat_pos_ecef)
#
#        cosgamma = radius_earth / (radius_earth + sat_pos_geo.altitude)
#        satgamma = inner_self_sat / (np.sqrt(inner_sat_sat) * np.sqrt(inner_self_self))
#        if satgamma > cosgamma:
#            return True
#        else:
#            return False


class SatelliteSGP4(Satellite):
    def __init__(self, tle_line1, tle_line2, gravity_model=wgs84):
        # run constructor function and copy all the instance variables to this instance
        self.__dict__ = twoline2rv(tle_line1, tle_line2, gravity_model).__dict__

    def construct_track(self, times):
        position = np.zeros((len(times), 3), dtype='float64')
        velocity = np.zeros((len(times), 3), dtype='float64')
        for i, time in enumerate(times):
            time_ints = [int(t) for t in time.strftime('%Y %m %d %H %M %S').split()]
            pos_eci, vel_eci = self.propagate(*time_ints)
            pos, vel = teme2ecef(time, pos_eci, vel_eci, polarmotion=False, lod=False)
            position[i] = pos
            velocity[i] = vel
        return position, velocity

    def propagate(self, *args, **kwargs):
        pos, vel = super().propagate(*args, **kwargs)
        return np.array(pos)*1000, np.array(vel)*1000


class SatellitePassTLE:

    def __init__(self, tle, times):
        self.station = DopTrackStation
        self.time = times
        self.tle = tle
        self.sgp4 = SatelliteSGP4(tle[0], tle[1])
        self.position, self.velocity = self.sgp4.construct_track(self.time)
        self.rangerate = self._calculate_rangerate(self.position,
                                                   self.velocity,
                                                   self.station['ecef'])
        self.tca = self._calculate_tca(self.time, self.rangerate)

    @staticmethod
    def _calculate_rangerate(positions, velocities, station_position):
        rangerates = np.zeros(len(positions), dtype=float)
        for i, (position, velocity) in enumerate(zip(positions, velocities)):
            direction = position - station_position
            rangerates[i] = np.inner(velocity, direction) / np.sqrt(np.inner(direction, direction))
        return rangerates

    @staticmethod
    def _calculate_tca(times, rangerates):
        if 0 not in rangerates:
            try:
                idx_first_positive = np.where(rangerates > 0)[0][0]
                idx_last_negative = idx_first_positive - 1
            except IndexError:
                return np.NaN
            fp = [0, (times[idx_first_positive] - times[idx_last_negative]).total_seconds()]
            xp = [rangerates[idx_last_negative], rangerates[idx_first_positive]]
            dt = np.interp([0], xp, fp)[0]
            tca = times[idx_last_negative] + timedelta(seconds=dt)
        else:
            tca = times[np.where(rangerates == 0)]
        return tca

    @classmethod
    def from_dataid(cls, dataid):
        times = read_rre(dataid)['datetime']
        meta = read_meta(dataid)
        tle = [meta['Sat']['Predict']['used TLE line1'],
               meta['Sat']['Predict']['used TLE line2']]
        return cls(tle, times)


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
