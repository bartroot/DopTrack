import numpy as np
from datetime import timedelta
from sgp4.model import Satellite
from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs84
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .recording import Recording
from .data import L1B
from .coordconv import teme2ecef, geodetic2ecef, ecef2geodetic
from .utils import GeodeticPosition

radius_earth = wgs84.radiusearthkm * 1000

_geodetic = GeodeticPosition(51.9989, 4.3733585, 95)
_ecef = geodetic2ecef(*_geodetic)
DopTrackStation = {'geodetic': _geodetic, 'ecef': _ecef}


class GroundStation:

    def __init__(self, name, geo):
        self.name
        self.position_geodetic = GeodeticPosition(*geo)
        self.position = geodetic2ecef(*self.position_geodetic)

    def satellite_inview(self, sat_pos_ecef):
        sat_pos_geo = ecef2geodetic(*sat_pos_ecef)

        inner_self_self = np.inner(self.position, self.position)
        inner_sat_sat = np.inner(sat_pos_ecef, sat_pos_ecef)
        inner_self_sat = np.inner(self.position, sat_pos_ecef)

        cosgamma = radius_earth / (radius_earth + sat_pos_geo.altitude)
        satgamma = inner_self_sat / (np.sqrt(inner_sat_sat) * np.sqrt(inner_self_self))
        if satgamma > cosgamma:
            return True
        else:
            return False


class SatelliteSGP4(Satellite):
    def __init__(self, tle_line1, tle_line2, gravity_model=wgs84):
        # run constructor function and copy all the instance variables to this instance
        self.__dict__ = twoline2rv(tle_line1, tle_line2, gravity_model).__dict__

    def construct_track(self, times):
        position = np.zeros((len(times), 3), dtype='float64')
        velocity = np.zeros((len(times), 3), dtype='float64')
        for i, time in enumerate(times):
            pos_eci, vel_eci = self.propagate(time)
            #  Not taking into account changes in length of day. See note for teme2ecef function.
            pos, vel = teme2ecef(time, pos_eci, vel_eci, polarmotion=True, lod=False)
            position[i] = pos
            velocity[i] = vel
        return position, velocity

    def propagate(self, time, **kwargs):
        time_ints = [int(t) for t in time.strftime('%Y %m %d %H %M %S').split()]
        pos, vel = super().propagate(*time_ints, **kwargs)
        return np.array(pos)*1000, np.array(vel)*1000

    def elevation(self, time, station_pos):
        # Not taking flattening into account
        sat_pos, _ = self.propagate(time)
        vec_range = sat_pos - station_pos
        phi = np.arccos(
                np.dot(station_pos, vec_range) /
                (np.linalg.norm(station_pos) * np.linalg.norm(vec_range)))
        return 90 - np.rad2deg(phi)

    def azimuth(time, groundstation):
        raise NotImplementedError


class SatellitePassTLE:

    def __init__(self, tle, time, tca=None):
        self.station = DopTrackStation
        self.satellite = SatelliteSGP4(tle[0], tle[1])

        self.tle = tle
        self.time = time
        self.position, self.velocity = self.satellite.construct_track(self.time)
        self.rangerate = self._calculate_rangerate(self.position, self.velocity, self.station['ecef'])
        self.elevation = np.array([self.satellite.elevation(t, self.station['ecef']) for t in self.time])

        if tca:
            self.tca = tca
        else:
            self.tca = self._calculate_tca(self.time, self.rangerate)

    @classmethod
    def from_recording(cls, dataid, num=1000):
        rec = Recording(dataid)
        dtime = np.linspace(0, rec.duration, num=num)
        time = [rec.start_time + timedelta(seconds=dt) for dt in dtime]

        return cls(rec.prediction['tle'], time)

    @classmethod
    def from_L1B(cls, dataid):
        rec = Recording(dataid)
        temp = cls.from_recording(dataid, num=1000)
        time = L1B.load(dataid).time

        return cls(rec.prediction['tle'], time, tca=temp.tca)

    def plot(self, savepath=None):
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(self.time, self.elevation)
        if savepath:
            fig.savefig(savepath, format='png', dpi=300)
            plt.close(fig)
        else:
            fig.show()

    def plot3d(self, savepath=None):
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(*zip(*self.position))
        ax.scatter(*self.station['ecef'], color='r')
        if savepath:
            fig.savefig(savepath, format='png', dpi=300)
            plt.close(fig)
        else:
            fig.show()

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


