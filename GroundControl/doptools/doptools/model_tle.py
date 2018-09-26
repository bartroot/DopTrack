import numpy as np
from datetime import timedelta
from sgp4.model import Satellite
from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs84

from .coordconv import teme2ecef
from .io import read_meta, read_rre
from .groundstation import DopTrackStation


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
