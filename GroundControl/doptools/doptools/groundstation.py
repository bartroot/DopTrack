import numpy as np
from sgp4.earth_gravity import wgs84
from datetime import timedelta

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

