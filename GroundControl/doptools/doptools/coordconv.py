"""Coordinate conversion functions.

This module contains functions for transformations between the different
reference frames used in doppler tracking.

Constants
---------
- `WGS84_omega` -- Earth's rotational speed.
- `WGS84_a` -- Earth's radius.
- `WGS84_e` -- Earth's eccentricity.

Data
----
- `NUTATION_COEFFS` -- Coefficients used in nutation theory.
- `EOPP` -- Earth orientation parameters Xp, Yp.
- `EOPC04` -- Values for UT1 difference (DUT1) and length of day (LOD)
- `DAT` -- Values for TAI difference (DAT).

Classes
-------
- `DatetimeConverter` -- Class for timestamp conversions.

Routines
--------
- `ecef2geodetic` -- Convert coordinates from ECEF to Geodetic.
- `geodetic2ecef` -- Convert coordinates from Geodetic to ECEF.
- `teme2ecef` -- Convert coordinates from TEME to ECEF.
- `gmst1982` -- Calculate a (deprecated) version of Greenwhich Mean Sidereal Time

"""
import numpy as np
from datetime import timedelta

from .utils import Position, GeodeticPosition
from .io import read_nutation_coeffs, read_eopp, read_eopc04, read_tai_utc


WGS84_omega = 7.2921151467e-05
WGS84_a = 6378137.0
WGS84_e = 0.08181919092890638

# TODO determine what happens when files are not found
NUTATION_COEFFS = read_nutation_coeffs()
EOPP = read_eopp()
EOPC04 = read_eopc04()
DAT = read_tai_utc()


class DatetimeConverter():
    """Class for converting timestamps between common formats.

    Attributes
    ----------
    utc : datetime.datetime
        Timestamp in Coordinated Universal Time (UTC).
    DUT1 : float
        Difference between UTC and UT1.
    DAT: float
        Difference between UTC and TAI.

    """

    def __init__(self, utc, dut1=None, dat=None):
        self.utc = utc
        if dut1:
            self.dut1 = dut1
        else:
            i = EOPC04.index.searchsorted(self.utc) - 1
            self.dut1 = EOPC04.loc[EOPC04.index[i]]['DUT1']
        if dat:
            self.dat = dat
        else:
            i = DAT.index.searchsorted(self.utc) - 1
            self.dat = DAT.loc[DAT.index[i]]['DAT']

    @property
    def ut1(self):
        """Timestamp in Universal Time (UT1)."""
        return self.utc + timedelta(seconds=self.dut1)

    @property
    def tai(self):
        """Timestamp in International Atomic Time (TAI)."""
        return self.utc + timedelta(seconds=self.dat)

    @property
    def tdt(self):
        """Timestamp in Terrestrial "Dynamical" Time (TDT/TT)."""
        return self.tai + timedelta(seconds=32.184)

    @property
    def tdb(self):
        """Timestamp in Barycentric Dynamical Time (TDB)."""
        M_earth = 6.240035939 + 628.3019560*self.Ttdt
        dt = (0.001658*np.sin(M_earth) + 0.00001385*np.sin(2*M_earth))
        return self.tdt + timedelta(seconds=dt)

    @property
    def MJDutc(self):
        """Modified Julian Date of Coordinated Universal Time (UTC)."""
        return self.JDutc - 2400000.5

    @property
    def JDutc(self):
        """Julian Date of Coordinated Universal Time (UTC)."""
        return self.time2juliandate(self.utc)

    @property
    def JDut1(self):
        """Julian Date of Universal Time (UT1)."""
        return self.time2juliandate(self.ut1)

    @property
    def JDtdt(self):
        """Julian Date of Terrestrial "Dynamical" Time (TDT/TT)."""
        return self.time2juliandate(self.tdt)

    @property
    def JDtdb(self):
        """Julian Date of Barycentric Dynamical Time (TDB)."""
        return self.time2juliandate(self.tdb)

    @property
    def Tut1(self):
        """Universal Time (UT1) in Julian Centuries."""
        return (self.JDut1 - 2451545) / 36525

    @property
    def Ttdt(self):
        """Terrestrial "Dynamical" Time (TDT/TT) in Julian Centuries."""
        return (self.JDtdt - 2451545) / 36525

    @property
    def Ttdb(self):
        """Barycentric Dynamical Time (TDB) in Julian Centuries."""
        return (self.JDtdb - 2451545) / 36525

    @staticmethod
    def time2juliandate(time):
        """Calculate Julian Date from a timestamp.

        Parameters
        ----------
        time : datetime.datetime
            Timestamp.

        Returns
        -------
        float

        """
        jd = (367*time.year
              - np.floor(7*(time.year + np.floor((time.month + 9)/12))/4)
              + np.floor(275*time.month/9)
              + time.day
              + 1721013.5
              + ((((time.second+time.microsecond/1000000)/60) + time.minute)/60 + time.hour)/24
              )
        return jd


def ecef2geodetic(x, y, z):
    """
    Convert and return position from ECEF coordinates
    to latitude, longitude, and altitude, using the WGS84 geo model.
    """

    lon = np.arctan2(y, x)
    alt = 0
    n = WGS84_a
    p = np.sqrt(x**2 + y**2)
    lat = 0
    previous_lat = 90

    # Iterate until tolerance is reached
    while abs(lat - previous_lat) >= 1e-9:
        previous_lat = lat
        sin_lat = z / (n * (1 - WGS84_e**2) + alt)
        lat = np.arctan((z + WGS84_e**2 * n * sin_lat) / p)
        n = WGS84_a / np.sqrt(1 - (WGS84_e * sin_lat)**2)
        alt = p / np.cos(lat) - n

    lon = np.rad2deg(lon)
    lat = np.rad2deg(lat)

    return GeodeticPosition(lat, lon, alt)


def geodetic2ecef(lat, lon, alt):
    """Reference frame transformation from Geodetic to ECEF.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    alt : float
        Altitude in [m].


    Returns
    -------
    tuple
        Position vector in the ECEF frame in [m].

    See Also
    --------
    ecef2geodetic : Transform from ECEF to Geodetic

    References
    ----------
    .. [1] Montenbruck, O. and Gill, E.,
        "Satellite Orbits",
        1st Edition,
        p. 187-189, 2005.

    Examples
    --------
    >>> geodetic2ecef(-7.26654999, 72.36312094, -63.667)
    (1917032.190, 6029782.349, -801376.113)
    """
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    N = WGS84_a / np.sqrt(1 - (WGS84_e * np.sin(lat))**2)

    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - WGS84_e**2) + alt) * np.sin(lat)

    return Position(x, y, z)


def teme2ecef(utc, r_teme, v_teme, polarmotion=True, lod=True, **kwargs):
    """Reference frame transformation from TEME to ECEF.

    Transforms position and velocity vectors from the true
    equator mean equinox (TEME) frame to an earth centered
    earth fixed (ECEF/ITRF) frame, taking into account sidereal
    time and polar motion.

    Parameters
    ----------
    utc : datetime
        Timestamp in UTC.
    r_teme : (3,) ndarray
        Position vector in the TEME frame.
    v_teme : (3,) ndarray
        Velocity vector in the TEME frame.
    polarmotion : bool or (xp, yp) tuple
        Take into account polar motion. If True find polar motion params
        from .eopp files. Specific polar motion params can be given as a tuple.
    lod : bool
        Take into account changes in the length of day.

    Returns
    -------
    r_ecef : (3,) ndarray
        Position vector in the ECEF frame.
    v_ecef: (3,) ndarray
        Velocity vector in the ECEF frame.

    Warnings
    --------
    The current implementation uses EOPP data (see doptools.ftp) which only gets
    downloaded from 2016-01-01. Any transformation at a time before this date will
    instead ignore polar motion, even if the polarmotion flag is set to True.

    Notes
    -----
    The units of the position and veloty vectors should be at same scale
    and the resulting vectors will have the same units.

    Taking into account changes in length of day should have near zero influence
    on results, even at extreme accuracy. Taking into polar motion will result in
    about 1km difference in position for a LEO satellite. These conclusions are
    from testing only.

    This function is adapted from the ``teme2ecef.m`` script by Vallado
    published on the Celestrak website. See [1]_.

    References
    ----------
    .. [1] Kelso et al.,
        "Revisiting Spacetrack Report #3",
        eq. C-2, 2006.
    """
    time = DatetimeConverter(utc, **kwargs)
    omega = _nutation_omega_moon(time.Ttdt)
    gmst = gmst1982(time.Tut1)
    if time.JDut1 > 2450449.5:
        gmst = gmst + _eqequinoxes_corr(omega)

    ST = _sidereal_matrix(gmst)
    r_pef = ST.dot(r_teme)
    if lod:
        lod_val = EOPC04.truncate(after=time.utc).iloc[-1]['LOD']
        thetasa = WGS84_omega*(1 - lod_val/86400)
    else:
        thetasa = WGS84_omega
    v_pef = ST.dot(v_teme) - np.cross(np.array([0, 0, thetasa]), r_pef)

    if polarmotion and time.utc.year >= 2016:
        if type(polarmotion) == tuple:
            xp, yp = polarmotion
        else:
            polar_params = EOPP.truncate(after=time.MJDutc).iloc[-1]
            xp, yp = polar_params['Xp'], polar_params['Yp']
        PM = _polar_matrix(xp, yp)
        r_ecef = PM.dot(r_pef)
        v_ecef = PM.dot(v_pef)
    else:
        r_ecef = r_pef
        v_ecef = v_pef

    return r_ecef, v_ecef


def gmst1982(Tut1):
    """Deprecated version of Greenwich Mean Sidereal Time (GMST).

    Note
    ----
    Should only be used in transformations to and from
    the TEME reference frame.

    Parameters
    ----------
    Tut1 : int
        UT1 in julian centuries since the epoch J2000.

    Returns
    -------
    gmst : rad

    Notes
    -----
    Originally defined in [1]_. Used in [2]_ and [3]_ for
    transformations to and from the TEME reference frame.

    References
    ----------
    .. [1] Aoki et al., "The New Definition of Universal Time", eq. 14, 1982.
    .. [2] McCarthy, "IERS Technical Note 13", p. 30, 1992.
    .. [3] Kelso et al., "Revisiting Spacetrack Report #3", eq. C-5, 2006.

    """
    gmst = (67310.54841
            + (876600.0 * 3600.0 + 8640184.812866)*Tut1
            + 0.093104*Tut1**2
            - 6.2e-6*Tut1**3) * 360 / 86400
    return np.deg2rad(gmst % 360)


def _eqequinoxes_corr(omega):
    return np.deg2rad((0.00264*np.sin(omega) + 0.000063*np.sin(2*omega)) / 3600)


def _rotation_matrix(version, ang):
    if version == 1:
        return np.array([[1, 0, 0],
                         [0, np.cos(ang), np.sin(ang)],
                         [0, -np.sin(ang), np.cos(ang)]])
    elif version == 2:
        return np.array([[np.cos(ang), 0, -np.sin(ang)],
                         [0, 1, 0],
                         [np.sin(ang), 0, np.cos(ang)]])
    elif version == 3:
        return np.array([[np.cos(ang), np.sin(ang), 0],
                         [-np.sin(ang), np.cos(ang), 0],
                         [0, 0, 1]])


def _nutation_Dpsi_Deps(Ttdb, expansion_order=106):
    a1 = np.array(NUTATION_COEFFS.loc[:, 'a1'])[:expansion_order]
    a2 = np.array(NUTATION_COEFFS.loc[:, 'a2'])[:expansion_order]
    a3 = np.array(NUTATION_COEFFS.loc[:, 'a3'])[:expansion_order]
    a4 = np.array(NUTATION_COEFFS.loc[:, 'a4'])[:expansion_order]
    a5 = np.array(NUTATION_COEFFS.loc[:, 'a5'])[:expansion_order]
    Ai = np.array(NUTATION_COEFFS.loc[:, 'Ai'])[:expansion_order]*0.0001/3600
    Bi = np.array(NUTATION_COEFFS.loc[:, 'Bi'])[:expansion_order]*0.0001/3600
    Ci = np.array(NUTATION_COEFFS.loc[:, 'Ci'])[:expansion_order]*0.0001/3600
    Di = np.array(NUTATION_COEFFS.loc[:, 'Di'])[:expansion_order]*0.0001/3600

    M_moon = _nutation_m_moon(Ttdb)
    M_sun = _nutation_m_sun(Ttdb)
    u_M_moon = _nutation_u_m_moon(Ttdb)
    D_sun = _nutation_d_sun(Ttdb)
    Omega_moon = _nutation_omega_moon(Ttdb)

    angles = a1*M_moon + a2*M_sun + a3*u_M_moon + a4*D_sun + a5*Omega_moon

    Dpsi = np.deg2rad(np.sum((Ai + Bi*Ttdb) * np.sin(angles)))
    Deps = np.deg2rad(np.sum((Ci + Di*Ttdb) * np.cos(angles)))
    return Dpsi, Deps


def _nutation_m_moon(Ttdb):
    return (2.355548394
            + (1325*(2*np.pi) + 3.470890872)*Ttdb
            + 0.000151795*Ttdb**2
            + 3.103e-7*Ttdb**3) % (2*np.pi)


def _nutation_m_sun(Ttdb):
    return (6.2400359
            + (99*(2*np.pi) + 6.2666106)*Ttdb
            - 0.0000027974*Ttdb**2
            - 5.81e-8*Ttdb**3) % (2*np.pi)


def _nutation_u_m_moon(Ttdb):
    return (1.627901934
            + (1342*(2*np.pi) + 1.431476084)*Ttdb
            - 0.000064272*Ttdb**2
            + 5.34e-8*Ttdb**3) % (2*np.pi)


def _nutation_d_sun(Ttdb):
    return (5.198469514
            + (1236*(2*np.pi) + 5.36010650)*Ttdb
            - 0.0000334086*Ttdb**2
            + 9.22e-8*Ttdb**3) % (2*np.pi)


def _nutation_omega_moon(Ttdb):
    return (2.182438624
            - (5*(2*np.pi) + 2.341119397)*Ttdb
            + 0.000036142*Ttdb**2
            + 3.87e-8*Ttdb**3) % (2*np.pi)


def _nutation_epsbar(Ttdb):
    return (0.4090928 - 0.000226966*Ttdb - 2.86e-9*Ttdb**2 + 8.8e-9*Ttdb**3) % (2*np.pi)


def _precession_zeta(Ttdb):
    return 0.01118086*Ttdb + 1.464e-6*Ttdb**2 + 8.7e-8*Ttdb**3


def _precession_z(Ttdb):
    return 0.01118086*Ttdb + 5.308e-6*Ttdb**2 + 8.9e-8*Ttdb**3


def _precession_Theta(Ttdb):
    return 0.009717173*Ttdb - 2.068e-6*Ttdb**2 - 2.02e-7*Ttdb**3


def _mean_anomaly_earth(Ttdt):
    return (6.240035939 + 628.3019560*Ttdt) % (2*np.pi)


def _precession_matrix(zeta, z, Theta):
    A = _rotation_matrix(3, -z)
    B = _rotation_matrix(2, Theta)
    C = _rotation_matrix(3, -zeta)
    return A @ B @ C


def _nutation_matrix(eps, Dpsi, epsbar):
    A = _rotation_matrix(1, -eps)
    B = _rotation_matrix(3, -Dpsi)
    C = _rotation_matrix(1, epsbar)
    return A @ B @ C


def _sidereal_matrix(gst):
    return _rotation_matrix(3, gst)


def _polar_matrix(xp, yp):
    """1980 theory only. maybe consolidate with iau2000. See polarm.m script."""
    A = _rotation_matrix(2, -xp)
    B = _rotation_matrix(1, -yp)
    return A @ B
