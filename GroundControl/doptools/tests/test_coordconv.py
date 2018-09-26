import pytest
from datetime import datetime
import numpy as np

import doptools.coordconv as coordconv


@pytest.fixture(scope='module')
def time1991():
    """vallado1997, example 1-6, p.87"""
    utc = datetime(1991, 4, 6, 7, 51, 28, 386200)
    time = coordconv.DatetimeConverter(utc)
    time.dut1 = 0.40233
    time.dat = 26
    return time


@pytest.fixture(scope='module')
def time2004():
    utc = datetime(2004, 4, 6, 7, 51, 28, 386000)
    time = coordconv.DatetimeConverter(utc)
    time.dut1 = -0.439961
    time.dat = 32
    return time


class TestDatetimeConverter():
    def test_time2juliandate(self):
        """vallado1997, example 1-4, p.68"""
        time = datetime(1996, 10, 26, 14, 20)
        jd = coordconv.DatetimeConverter.time2juliandate(time)
        np.testing.assert_almost_equal(jd, 2450383.09722222, decimal=8)

    def test_ut1(self, time1991):
        """vallado1997, example 1-6, p.87"""
        assert time1991.ut1 == datetime(1991, 4, 6, 7, 51, 28, 788530)

    def test_tdt(self, time1991):
        """vallado1997, example 1-6, p.87"""
        assert time1991.tdt == datetime(1991, 4, 6, 7, 52, 26, 570200)

    def test_tdb(self, time1991):
        """vallado1997, example 1-6, p.87"""
        assert time1991.tdb == datetime(1991, 4, 6, 7, 52, 26, 571857)

    def test_JDtdt(self, time1991):
        """vallado1997, example 1-6, p.87"""
        np.testing.assert_almost_equal(time1991.JDtdt, 2448352.8280853032, decimal=10)

    def test_JDtdb(self, time1991):
        """vallado1997, example 1-6, p.87"""
        np.testing.assert_almost_equal(time1991.JDtdb, 2448352.8280853224, decimal=10)

    def test_Ttdt(self, time1991):
        """vallado1997, example 1-6, p.87"""
        np.testing.assert_almost_equal(time1991.Ttdt, -0.087396904, decimal=9)

    def test_Ttdb(self, time1991):
        """vallado1997, example 1-6, p.87"""
        np.testing.assert_almost_equal(time1991.Ttdb, -0.08739690, decimal=8)


def test_ecef2geodetic():
    """satellite orbits 3rd 2000 p.192"""
    geodetic = (-7.26654999, 72.36312094, -63.667)
    ecef = (1917032.190, 6029782.349, -801376.113)
    np.testing.assert_array_almost_equal(coordconv.ecef2geodetic(*ecef), geodetic, decimal=3)


def test_geodetic2ecef():
    """satellite orbits 3rd 2000 p.192"""
    geodetic = (-7.26654999, 72.36312094, -63.667)
    ecef = (1917032.190, 6029782.349, -801376.113)
    np.testing.assert_array_almost_equal(coordconv.geodetic2ecef(*geodetic), ecef, decimal=3)


def test_teme2ecef_position():
    """spacetrack #3 revision 2, p.32. Does not achieve exact accuracy as reference (5 decimals)."""
    utc = datetime(2004, 4, 6, 7, 51, 28, 386000)
    r_teme = np.array([5094180.10720, 6127644.70520, 6380344.53270])
    v_teme = np.array([-4746.131494, 785.817998, 5531.931288])
    r_ecef = np.array([-1033479.38300, 7901295.27540, 6380356.59580])
    polar_params = (np.deg2rad(-0.140682/3600), np.deg2rad(0.333309/3600))
    dut1, dat = -0.439961, 32
    r_ecef_test, v_ecef_test = coordconv.teme2ecef(utc,
                                                   r_teme,
                                                   v_teme,
                                                   polarmotion=polar_params,
                                                   dut1=dut1,
                                                   dat=dat)
    np.testing.assert_array_almost_equal(r_ecef_test, r_ecef, decimal=1)


def test_teme2ecef_velocity():
    """spacetrack #3 revision 2, p.32. Does not achieve exact accuracy as reference (6 decimals)."""
    utc = datetime(2004, 4, 6, 7, 51, 28, 386000)
    r_teme = np.array([5094180.10720, 6127644.70520, 6380344.53270])
    v_teme = np.array([-4746.131494, 785.817998, 5531.931288])
    v_ecef = np.array([-3225.636520, -2872.451450, 5531.924446])
    polar_params = (np.deg2rad(-0.140682/3600), np.deg2rad(0.333309/3600))
    dut1, dat = -0.439961, 32
    r_ecef_test, v_ecef_test = coordconv.teme2ecef(utc,
                                                   r_teme,
                                                   v_teme,
                                                   polarmotion=polar_params,
                                                   dut1=dut1,
                                                   dat=dat)
    np.testing.assert_array_almost_equal(v_ecef_test, v_ecef, decimal=5)


def test_gmst1982(time1991):
    """Based on manual calculation. Test case from reference would be preferable."""
    np.testing.assert_almost_equal(coordconv.gmst1982(time1991.Tut1), 5.4449747718)


def test_nutation_m_moon(time1991):
    """vallado1997, example 1-6, p.88"""
    M_moon = coordconv._nutation_m_moon(time1991.Ttdb)
    np.testing.assert_almost_equal(np.rad2deg(M_moon), -170.74050383 + 360)


def test_nutation_m_sun(time1991):
    """vallado1997, example 1-6, p.88"""
    M_sun = coordconv._nutation_m_sun(time1991.Ttdb)
    np.testing.assert_almost_equal(np.rad2deg(M_sun), -268.6778207 + 360, decimal=5)


def test_nutation_u_m_moon(time1991):
    """vallado1997, example 1-6, p.88"""
    u_M_moon = coordconv._nutation_u_m_moon(time1991.Ttdb)
    np.testing.assert_almost_equal(np.rad2deg(u_M_moon), -17.088405322 + 360)


def test_nutation_d_sun(time1991):
    """vallado1997, example 1-6, p.88"""
    D_sun = coordconv._nutation_d_sun(time1991.Ttdb)
    np.testing.assert_almost_equal(np.rad2deg(D_sun), -97.11660008 + 360)


def test_nutation_omega_moon(time1991):
    """vallado1997, example 1-6, p.88"""
    Omega_moon = coordconv._nutation_omega_moon(time1991.Ttdb)
    np.testing.assert_almost_equal(np.rad2deg(Omega_moon), 294.0820589)


def test_nutation_Dpsi(time1991):
    """vallado1997, example 1-6, p.88"""
    Dpsi, Deps = coordconv._nutation_Dpsi_Deps(time1991.Ttdb)
    np.testing.assert_almost_equal(np.rad2deg(Dpsi), 0.004185849)


def test_nutation_Deps(time1991):
    """vallado1997, example 1-6, p.88"""
    Dpsi, Deps = coordconv._nutation_Dpsi_Deps(time1991.Ttdb)
    np.testing.assert_almost_equal(np.rad2deg(Deps), 0.00117066)


def test_nutation_epsbar(time1991):
    """vallado1997, example 1-6, p.88"""
    epsbar = coordconv._nutation_epsbar(time1991.Ttdb)
    np.testing.assert_almost_equal(np.rad2deg(epsbar), 23.440427525)


def test_precession_zeta(time1991):
    """vallado1997, example 1-6, p.87"""
    zeta = coordconv._precession_zeta(time1991.Ttdb)
    np.testing.assert_almost_equal(np.rad2deg(zeta), -0.05598722)


def test_precession_z(time1991):
    """vallado1997, example 1-6, p.87"""
    z = coordconv._precession_z(time1991.Ttdb)
    np.testing.assert_almost_equal(np.rad2deg(z), -0.05598554)


def test_precession_Theta(time1991):
    """vallado1997, example 1-6, p.87"""
    Theta = coordconv._precession_Theta(time1991.Ttdb)
    np.testing.assert_almost_equal(np.rad2deg(Theta), -0.04865938)


def test_mean_anomaly_earth():
    Ttdt = -0.087396904
    M = coordconv._mean_anomaly_earth(Ttdt)
    np.testing.assert_almost_equal(np.rad2deg(M), -268.6778195 + 360, decimal=5)


def test_precession_matrix_zero():
    m = coordconv._precession_matrix(0, 0, 0)
    np.testing.assert_array_equal(m, np.identity(3))


def test_sidereal_matrix_zero():
    m = coordconv._sidereal_matrix(0)
    np.testing.assert_array_equal(m, np.identity(3))


def test_nutation_matrix_zero():
    m = coordconv._nutation_matrix(0, 0, 0)
    np.testing.assert_array_equal(m, np.identity(3))
