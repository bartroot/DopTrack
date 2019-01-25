import os
import pytest
import numpy as np
from shutil import copyfile

import doptools.data as data
from doptools.io import Database


@pytest.fixture(scope='module')
def dataid():
    return 'Delfi-C3_32789_201602210946'

class TestL0:
    def test_available_data(self, dataid):
        d = data.L0(dataid)
        assert d.recording.dataid == dataid == d.dataid

    def test_unavailable_data(self):
        dataid = 'Delfi-C3_32789_000000000000'
        with pytest.raises(FileNotFoundError, message='L0 data file should not be available in database'):
            data.L0(dataid)

    def test_bad_dataid(self):
        dataid = 'Delfi-C3_32789'
        with pytest.raises(TypeError, message='DataID should be wrong'):
            data.L0(dataid)

    def test_data_yields_array_with_correct_length(self, dataid):
        dt = 0.1
        d = data.L0(dataid)
        a = next(d.data(dt))
        assert len(a) == 2 * d.recording.sample_freq * dt


class TestL1A:
    def test_construct_spectrum(self):
        input_data = np.arange(2000)
        data.L1A._construct_spectrum(input_data, nfft=10000)

    @pytest.mark.parametrize('input_data', [False, True, None, 0, 1, set([1, 2, 3])])
    def test_construct_spectrum_with_bad_input(self, input_data):
        with pytest.raises(TypeError, message='Expecting TypeError'):
            data.L1A._construct_spectrum(input_data)

    def test_to_decibel(self):
        a = np.array([0, 1, 100])
        b = data.L1A._to_decibel(a)
        assert all(b == np.array([-np.inf, 0, 20]))


class TestL1B:
    def test_estimate_fca(self):
        time_sec = np.arange(10)
        frequency = np.arange(10)
        tca_sec = 3
        fca = data.L1B.estimate_fca(time_sec, frequency, tca_sec)
        assert fca.nominal_value == pytest.approx(3)


def test_data_integration(dataid):
    db = Database()
    if dataid not in db.dataids['L0']:
        pytest.skip(msg=f'Skipping data integration since L0 file not available: {dataid}')

    recpath = db.filepath(dataid, level='L0', meta=True)
    new_recpath = recpath.parent / 'test_00000_000000000000.yml'
    copyfile(recpath, new_recpath)

    s = data.L1A.create(dataid)
    test_dataid = 'test_00000_000000000000'
    s.dataid = test_dataid
    s.save()
    s.load(test_dataid)
    # Cleanup
    path = db.filepath(test_dataid, level='L1A')
    os.remove(path)
    path = db.filepath(test_dataid, level='L1A', meta=True)
    os.remove(path)
    s.dataid = dataid

    d = data.L1B.create(s)
    test_dataid = 'test_00000_000000000000'
    d.dataid = test_dataid
    d.save()
    d.load(test_dataid)
    # Cleanup
    path = db.filepath(test_dataid, level='L1B')
    os.remove(path)
    d.dataid = dataid

    r = data.L1C.create(d)
    test_dataid = 'test_00000_000000000000'
    r.dataid = test_dataid
    r.save()
    r.load(test_dataid)
    # Cleanup
    path = db.filepath(test_dataid, level='L1C')
    os.remove(path)
    r.dataid = dataid

    os.remove(new_recpath)

