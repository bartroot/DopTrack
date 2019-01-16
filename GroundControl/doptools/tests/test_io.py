import pytest
import logging
from pathlib import Path

import doptools.io as io
from doptools.config import Config

logger = logging.getLogger(__name__)
config = Config(configpath=Path(__file__).parent / 'test_dopconfig.yml')


class TestDatabase:

    @pytest.mark.parametrize('level', ['L0', 'L1A', 'L1B'])
    def test_filepath_when_data_is_available(self, level):
        db = io.Database(config=config)
        db.filepath('Delfi-C3_32789_201602210946', level=level)

    @pytest.mark.parametrize('level', ['L0', 'L1A'])
    def test_filepath_when_metadata_is_available(self, level):
        db = io.Database(config=config)
        db.filepath('Delfi-C3_32789_201602210946', level=level, meta=True)

    @pytest.mark.parametrize('level', ['L0', 'L1A', 'L1B'])
    def test_filepath_when_data_is_unavailable(self, level):
        with pytest.raises(FileNotFoundError, message='Expecting FileNotFoundError'):
            db = io.Database(config=config)
            db.filepath('NOAA-19_33591_201604151259', level=level)

    def test_filepath_when_metadata_is_unavailable(self):
        with pytest.raises(FileNotFoundError, message='Expecting FileNotFoundError'):
            db = io.Database(config=config)
            db.filepath('NOAA-19_33591_201604151259', level='L0', meta=True)

    def test_filepath_when_metadata_does_not_exist_for_given_level(self):
        with pytest.raises(RuntimeError, message='Expecting RuntimeError'):
            db = io.Database(config=config)
            db.filepath('Delfi-C3_32789_201602210946', level='L1B', meta=True)

    @pytest.mark.parametrize('level', [True, False, None, 999, 'bad_string'])
    def test_filepath_when_given_incorrect_level(self, level):
        with pytest.raises(KeyError, message='Expecting KeyError'):
            db = io.Database(config=config)
            db.filepath('Delfi-C3_32789_201602210946', level=level)


class TestDataID:

    @pytest.mark.parametrize('dataid', ['NOAA-19_33591_201604151259',
                                        'Delfi-C3_32789_201602210946',
                                        'Tianwang-1A_40928_201603130831'])
    def test_valid_dataids(self, dataid):
        io.DataID(dataid)

    @pytest.mark.parametrize('dataid', ['NOAA-19_999_201604151259',
                                        'Delfi-C3_99999999_2016022109460'])
    def test_invalid_dataid_with_incorrect_length_satnum(self, dataid):
        with pytest.raises(TypeError, message='Expecting TypeError'):
            io.DataID(dataid)

    @pytest.mark.parametrize('dataid', ['NOAA-19_33591_2016041',
                                        'Delfi-C3_32789_201602210946999999'])
    def test_invalid_dataid_with_incorrect_length_strtimestamp(self, dataid):
        with pytest.raises(TypeError, message='Expecting TypeError'):
            io.DataID(dataid)

    @pytest.mark.parametrize('dataid', ['NOAA-19_33591',
                                        'Delfi-C3_32789_201602210946_999999'])
    def test_invalid_dataid_with_incorrect_number_of_parts(self, dataid):
        with pytest.raises(TypeError, message='Expecting TypeError'):
            io.DataID(dataid)

    def test_if_correct_satname_is_returned(self):
        dataid = io.DataID('NOAA-19_33591_201604151259')
        assert dataid.satname == 'NOAA-19'

    def test_if_correct_satnum_is_returned(self):
        dataid = io.DataID('NOAA-19_33591_201604151259')
        assert dataid.satnum == '33591'

    def test_if_correct_strtimestamp_is_returned(self):
        dataid = io.DataID('NOAA-19_33591_201604151259')
        assert dataid.strtimestamp == '201604151259'
