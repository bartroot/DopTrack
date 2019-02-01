import pytest
import logging
from pathlib import Path

from doptrack.config import Config

logger = logging.getLogger(__name__)


def test_config_can_find_file():
    Config(configpath=Path(__file__).parent / 'test_dopconfig.yml')


def test_config_cannot_find_file():
    with pytest.raises(FileNotFoundError, message='Expecting FileNotFoundError'):
        Config(configpath=Path(__file__).parent / 'test_dopconfig999.yml')


def test_config_contains_correct_keys():
    config = Config(configpath=Path(__file__).parent / 'test_dopconfig.yml')
    assert hasattr(config, 'paths')
    assert hasattr(config, 'station')
    assert hasattr(config, 'space-track.org')
    assert hasattr(config, 'runtime')


def test_config_contains_correct_subkeys():
    config = Config(configpath=Path(__file__).parent / 'test_dopconfig.yml')
    assert 'default' in config.paths.keys()
    assert 'L0' in config.paths.keys()
    assert 'L1A' in config.paths.keys()
    assert 'L1B' in config.paths.keys()
    assert 'L2' in config.paths.keys()
    assert 'output' in config.paths.keys()
    assert 'external' in config.paths.keys()
    assert 'logs' in config.paths.keys()


def test_config_has_correct_value():
    config = Config(configpath=Path(__file__).parent / 'test_dopconfig.yml')
    assert issubclass(type(config.paths['default']), Path)
    assert type(config.paths['L0']) == set
    assert issubclass(type(config.paths['L1A']), Path)
    assert config.station['latitude'] == 51.9989
    assert type(config.station['latitude']) is float
    assert config.runtime['logging'] is False
