from pathlib import Path
import yaml
import logging


__all__ = ['Config']


logger = logging.getLogger(__name__)


class Config:

    def __init__(self, configpath=None):

        if configpath:
            configpath = Path(configpath).resolve()
        else:
            configpath = Path.home() / 'dopconfig.yml'
            if not configpath.is_file():
                logger.warning('Could not find config file in HOME folder. Using default config.')
                configpath = Path(__file__).parent / '../dopconfig.example.yml'

        with open(configpath) as metafile:
            config = yaml.load(metafile)
        config['paths']['config'] = configpath

        # Get full default path
        if config['paths']['default'] is None:
            defaultpath = Path.home() / 'data'

        defaultpath = Path(defaultpath)
        if not defaultpath.is_absolute():
            defaultpath = configpath.parent / defaultpath
        config['paths']['default'] = defaultpath

        # Get remaining full paths
        for key, path in config['paths'].items():
            if key == 'L0':
                if isinstance(path,  dict):
                    config['paths'][key] = {Path(subpath) for subpath in path.values()}
                elif path is None:
                    config['paths'][key] = {config['paths']['default'] / key}
                else:
                    config['paths'][key] = {Path(path)}
            elif path is None:
                config['paths'][key] = config['paths']['default'] / key
            else:
                config['paths'][key] = Path(path)

        # Add config dictionary to instance dictionary
        self.__dict__.update(config)
