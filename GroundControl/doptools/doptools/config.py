from pathlib import Path
import yaml


class Config:

    def __init__(self, configpath=None):

        if configpath:
            configpath = Path(configpath).resolve()
        else:
            configpath = Path.home() / 'dopconfig.yml'

        with open(configpath) as metafile:
            config = yaml.load(metafile)
        config['paths']['config'] = configpath

        # Get full default path
        defaultpath = Path(config['paths']['default'])
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
