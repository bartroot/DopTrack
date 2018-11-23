from pathlib import Path
import yaml


class Config:

    def __init__(self, configpath=None):

        if configpath:
            configpath = Path(configpath)
        else:
            configpath = Path.home() / 'dopconfig.yml'

        with open(configpath) as metafile:
            config = yaml.load(metafile)
        config['paths']['config'] = configpath

        # Get full default path
        config['paths']['default'] = Path(config['paths']['default'])

        # Get remaining full paths
        for path in config['paths']:
            if config['paths'][path] == 'None':
                config['paths'][path] = config['paths']['default'] / path
            elif path == 'recordings':
                recpaths = []
                for _, recpath in config['paths']['recordings'].items():
                    recpaths.append(Path(recpath))
                config['paths']['recordings'] = set(recpaths)
            else:
                config['paths'][path] = Path(config['paths'][path])

        # Add config dictionary to instance dictionary
        self.__dict__.update(config)
