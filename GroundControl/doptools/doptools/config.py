import os
import yaml


# TODO Config path should be changed to $HOME.
BASE_PATH = os.path.join(os.path.dirname(__file__), '..')
BASE_PATH = os.path.normpath(BASE_PATH)
CONFIG_PATH = os.path.abspath(os.path.join(BASE_PATH, 'config.yml'))
CONFIG_PATH = os.path.normpath(CONFIG_PATH)


class Config:

    def __init__(self, configpath=CONFIG_PATH):

        self.configpath = configpath
        with open(self.configpath) as metafile:
            config = yaml.load(metafile)

        # Get full default path
        config['paths']['default'] = _normpath(config['paths']['default'], BASE_PATH)

        # Get remaining full paths
        for path in config['paths']:
            if config['paths'][path] == 'None':
                config['paths'][path] = os.path.join(config['paths']['default'], path)
            elif path == 'recordings':
                recpaths = []
                for _, recpath in config['paths']['recordings'].items():
                    recpaths.append(_normpath(recpath, config['paths']['default']))
                config['paths']['recordings'] = set(recpaths)
            else:
                if not os.path.isabs(config['paths'][path]):
                    config['paths'][path] = _normpath(config['paths'][path],
                                                      config['paths']['default'])

        # Add config dictionary to instance dictionary
        self.__dict__.update(config)


def _normpath(path, defaultpath):
    if not os.path.isabs(path):
        path = os.path.join(defaultpath, path)
    return os.path.normpath(path)
