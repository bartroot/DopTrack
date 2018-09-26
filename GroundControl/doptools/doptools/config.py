import os
import configparser


# TODO Config path should be changed to $HOME.
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.ini'))
CONFIG_PATH = os.path.normpath(CONFIG_PATH)
config = configparser.ConfigParser()
config.read(CONFIG_PATH)


if os.path.isabs(config['data']['dir']):
    DATA_PATH = os.path.join(config['data']['dir'])
else:
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', config['data']['dir'])
DATA_PATH = os.path.normpath(DATA_PATH)

config['path'] = {'data': DATA_PATH,
                  'recordings': os.path.join(DATA_PATH, 'recordings'),
                  'spectrograms': os.path.join(DATA_PATH, 'spectrograms'),
                  'rre': os.path.join(DATA_PATH, 'rre'),
                  'external': os.path.join(DATA_PATH, 'external')}
