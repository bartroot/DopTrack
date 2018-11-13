DopTools - DopTrack Toolbox
===========================


Setup
-----

A config file is required to run doptools. This config file should be placed in the home folder of the user and named `dopconfig.yml`
An example of a config file `dopconfig.yml.example` can be found in the doptools folder.

The most important section of the config file is the `paths` section. This section specifies paths to the data in the database. 
The names of the paths corespond to the data levels (see further down). 

```yaml
paths:
    default: C:/Users/john/data
    L0: None
    L1A: None
    L1B: None
    L2: None
    external: None
```

If a path, e.g. L1A, is set to `None` then the path will be set to `<default>/L1A`.
The L0 data (recordings) can be spread out in multiple folders.
In this case the paths are added as follows:

```yaml
paths:
    default: C:/Users/john/data
    L0: 
        path1: C:/Users/john/recordingsA
        path2: D:/recordingsB
    L1A: None
    L1B: None
    L2: None
    external: None
```

The keys of the paths (here `path1` and `path2`) can be anything as long as they are different.



Data Processing
---------------

### Data levels

The data is organised into levels corresponding to the levels used by ESA.

* **Level 0**: Raw data. Recordings stored as a data file and a meta file.
* **Level 1A**: Processed data. Spectrograms of recordings  stored as a data file and a meta file.
* **Level 1B**: Processed data. Frequency data points extracted from spectrograms. 
* **Level 2**: Modeled data. Range rate data etc. from frequency data points.

### Database

A database object is available in the io module for easy access and inspection of the database.
```python
from doptools.io import Database

db = Database()
```

This database object is a snapshot of the database taken at instantiation of the class. 
If files in the database changes after instantiation, then the database has to be instantiated anew in order to update.

The database gives easy access to all the paths defined in the config file.
```python
>>> db.paths
{'default': WindowsPath('C:/Users/john/data'),
 'L0': WindowsPath('C:/Users/john/data/L0'),
 'L1A': WindowsPath('C:/Users/john/data/L1A'),
 'L1B': WindowsPath('C:/Users/john/data/L1B'),
 'L2': WindowsPath('C:/Users/john/data/L2'),
 'external': WindowsPath('C:/john/data/external'),
 'config': WindowsPath('C:/Users/john/dopconfig.yml')}
```
In this case the paths are given as WindowsPath objects from the pathlib module, but the code also works on unix where they will automatically be given as PosixPath objects.

Given a data id and a data level the database object also gives access to the path of the specific file, if it exists. 
For example, the following returns the path of the L0 (recording) meta file for a specific dataid.
```python
>>> db.filepath('Delfi-C3_32789_201608062054', level='L0', meta=True)
WindowsPath('C:/Users/john/data/L0/Delfi-C3_32789_201608062054.yml')
```

All data id's for a specific level can be found in the dataids dictionary.
```python
>>> db.dataids['L1B']
{'Delfi-C3_32789_201706092221',
 'Delfi-C3_32789_201607151229',
 'Delfi-C3_32789_201602210946'}
```


### Level 1A (spectrograms)

The model_doptrack module gives access to a spectrogram class.

```python
from doptools.model_doptrack import Spectrogram
```

To generate a spectrogram of a specific satellite pass simply pass the data id to the class constructor.

```python
s = Spectrogram.create('Delfi-C3_32789_201602210946', nfft=250000, dt=0.2)
```

The spectrogram can be plotted, saved, and loaded using the coresponding class methods.

```python
s.plot()
s.save()
s_loaded = Spectrogram.load('Delfi-C3_32789_201602210946')
```

Loading an already saved spectrogram is much faster than creating one from scratch since no calculations have to be done.
The saved spectrograms can however take up a lot of harddisk space if `nfft` is large and `dt` is small.


### Level 1B (frequency datapoints)


### Level 2


Data Analysis
-------------















































