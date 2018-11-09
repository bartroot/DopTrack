DopTools - DopTrack Toolbox
===========================


Setup
-----

A config file is required to run doptools. This config file should be placed in the home folder of the user and should be named `dopconfig.yml`
An example of a config file `dopconfig.yml.example` can be found in the doptools package folder.

The most important section of the config file is the `paths` section. This section specifies paths to the data in the database. 
If a path, e.g. L1A, is set to `None` then the path will be set to `[default]/[L1A]`. 

```yaml
paths:
    default: .home/data
    L0: None
    L1A: None
    L1B: None
    L2: None
    external: None
```

The L0 data (recordings) can be spread out in multiple folders.
In this case the paths are added as follows:

```yaml
paths:
    default: ../../../data
    L0: 
	    path1: /home/recordingsA
		path2: /home/recordingsB
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

A database object is available in the io module for easy access to and inspection of to the database.
```python
from doptools.io import Database

db = Database()
```

This database object is a snapshot of the database taken at instantiation of the class. 
If files in the database changes after instantiation, then the database has to be instantiated anew in order to update.

The database gives easy access to all the paths defined in the config file.
```python
>> db.paths
{'default': WindowsPath('C:/Users/john/data'),
 'L0': WindowsPath('C:/Users/john/data/L0'),
 'L1A': WindowsPath('C:/Users/john/data/L1A'),
 'L1B': WindowsPath('C:/Users/john/data/L1B'),
 'L2': WindowsPath('C:/Users/john/data/L2'),
 'external': WindowsPath('C:/john/data/external'),
 'config': WindowsPath('C:/Users/john/dopconfig.yml')}
```

Given a dataid and a data level it also gives access to the path to the specific file, if it exists. 
For example, the following returns the path of the L0 (recording) meta file for a specific dataid.
```python
>> db.filepath('Delfi-C3_32789_201608062054', level='L0', meta=True)
WindowsPath('C:/Users/john/data/L0/Delfi-C3_32789_201608062054.yml')
```

All dataids for a specific level can be found in the dataids dictionary.
```python
>> db.dataids['L1B']
{'Delfi-C3_32789_201706092221',
 'Delfi-C3_32789_201607151229',
 'Delfi-C3_32789_201602210946'}
```


### Level 1A (spectrograms)


### Level 1B (frequency datapoints)


### Level 2


Data Analysis
-------------















































