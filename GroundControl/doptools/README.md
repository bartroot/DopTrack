DopTools - DopTrack Toolbox
===========================


Setup
-----

The doptools package is "hidden" inside the DopTrack repository. 
The doptools package can technically be used by setting the current working directory to the package directory, but it is easier to just add the package directory to the `PYTHONPATH` variable.
This can be done by adding a path (`.pth`) file with any name, for example `user_paths.pth`, in the `PythonXX/Lib/site-packages` folder. Then simply add a line with the doptools path in the file:
```
C:\Users\john\DopTrack\GroundControl\doptools
```

A config file is required to run doptools. This config file should be placed in the home folder of the user and named `dopconfig.yml`
An example of a config file, `dopconfig.yml.example`, can be found in the doptools folder.

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
In this case the paths are given as WindowsPath objects from the `pathlib` module, but the code also works on unix where they will automatically be given as PosixPath objects.

Given a data id and a data level the database object also gives access to the path of the specific file, if it exists. 
For example, the L0 (recording) data consists of both a data file and an associated meta file. The `filepath` method returns the path of the data file by default, but can also return the path of the meta file as well. 
The following method call returns the path of the L0 (recording) meta file for the specific dataid.
```python
>>> db.filepath('Delfi-C3_32789_201608062054', level='L0', meta=True)
WindowsPath('C:/Users/john/data/L0/Delfi-C3_32789_201608062054.yml')
```

Sets of data id's for each level can be found in the dataids dictionary.
```python
>>> db.dataids['L1B']
{'Delfi-C3_32789_201706092221',
 'Delfi-C3_32789_201607151229',
 'Delfi-C3_32789_201602210946'}
```
The set of all data ids in the databse, i.e. the union of all levels, is also in the dictionary under the key `all`.

Since these collections of data ids are returned as sets, the desired group of data ids can often be found simply by using standard set operations: `+`, `-`, `union`, and `intersection`.
For example, to find all data ids of which there is recording data (L0), but which have not yet been processed to frequency data points (L1B) use:
```python
L0_dataids = db.dataids['L0']
L1B_dataids = db.dataids['L1B']

dataids_to_process = L0_dataids - L1A_dataids
```
Or to find all data ids that has either been processed to L1B or L2 (in most cases L2 will be a subset of L1B, but lets assume it is not):
```python
L1B_dataids = db.dataids['L1B']
L2_dataids = db.dataids['L2']

new_dataids = set.union(L0_dataids, L1A_dataids)
```








### Level 1A (spectrograms)

The model_doptrack module gives access to a spectrogram class.

```python
from doptools.model_doptrack import Spectrogram
```

To generate a spectrogram of a specific satellite pass simply pass the data id to the class constructor.

```python
spec = Spectrogram.create('Delfi-C3_32789_201602210946', nfft=250000, dt=0.2)
```

The spectrogram can be saved using the corresponding instance method. A saved spectrogram can be loaded by calling the `load` class constructor.

```python
spec.save()
loaded_spec = Spectrogram.load('Delfi-C3_32789_201602210946')
```

Loading an already saved spectrogram is much faster than creating one from scratch since no calculations have to be done.
The saved spectrograms can however take up a lot of harddisk space if `nfft` is large and `dt` is small.







### Level 1B (frequency datapoints)

The frequency data points can be extracted from a spectrogram by using the `ExtractedData` class.

```python
from doptools.extraction import ExtractedData

extracted_data = ExtractedDataPoints('Delfi-C3_32789_201602210946')
```

This will extract the data points during object instantiation, after which the data can be used directly, for example plotted, or saved to the database. This data can then be loaded using the class constructor.
```python
extracted_data.save()
loaded_extracted_data = ExtractedData.load('Delfi-C3_32789_201602210946')
```

In addition to the time, frequency, and power data, the `ExtractedData` object also contains time of closest approach, frequency at closest approach, and also the fitting function used during extraction.
```python
extracted_data.fit_func
```
All these are available both after instantiation, i.e. running the extracting algorithm, as well as after loading already saved data.

The data can also be quickly plotted for inspection.
```python
extracted_data.plot()
```

![Extracted data](/docs/images/extracted.png "Test")





### Level 2









Data Analysis
-------------















































