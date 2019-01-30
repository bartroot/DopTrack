Setup
*****

.. warning::
        The doptools package requires Python 3.6+.


Installing doptools
===================


First we have to install the doptools package.
The doptools package is located inside the DopTrack github repository, so we start by downloading this repository.
We assume that we only want to use doptools  and not develop on it. If you want to develop on doptools, see :doc:`../dev-guide`.

We start by cloning the repository::

        $ git clone https://github.com/DopTrack/DopTrack


Add doptools to the python path
-------------------------------

Now that we have the repository, we want to make doptools usable in python by adding it to the ``PYTHONPATH`` variable.
The simplest way to do this is by adding a path file (``.pth``) with any name, for example ``user_paths.pth``, in the ``PythonXX/Lib/site-packages`` folder.
You then add the path to doptools in the file::

        C:\Users\john\DopTrack\GroundControl\doptools

You can add any number of folders to ``PYTHONPATH`` by simply adding a new line in this file with the path to the folder.


Installing dependencies
-----------------------

Finally, we have to install the dependicies of doptools by using ``pip``. In a terminal, move to the doptools folder and run the following command::

        $ python3 -m pip install -r requirements.txt


Now you should be able to run python and import ``doptools``::

        >> import doptools
        >> doptools
        <module 'doptools' from 'C:\\Users\\john\\DopTrack\\GroundControl\\doptools\\doptools\\__init__.py'>


Setting up the configuration
============================

The configuration determines how doptools will run on your system and is specified in a config file.
This config file should be placed in the home folder of the user and named ``dopconfig.yml``
An example of a config file, ``dopconfig.example.yml``, can be found in the doptools folder.

.. code-block:: yaml
        :linenos:

        paths:
            default: <absolute path>
            L0: null
            L1A: null
            L1B: null
            L1C: null
            output: null
            external: null
            logs: null
            analysis: null

        station:
            latitude: 51.9989
            longitude: 4.3733585
            altitude: 95

        space-track.org:
            user: <USERNAME>
            password: <PASSWORD>

        runtime:
            logging: false


Setting up data paths
---------------------

The most important section of the config file is the ``paths`` section. This section specifies paths to the data in the database.
The keys of the paths correspond either to a data level, such as ``L1B`` (see :ref:`data-levels`), or to a general type of data, such as ``analysis``.
When saving or loading data these folders are where doptools will look for the data files.

.. note::
        Any path, except ``default``, which is set to ``null`` will be assigned a default path of ``<default>/<key>``.
        For example, if ``default`` is set to ``C:/Users/john/data`` and ``L0`` is set to ``null`` in the config file, then doptools will assume that the actual L0 folder is at ``C:/Users/john/data/L0``.


Using multiple folders for raw data
-----------------------------------

The L0 data (raw recorded data) can take up a lot of space so it might be spread out over different disks.
Because of this the L0 path can also be a list of paths, which are added as follows:

.. code-block:: yaml
        :linenos:
        :emphasize-lines: 3,4,5

        paths:
            default: C:/Users/john/data
            L0:
                path1: C:/Users/john/recordingsA
                path2: D:/recordingsB
            L1A: null
            L1B: null
            L1C: null
            output: null
            external: null
            logs: null
            analysis: null

The doptools package will then be able to find raw data files in either of these two folders.
In this case all the remaining data folders are null and are therefore set to ``<default>/<key>``.

.. note::
        The keys of the L0 paths (here ``path1`` and ``path2``) can be anything as long as they are different.

For further explanation of the configuration see :py:mod:`doptools.config`


Setting up the database
=======================

Finally, the database has to be set up with the correct folder structure.
This structure should of course follow the paths given in the config file.
The folders can either be set up manually, or, if the config file is set up correctly, automatically by using the ``Database`` object::

        >> from doptools.io import Database
        >> db = Database()
        >> db.setup()
        INFO:doptools.io:Directory already exists: /home/john/data
        INFO:doptools.io:Directory already exists: /home/john/data/L0
        INFO:doptools.io:Created directory: /home/john/data/L1A
        INFO:doptools.io:Created directory: /home/john/data/L1B
        INFO:doptools.io:Created directory: /home/john/data/L1C
        INFO:doptools.io:Created directory: /home/john/data/output
        INFO:doptools.io:Created directory: /home/john/data/external
        INFO:doptools.io:Created directory: /home/john/data/logs
        INFO:doptools.io:Created directory: /home/john/data/analysis
        INFO:doptools.io:Created directory: /home/john/data/output/L1B
        INFO:doptools.io:Created directory: /home/john/data/output/L1B_failed
        INFO:doptools.io:Created directory: /home/john/data/external/eopp
        INFO:doptools.io:Created directory: /home/john/data/analysis/passes

Here we already had an L0 data folder, maybe with some already recorded data, and we were notified of it.
For more information about the ``Database`` object see :py:class:`doptools.io.Database`.


