Data Model (``doptools.data``)
******************************


.. _data-levels:

Data Levels
===========

The DopTrack data model uses levels which signifies how far a dataset is in processing.
These levels conform with the data levels specified by :abbr:`ESA (European Space Agency)`.

=====  ===========
Level  Explanation
=====  ===========
L0     Recorded signal as complex binary data (``.fc32``) 
L1A    Spectrogram of recorded signal
L1B    Frequency data extracted from spectrogram
L1C    Range-rate data modelled from frequency
L2A    Orbital parameters estimated from range-rate
=====  ===========


Reference/API
=============

.. automodapi:: doptools.data
        :include-all-objects:
        :no-inheritance-diagram:












