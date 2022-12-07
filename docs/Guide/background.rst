Sample Data and Background
===========================

Sample data
::::::::::::

Download data from the following link: `Structure List <https://github.com/boliqq07/mxene/blob/master/test/mxene_data/structures.pkl_pd>`_ .

**Usage**:

    >>> import pandas as pd
    >>> from pymatgen.core import Structure
    >>> structures = pd.read_pickle("structures.pkl_pd")
    >>> structure = structures[0]

    >>> from mxene.core.mxenes import MXene
    >>> structure = MXene.from_structure(structure)


Background
::::::::::::

The ``Structure`` from ``pymatgen`` is one class to represent the crystal structure data, which contain all message
of atoms and their sites. More details link:
`pymatgen Structure <https://pymatgen.org/usage.html#reading-and-writing-structures-molecules>`_ .

The class ``MXene`` from ``mxene`` is one sub-class ``Structure`` . Therefore, besides the same functions and properties
of ``Structure``, the ``MXene`` has more powerful and specialized function for 2D MXenes materials.