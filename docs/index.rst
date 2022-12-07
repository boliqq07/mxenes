Welcome to mxene's documentation!
=================================

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:

   mxene_tools
   Guide/index
   Examples/index
   Contact/index
   src/modules

This software ``mxene`` is one toolkit for **MXene** material generation,
calculation and analysis.

``mxene`` mainly includes some tools for batch generation of **MXene** structures
and crystal structure feature extraction,
aiming to achieve data mining in universities.
In addition, ``mxene`` can also be used to check the rationality of vasp calculation process and results.

Using this software, you could:

1. Generate initial various kinds **MXene** structure and relevant VASP input.

2. Monitor VASP calculation processing.

3. Organize **MXene** data.

4. Extract structure features from crystal structure.

**Install:**

    pip install mxene

**Install configuration:**

Before first using, we recommend using the following methods
to generate the configuration file.

>>> from mxene.conf_files import write_batch
>>> write_batch()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
