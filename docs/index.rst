Welcome to mxene's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   mxene_tools
   Guide/index
   Examples/index
   Contact/index
   src/modules

mxene includes some tools for batch generation of mxene structures and extraction of crystal structure features, which
can be used as the input of VASP for subsequent calculation, and can also analyze and extract some calculation results
to achieve data mining in universities. In addition, mxene can also be used to check the rationality of vasp calculation
process and results.

This software is optimized for 'MXene' material generation,
calculation and analysis to achieve efficient data mining.

Using this software, you could:

1. Generate initial various kinds mxene structure and relevant VASP input.

2. Monitor VASP calculation processing.

3. Organize mxene data.

4. Extract structure features from crystal structure.

# Install

```bash
pip install mxene
```

# Initial configuration

Before first using, we recommend using the following methods
to generate the configuration file.

>>> from mxene.conf_files import write_batch
>>> write_batch()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
