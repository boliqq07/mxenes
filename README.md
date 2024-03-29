# mxene
[![Python Versions](https://img.shields.io/pypi/pyversions/mxene.svg)](https://pypi.org/project/mxene/)
[![Version](https://img.shields.io/github/tag/boliqq07/mxene.svg)](https://github.com/boliqq07/releases/latest)
![pypi Versions](https://badge.fury.io/py/mxene.svg)
[![Documentation Status](https://readthedocs.org/projects/mxene/badge/?version=latest)](https://mxene.readthedocs.io/en/latest/?badge=latest)

mxene is one software to solving parallel problems for VASP calculation.

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

```
>>> from mxene.conf_files import write_batch
>>> write_batch()
```

Support
----------------------
[![Jetbrains](jetbrains.svg)](https://jb.gg/OpenSource)
