# -*- coding: utf-8 -*-

# @Time  : 2022/11/3 21:31
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

from typing import List, Union, Tuple, Sequence
import numpy as np
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import DummySpecies, Element, Species

VectorFloatLike = Union[Sequence[float], np.ndarray]
VectorIntLike = Union[Sequence[int], np.ndarray]
VectorLike = Union[Sequence, np.ndarray]

ListTuple = Union[List, Tuple]

ArrayLike = Union[Sequence[float], Sequence[Sequence[float]], Sequence[np.ndarray], np.ndarray]

SpeciesLike = Union[str, Element, Species, DummySpecies]

CompositionLike = Union[str, Element, Species, DummySpecies, dict, Composition]