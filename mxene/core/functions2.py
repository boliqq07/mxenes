# -*- coding: utf-8 -*-

# @Time  : 2022/10/2 13:20
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

from typing import Union

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms, FixedPlane
from ase.constraints import FixedLine


def get_fixed_atoms(atoms: Atoms, fixed_type: Union[str, float, None] = "base", coords_are_cartesian=True):
    if fixed_type is None:
        pass
    else:
        if not coords_are_cartesian:
            coords = atoms.cell.scaled_positions(atoms.positions)
        else:
            coords = atoms.positions
        fixed_array = np.full_like(coords, False).astype(bool)
        if fixed_type == "base":
            fixed_array[-1] = True
        else:
            if isinstance(fixed_type, float):
                index = coords[:, -1] > fixed_type
                fixed_array[index] = True
        return fixed_array


def fixed_atoms(atoms: Atoms, fixed_type: Union[str, float, None] = "base", doping_fixed_type="line",
                doping_direction=(0, 0, 1), coords_are_cartesian=True):
    fixed_array = get_fixed_atoms(atoms, fixed_type=fixed_type, coords_are_cartesian=coords_are_cartesian)
    fixed_array = ~fixed_array[:, -1]
    fixed_array[-1] = False
    FixAtoms(mask=fixed_array)

    if doping_fixed_type == "line":
        FixedLine(len(atoms) - 1,
                         direction=doping_direction, )

    elif doping_fixed_type == "plane":
        FixedPlane(len(atoms) - 1,
                          direction=doping_direction, )
    else:
        raise TypeError
    atoms.set_constraint()
    return atoms
