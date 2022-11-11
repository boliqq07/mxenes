# -*- coding: utf-8 -*-
import functools
# @Time  : 2022/11/4 16:52
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

"""
############## Usage 1 ###################

sym_potcar_map = check_potcar(potpath=r"POT-database")
potcar = get_potcar(poscar, sym_potcar_map=sym_potcar_map)

############## Usage 2 (Avoid repeated calls) ###################

sym_potcar_map = check_potcar(potpath=r"POT-database")

@functools.lru_cache(200)
def get_potcar_lru(sym):
    return Potcar(sym, sym_potcar_map=sym_potcar_map)

potcar = get_potcar_lru(tuple(poscar.site_symbols))

"""

import itertools
import os
from typing import Union
from pymatgen.core import Structure
from pymatgen.io.vasp import Potcar, Poscar


def check_potcar(potpath):
    """
    Check and get the potcar file.
    The potcar file name should start from element such as 'Ag-POSCAR','Ag','Ag_sl'.
    """
    try:
        potpaths = os.listdir(potpath)
        sym_potcar_map = {}
        for i in potpaths:
            with open(potpath + "/" + i) as f:
                te = f.readlines()
                text = "".join(te)
                i = i.split("-")[0]
                i = i.split("_")[0]
                sym_potcar_map.update({i: text})
    except BaseException:
        sym_potcar_map = None
    return sym_potcar_map

# potpath = r"POT-database"  # POT-database should be offered
# sym_potcar_map = check_potcar(potpath=r"POT-database")

# @functools.lru_cache(120)
# def get_potcar_lru(sym):
#     return Potcar(sym, sym_potcar_map=sym_potcar_map)


def get_potcar(poscar: Union[Poscar, Structure], sym_potcar_map: dict):
    """
    Read from sym_potcar_map according poscar message.
    """
    if isinstance(poscar, Structure):
        syms = [site.specie.symbol for site in poscar]
        site_symbols = [a[0] for a in itertools.groupby(syms)]
        sym = tuple(site_symbols)
    else:
        sym = tuple(poscar.site_symbols)

    if sym_potcar_map is not None:
        POTCAR = Potcar(sym, sym_potcar_map=sym_potcar_map)
    else:
        POTCAR = Potcar(sym)
    return POTCAR