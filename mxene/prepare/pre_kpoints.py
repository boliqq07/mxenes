# -*- coding: utf-8 -*-
# @Time  : 2022/11/4 16:52
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

from pymatgen.io.vasp import Kpoints

kpoints331 = Kpoints(kpts=((3, 3, 1),))

kpoints221 = Kpoints(kpts=((5, 5, 1),))

kpoints111 = Kpoints(kpts=((11, 11, 1),))

kp_dict ={
    (3, 3, 1): kpoints331,
    (2, 2, 1): kpoints221,
    (1, 1, 1): kpoints111,
}