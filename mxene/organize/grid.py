# -*- coding: utf-8 -*-

# @Time  : 2022/10/2 13:20
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License
import itertools
from typing import List, Tuple

import numpy as np
from pymatgen.core import Structure
from sklearn.utils import check_random_state

from mgetool.tool import group_spilt_array
from mxene.core.mxenes import MXene


def group_space(structures: List[Structure]) -> Tuple[List[Structure], List[np.ndarray]]:
    """Group structure by formula."""
    array = np.array([si.composition.reduced_formula for si in structures])
    group_list = group_spilt_array(array)
    sts = np.array(structures, dtype=object)
    structures_group_list = [sts[i] for i in group_list]
    return structures_group_list, group_list


##### Generate ####

# nm_list = ["B", "C", "N", "F", "P", "S", "Cl"]
# tm_list = ["Sc", "Y", "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Mn",
#            "Re", "Fe", "Ru", "Os", "Co", "Rh", "Ir", "Ni", "Pd", "Pt", "Cu", "Ag", "Au", "Zn", "Cd"]
# doping_list = nm_list + tm_list
# cn = ["C", "N"]
# bm = ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"]
# bm_size = {"Ti": 1.47, "Zr": 1.60, "Hf": 1.58, "V": 1.34, "Nb": 1.46, "Ta": 1.46, "Cr": 1.27, "Mo": 1.38, "W": 1.39}
#
# # am_list = ["Al", "Ca", "Li", "Mg", "Na", "K", "Zn", "H"]
# # tem_list = ["O", "F", "OH", "Cl", None]
# # tem_z_axis = {"O": 1.0, "F": 0.9, "Cl": 1.2}
# # am_z_axis = {"K": 2.1, "Li": 1.3, "H": 1.0, "Ca": 1.6, "Na": 1.6, "Mg": 1.4, "Al": 1.1, "Zn": 1.2}
#
# # 成分
# same2 = ((i, i) for i in bm)
# same3 = ((i, i, i) for i in bm)
# same4 = ((i, i, i, i) for i in bm)
#
# diff2 = itertools.combinations(bm, 2)
# diff3 = ((j[0], j[1], j[0]) for j in itertools.permutations(bm, 2))
# diff4 = ((j[0], j[1], j[0], j[1]) for j in itertools.permutations(bm, 2))
#
# its = itertools.chain(same2, same3, same4, diff2, diff3, diff4)
# its = [i for i in its if abs(bm_size[i[0]] - bm_size[i[0]]) < 0.15]
#
# # 晶胞尺寸
# lattice6 = [(i, i, 25, 90, 90, 120) for i in list(np.linspace(2.6, 3.3, 140))]
#
# pure_space = {"super_cell": [(1, 1, 1), ],
#               "base": its,
#               "carbide_nitride": cn,
#               "terminal_site": ["fcc", "hcp"],  # O位置
#               # "lattice": lattice6,  # 尺寸
#               # 'layer_space': list(np.linspace(-0.20, 0.20, 40) + 1.25),  # 层间距
#               # 'terminal_z_axis': list(np.linspace(-0.15, 0.15, 30) + 1.0),  # 层间距
#               }
#
# dop_space = {
#     # "super_cell": [(2, 2, 1), (3, 3, 1)],
#     "super_cell": [ (3, 3, 1)],
#              "base": its,
#              "carbide_nitride": cn,
#              "terminal_site": ["fcc", "hcp"],  # O位置
#              # "lattice": lattice6,  # 尺寸
#              "doping": tm_list + nm_list,
#              # 'layer_space': list(np.linspace(-0.20, 0.20, 40) + 1.25),  # 层间距
#              # 'terminal_z_axis': list(np.linspace(-0.15, 0.15, 30) + 1.0),  # 层间距
#              }
#
# rdm = np.random.RandomState(1)
#

# pg1 = ParameterGrid([pure_space, dop_space])
#
# mxs = [MXene.from_standard(**i, random_state=rdm) for i in tqdm(pg1)]


#### Augment mxene ####

# factor = itertools.product(np.linspace(-0.1, 0.1, 20), np.linspace(-0.01, 0.01, 20))
# center_move = np.linspace(-1, 1, 100)
#
# dop_extrusion_space = {"factors": factor, "center_move": center_move}
# #
# pg2 = ParameterGrid(dop_extrusion_space)
#
# kwarg_extrusion_list = [i for i in pg2]
#
# lattice_strain = {"strain":[[i, i, 0] for i in np.linspace(-0.1, 0.1, 40)]}
#
# pg22 = ParameterGrid(lattice_strain)
#
# kwarg_strain_list = [i for i in pg22]


def random_augment_base_mxene(structures: List[MXene], numbers=512, random_state=None, lr=1.0,
                              extrusion=True, strain=True, tun_layer=True, add_noise=True,
                              kwarg_strain_list=None, kwarg_extrusion_list=None):
    """添加结构变化，批量产生随机数据。用于随机优化。
    Add structural changes to generate random data in batches. Used for stochastic optimization.

    1. 晶胞整体尺寸 adjust_lattice
    2. 层间距  tun_layer
    3. 掺杂原子位置局部变形 extrusion
    4. 随机位移 add_noise
    """
    if extrusion:  # drop this , but use one auto get.
        pass

    if kwarg_extrusion_list is None:
        kwarg_extrusion_list = [{}]
    if kwarg_strain_list is None:
        kwarg_strain_list = [{}]

    rdm = check_random_state(random_state)
    base_number = len(structures)
    dup = numbers // base_number - 1
    left = numbers % base_number

    res = []

    # doping
    for label, mxi in enumerate(structures):

        is_dop_for_extrusion = True if mxi.symbol_set[-1] in mxi._predefined_dop_list \
                                       and mxi.symbol_set[-1] != "O" else False

        for i in range(dup):
            kwarg_extrusion = rdm.choice(kwarg_extrusion_list)
            kwarg_strain = rdm.choice(kwarg_strain_list)
            mxi0 = mxi.relax_base(random_state=rdm, lr=lr,
                                  extrusion=is_dop_for_extrusion, strain=strain, add_noise=add_noise,
                                  tun_layer=tun_layer,
                                  kwarg_extrusion=kwarg_extrusion, kwarg_strain=kwarg_strain)
            mxi0.mark_label = mxi.mark_label
            res.append(mxi0)

    for mxi in res[:left]:
        is_dop_for_extrusion = True if mxi.symbol_set[-1] in mxi._predefined_dop_list \
                                       and mxi.symbol_set[-1] != "O" else False
        kwarg_extrusion = rdm.choice(kwarg_extrusion_list)
        kwarg_strain = rdm.choice(kwarg_strain_list)

        mxi0 = mxi.relax_base(random_state=rdm, lr=lr,
                              extrusion=is_dop_for_extrusion, strain=strain, add_noise=add_noise,
                              tun_layer=tun_layer,
                              kwarg_extrusion=kwarg_extrusion, kwarg_strain=kwarg_strain)
        mxi0.mark_label = mxi.mark_label
        res.append(mxi0)

    res.extend(structures)

    return res

def augment_base_mxene(structure: MXene, lr=1.0, extrusion=True, strain=True, tun_layer=True,
                       add_noise=False, kwarg_strain_list=None, kwarg_extrusion_list=None):
    if extrusion:  # drop this , but use one auto get.
        pass
    if kwarg_extrusion_list is None:
        kwarg_extrusion_list = [{}]
    if kwarg_strain_list is None:
        kwarg_strain_list = [{}]
    resi = []
    mxi = structure

    is_dop_for_extrusion = True if mxi.symbol_set[-1] in mxi._predefined_dop_list \
                                   and mxi.symbol_set[-1] != "O" else False

    for kwarg_extrusion, kwarg_strain in itertools.product(kwarg_extrusion_list, kwarg_strain_list):
        mxi0 = mxi.relax_base(random_state=None, lr=lr, extrusion=is_dop_for_extrusion, strain=strain,
                              add_noise=add_noise, tun_layer=tun_layer, kwarg_extrusion=kwarg_extrusion,
                              kwarg_strain=kwarg_strain)
        mxi0.mark_label = mxi.mark_label
        resi.append(mxi0)
    return resi
def exhaustion_augment_base_mxene(structures: List[MXene], lr=1.0,
                                  extrusion=True, strain=True, tun_layer=True, add_noise=False,
                                  kwarg_strain_list=None, kwarg_extrusion_list=None):
    """仅用于测试！！！ 添加结构变化，批量产生随机数据。用于穷举测试。
    For testing only!! Add structural changes to generate random data in batches. Used for exhaustive testing.

    1. 晶胞整体尺寸 adjust_lattice
    2. 层间距  tun_layer
    3. 掺杂原子位置局部变形 extrusion
    4. 随机位移 add_noise
    """

    res = []

    for label, mxi in enumerate(structures):
        resi = augment_base_mxene(mxi, lr =lr, extrusion = extrusion, strain = strain, tun_layer = tun_layer,
                           add_noise = add_noise,kwarg_strain_list = kwarg_strain_list,
                           kwarg_extrusion_list = kwarg_extrusion_list)

        res.extend(resi)

    return res


#### Augment absorb mxene ####

# factor = itertools.product(np.linspace(-0.1, 0.1, 20), np.linspace(-0.01, 0.01, 20))
# center_move = np.linspace(-0.2, 0.2, 100) # just for H
#
# dop_extrusion_space = {"factors": factor, "center_move": center_move}
# #
# pg2 = ParameterGrid(dop_extrusion_space)
#
# kwarg_extrusion_list = [i for i in pg2]


##### Add H  ########

# absorb_space = {'site_name': ['S0', 'S1', 'S2'],
#                    'site_type': ["top", "center"],
#                    'offset_z': np.linspace(-0.1, 0.1, 20)}
#
# pg3 = ParameterGrid(absorb_space)
#
# kwarg_absorb_list = [i for i in pg3]


def random_add_absorb_H_batch(structures: List[MXene], random_state=None,
                              kwarg_absorb_list=None):
    """Add H the number is changed."""
    rdm = check_random_state(random_state)

    if kwarg_absorb_list is None:
        kwarg_absorb_list = [{}]

    res = []
    for mxi in structures:

        mxi0 = mxi.copy()

        kwarg_absorb = rdm.choice(kwarg_absorb_list)

        nm = True if mxi0.symbol_set[-1] in mxi0._predefined_nm_list else False

        doped = mxi0.doped

        if not nm:
            kwarg_absorb['site_type'] = "top"

        if not doped:
            kwarg_absorb['site_type'] = "center"
            kwarg_absorb['pure'] = True
            kwarg_absorb['center'] = None

        mxi0 = mxi0.add_absorb(add_noise=False, up_down="up",
                               equivalent="fin_opt", absorb="H",
                               ignore_index=-1, tol=0.2, **kwarg_absorb)

        mxi0.mark_label = mxi.mark_label

        res.append(mxi0)

    return res


def add_absorb_H_batch(structure: MXene, kwarg_absorb_list=None):
    """Testing!!!  Add H the number is changed."""

    if kwarg_absorb_list is None:
        kwarg_absorb_list = [{}]

    mxi = structure

    resi = []
    for kwarg_absorb in kwarg_absorb_list:
        mxi0 = mxi.copy()
        nm = True if mxi0.symbol_set[-1] in mxi0._predefined_nm_list else False
        doped = mxi0.doped
        if not nm:
            kwarg_absorb['site_type'] = "top"

        if not doped:
            kwarg_absorb['site_type'] = "center"
            kwarg_absorb['pure'] = True
            kwarg_absorb['center'] = None

        mxi0 = mxi0.add_absorb(add_noise=False, up_down="up",
                               equivalent="fin_opt", absorb="H",
                               ignore_index=-1, tol=0.2, **kwarg_absorb)

        mxi0.mark_label = mxi.mark_label

        resi.append(mxi0)

    return resi

def exhaustion_add_absorb_H_batch(structures: List[MXene],
                                  kwarg_absorb_list=None):
    """Testing!!!  Add H the number is changed."""
    res = []
    for mxi in structures:
        resi = add_absorb_H_batch(mxi, kwarg_absorb_list=kwarg_absorb_list)
        res.append(resi)
    return res
