# -*- coding: utf-8 -*-

# @Time  : 2022/10/2 13:20
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

"""This is some base support function for mxene.py."""
import functools
import itertools
import numbers
import pathlib
import warnings
from collections import Counter
from typing import List, Union, Dict, Tuple

import numpy as np
from numpy import atleast_1d
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from mgetool.cluster import coarse_cluster_array


def middle(st: Structure, ignore_index=None, tol=0.01):
    """Calculate the core layer - middle layer."""
    frac = st.frac_coords[:, -1]
    if ignore_index:
        mark = np.full_like(frac, True)
        mark[ignore_index] = False
        frac_mark = frac[ignore_index]
    else:
        frac_mark = frac
    mid = np.mean(frac_mark)
    middle_atoms = [i for i, z in enumerate(frac) if abs(z - mid) < tol]
    return middle_atoms


def _get_err_decrease_and_increase(label):
    """返回离群值"""
    label_dict = Counter(label)
    common_num = Counter(list(label_dict.values())).most_common(1)[0][0]

    assert common_num >= 3, "Just for super_cell"

    err = []
    decrease = []
    increase = []

    for k in label_dict.keys():
        if label_dict[k] <= 1:
            err.append(k)
        if label_dict[k]==common_num-1:
            decrease.append(k)
        if label_dict[k] == common_num+1:
            increase.append(k)

    if len(err) > 1 or len(decrease)>1 or len(increase) > 1:
        raise ValueError("Just for single doping, if absorbed, "
                         "please use ignore_index to jump the absorb atoms. "
                         "Or the tolerance is too small.")

    err = err[0] if len(err)==1 else None
    increase = increase[0] if len(increase) == 1 else None
    decrease = decrease[0] if len(decrease) == 1 else None

    return err,decrease,increase


def coarse_and_spilt_array_ignore_force_plane(array: np.ndarray, ignore_index: Union[int, np.ndarray] = None,
                                              tol=0.5, force_plane: bool = True, reverse: bool = True,
                                              force_finite: bool = True, method=None, n_cluster: int = 3) -> np.ndarray:
    """
    Split 1D by distance or group.

    'ignore_index' means the atoms is not used to calculated.
    'force_finite' and 'force_plane' are two method to settle the un-grouped atoms.

    (1). For easily compartmentalized array, use the default parameters.

    (2). Used 'ignore_index' and 'force_finite' to drop the add (absorb) atom to make the left atoms to be grouped,
         change the 'tol' to check the result.
    (3). Used 'ignore_index'=None and 'force_plane' to make the 'out-of-station' doped atom to be grouped.
         the 'tol' could be appropriate smaller (less than interlayer spacing).
    (4). For absorb + doped system, 'ignore_index', 'force_finite' and 'force_plane' could be used together,
         change the 'tol' to check the result. But, for large structural deformations array.
         This function is not effective always.
    (5). If all the parameter are failed, please generate the input array by hand or
         tune the value in array.

    Examples:
    >>> coarse_and_spilt_array_ignore_force_plane(np.array([0.1,0.11,0.12,0.15,0.16,0.17]),
    ...                                                 ignore_index= None, tol=0.02,force_plane=True,
    ...                                                 reverse=True, force_finite=True)
    array([1, 1, 1, 0, 0, 0])

    Args:
        array: (np.ndarray), the array to be grouped. 1D array
        tol: (float), tolerance distance for spilt.(less than interlayer spacing)
        ignore_index: (int, np.ndarray), jump 'out-of-station' atoms.
        force_plane: (bool), change the single atom to the nearest group.
        force_finite: (bool), force to change the index finite.
        reverse: (bool), reverse the label.
        method: deprecated.
        n_cluster: deprecated.

    Returns:
        labels: (np.ndarray) with shape (n,).

    """
    array = array.ravel()

    if isinstance(ignore_index, int):
        ignore_index = [ignore_index, ]

    mark = np.full_like(array, True).astype(bool)
    res_label = np.full_like(array, -np.inf)  # all is -np.inf now
    if ignore_index is not None:
        mark[ignore_index] = False
        array = array[mark]

    sel = np.where(array)[0]  # -np.inf index

    label = coarse_cluster_array(array, tol=tol, reverse=reverse, method=method, n_cluster=n_cluster)

    # 1. 分配
    # decrease 为缺失组
    # increase 为冗余组
    # err 为独立值

    if force_plane:
        err, decrease, increase = _get_err_decrease_and_increase(label)
        if err is None and decrease is None and increase is None: # s1
            pass  # 无错误
        elif err is None and decrease is None and increase is not None:# s2
            pass  # 多余自动划分好
        elif err is None and decrease is not None and increase is None:# s3
            pass  # 少1，可能为ignore_index跳过,后续补全, s4 ignore_index处理后，将运行到s3
        elif err is None and decrease is not None and increase is not None:# s4
            # 划分错误，将一层的一个原子划分到另一层。
            # deindex = np.where(label == decrease)[0]
            inindex = np.where(label == increase)[0]
            lab = coarse_cluster_array(array[inindex], tol=tol/2, reverse=reverse)
            if len(set(lab))==2:
                nim = np.argmin(lab)
                axm = np.argmax(lab)
                if len(lab==lab[axm])>len(lab==lab[nim]):
                    err_x = nim
                else:
                    err_x = axm
                err_index = inindex[err_x]
                label[err_index] = decrease
            else:
                raise ValueError("One atom is split to other layer, (Alayer+1 and Blayer-1)\n"
                                 "Please 1. change (decrease) the tolerance or 2.offer reformed array"
                                 "3. using  ignore_index and force_infinite to fix it.")
        elif err is not None and decrease is None: #(increase in[None, not None])# s5,6
            # 进行吸附原子压平，强制分为最近邻组。通常这种操作是错的。
            if increase is not None:
                warnings.warn("There are more than one single atoms, please check it!")
            warnings.warn("Try to force dispense the single atom to nearist unbroken group, \n"
                          "please check carefully !!! or change the force_plane=False")
            err_index = np.where(label == err)[0]
            err_array = np.mean(array[err_index])
            loss_index = np.argsort(np.abs(array - err_array))[1]
            loss = label[loss_index]
            label[err_index] = loss
        elif err is not None and decrease is not None and increase is None:# s7
            # 通常的错误划分
            err_index = np.where(label == err)[0]
            label[err_index] = decrease
        elif err is not None and decrease is not None and increase is not None:# s8
            # 无法定义的错误划分。
            raise NotImplementedError("Bad spilt! "
                                      "Please 1. change the tolerance or 2.offer reformed array."
                                      )
        else:
            pass

    res_label[sel] = label  # ignore is -np.inf

    # 2. 删除空组
    i = 0
    m = max(res_label)
    while i <= m:
        if len(np.where(res_label == i)[0]) == 0:
            s = min(res_label[res_label > i]) - i
            res_label[res_label > i] -= s
        i += 1
        m = max(res_label)

    # 3. 补全有理数
    # 适用于划分层（A-1,B+1）错误,如 s3，或者多余吸附原子，不适合放到上面
    # if force_plane:

    if force_finite:
        if np.any(np.isinf(res_label)):
            warnings.warn("Some atoms is not be split (grouped), The code try to reformed it,\n "
                          "but it could lead to an error, manual checking is suggested.")
            val_index = np.isfinite(res_label)
            err_index = np.isinf(res_label)

            res_label2 = res_label.copy()
            res_label2[err_index] = np.max(res_label[val_index])+1 # 改值,
            lll = coarse_cluster_array(array=res_label2.astype(np.float32), tol= 0.2, )
            err, decrease, increase = _get_err_decrease_and_increase(lll)

            if decrease is not None and increase is None:
                # 处理s3
                res_label[err_index] = np.where(lll == decrease)[0][0]
                warnings.warn("The add atoms is split one group, This could be error.")
            else:
                # 吸附
                if force_plane:
                    warnings.warn("Fore absorb system, force_plane is not plane the absorb atom.")
                res_label[err_index] = max(res_label[val_index]) + 1
                warnings.warn("The add atoms is split one group, This could be error.")

    if np.all(np.isfinite(res_label)):
        res_label = res_label.astype(int)

    return res_label


@functools.lru_cache(4)
def _get_same_level(names: Tuple, zs: Tuple, center: int, neighbors_name: Union[str, Tuple[str]] = "O",
                    n_cluster: int = 2, tol: float = 0.5, method: str = "k_means"):
    """Get the same level with center, escape repetitive computation."""
    # Make sure in same z-axis level, get the index of O
    names = np.array(names)
    zs = np.array(zs)
    if isinstance(neighbors_name, str):
        neighbors_name = [neighbors_name, ]

    o_index = [np.where(names == i)[0] for i in neighbors_name]
    o_index = list(itertools.chain(*o_index))
    center_m0_z_coords = zs[center]
    o_z_coords = zs[o_index]

    temp_index = coarse_cluster_array(np.abs(o_z_coords - center_m0_z_coords), tol=tol, n_cluster=n_cluster,
                                      method=method)

    samez_o_index = np.array(o_index)[temp_index == 0]
    return samez_o_index


def get_nearest_plane_atom_index(structure: Structure, center: int, special_name: Union[str, List, Tuple] = "O",
                                 axis: int = 2, n_cluster: int = 2, tol: float = 0.5, method: str = "k_means"
                                 ) -> np.ndarray:
    """
    Get the nearest atom indexes (one plane of special atoms) to center.

    Args:
        structure: (Structure), pymatgen structure.
        center: (float), base reference atom.
        special_name: (str,List,Tuple), special atom names.
        axis: (int), along with axis.
        tol: (float) tolerance distance for spilt.
        method:(str) default None. others: "agg", "k_means", "cluster", "k_means_user".
        n_cluster: (int) number of cluster.

    Returns:
        index:(np.ndarray), array of atoms index (Relative to the whole structure's atom list).
    """
    if special_name is None:
        special_name = structure.symbol_set
    if isinstance(special_name, str):
        special_name = (special_name,)
    if isinstance(special_name, list):
        special_name = tuple(list(set(special_name)))
    names = [i.specie.name for i in structure.sites]
    names = tuple(names)
    zs = tuple(structure.cart_coords[:, axis].tolist())
    return _get_same_level(names, zs, center, neighbors_name=special_name, tol=tol, n_cluster=n_cluster,
                           method=method)


def get_plane_neighbors_to_center_raw(st: Structure, center, neighbors_name: Union[List, Tuple, str] = "O",
                                      ignore_index: Union[int, List, Tuple] = None,
                                      r=6.0, plane=True, n_cluster=2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get neighbor to center atom for next process.

    Args:
        st: (Structure),
        center: (str), name of center.
        neighbors_name: (str, tuple of str) neighbors_name or names tuple.
        ignore_index:(int,list),  ignore_index. refer to all structure atom list.
        r: (float), cut radius.
        plane: just for nearest plane.
        n_cluster: n for plane.

    Returns:
        np.ndarray, np.ndarray, np.ndarray
    """
    # Make sure in same z-axis level, get the index of O
    sites = st.sites
    center_m0 = center

    if plane is True:
        samez_o_index = get_nearest_plane_atom_index(st, center, special_name=neighbors_name, n_cluster=n_cluster)
    else:
        if isinstance(neighbors_name, str):
            samez_o_index = st.indices_from_symbol(neighbors_name)
        else:
            neighbors_name = list(set(neighbors_name))
            samez_o_index = [st.indices_from_symbol(i) for i in neighbors_name]
            samez_o_index = list(itertools.chain(samez_o_index))

    if ignore_index is None:
        pass
    else:
        if isinstance(ignore_index, int):
            ignore_index = [ignore_index, ]
        for i in ignore_index:
            samez_o_index = np.delete(samez_o_index, np.where(samez_o_index == i))

    # get distance

    center_indices, points_indices, offset_vectors, distances = st.get_neighbor_list(r=r,
                                                                                     sites=[sites[center_m0], ],
                                                                                     numerical_tol=1e-4,
                                                                                     exclude_self=True)
    # Relative to the center
    temp_index2 = np.min(np.abs(points_indices - samez_o_index.reshape(-1, 1)), axis=0) == 0
    points_indices, offset_vectors, distances = points_indices[
        temp_index2], offset_vectors[temp_index2], distances[temp_index2]
    return points_indices, offset_vectors, distances


def get_plane_neighbors_to_center(st: Structure, neighbors_name: Union[List, Tuple, str] = "O",
                                  ignore_index: Union[int, List, Tuple] = None, center_index=44,
                                  r=6.0, top=3, tol=0.6, plane=True, n_cluster=2) -> Dict[int, Dict]:
    """
    Get neighbor to center atom.

    Args:
        center_index: center index.  center_name or center_index should be offered.
        st: (Structure), pymatgen structure.

        neighbors_name: (str, tuple of str) neighbors_name or names tuple.
        ignore_index:(int,list),  ignore_index. refer to all structure atom list.
        r: (float), cut radius.
        top: (int), return top group.
        tol: (float), tolerance for group.
        plane: just for nearest plane.
        n_cluster: number of plane
    Returns:
        points_and_distance_to_center: (dict), points and distance.
    """
    # Make sure in same z-axis level, get the index of O
    sites = st.sites

    if plane is True:
        samez_o_index = get_nearest_plane_atom_index(st, center_index, special_name=neighbors_name, n_cluster=n_cluster)
    else:
        if isinstance(neighbors_name, str):
            samez_o_index = np.array(list(st.indices_from_symbol(neighbors_name)))
        else:
            neighbors_name = list(set(neighbors_name))
            samez_o_index = [st.indices_from_symbol(i) for i in neighbors_name]
            samez_o_index = np.array(list(itertools.chain(samez_o_index)))

    if ignore_index is None:
        pass
    else:
        if isinstance(ignore_index, int):
            ignore_index = [ignore_index, ]
        for i in ignore_index:
            samez_o_index = np.delete(samez_o_index, np.where(samez_o_index == i))

    # get distance

    center_indices, points_indices, offset_vectors, distances = st.get_neighbor_list(r=r,
                                                                                     sites=[sites[center_index], ],
                                                                                     numerical_tol=1e-4,
                                                                                     exclude_self=True)
    # Relative to the center
    temp_index2 = np.min(np.abs(points_indices - samez_o_index.reshape(-1, 1)), axis=0) == 0
    points_indices, offset_vectors, distances = points_indices[
        temp_index2], offset_vectors[temp_index2], distances[temp_index2]

    # get top 3 neighbors
    labels = coarse_cluster_array(distances, tol=tol, method=None)
    points_and_distance_to_m0 = {}

    if top == 1:
        points_and_distance_to_m0_si = {}
        points_indices0 = points_indices[labels == 0]
        offset_vectors = offset_vectors[labels == 0, :]
        distances0 = distances[labels == 0]
        for k, v, off in zip(points_indices0, distances0, offset_vectors):
            points_and_distance_to_m0_si.update({k: (v, off)})
        return {0: points_and_distance_to_m0_si}

    for i in range(top):
        points_and_distance_to_m0_si = {}
        points_indices0 = points_indices[labels == i]
        offset = offset_vectors[labels == i, :]
        distances0 = distances[labels == i]
        for k, v, off in zip(points_indices0, distances0, offset):
            points_and_distance_to_m0_si.update({k: (v, off)})
        points_and_distance_to_m0.update({i: points_and_distance_to_m0_si})

    return points_and_distance_to_m0


def get_center_name(structure: Structure, jump_atom_type: Tuple = ("C", "O", "N", "H", "P"),
                    center_index: int = None, ref_center_index=None,
                    ignore_index: Union[int, List, Tuple] = None) -> Tuple:
    """
    Judge the center and return the name of center, return name and index.

    Args:
        structure: (Structure), pymatgen structure.
        jump_atom_type: (tuple,), jump atom type.
        center_index: (int), use the center_index directly.
        ignore_index:(int,list),  ignore_index. refer to all structure atom list.
        ref_center_index:(int), if cant find center, use this ref_center_index.

    Returns:
        name:(str), center name.
        index:(int), center index.
    """

    if center_index is not None:
        return structure.sites[center_index].specie.name, center_index

    if isinstance(ignore_index, int):
        ignore_index = [ignore_index, ]
    if isinstance(ignore_index, (tuple, list)):
        sites = [s for i, s in enumerate(structure.sites) if i not in ignore_index]
    else:
        sites = structure.sites

    counts = Counter([i.species_string for i in sites if i.species_string not in jump_atom_type])
    counts = list(counts.items())
    counts2 = sorted(counts, key=lambda x: x[1], reverse=False)
    counts2 = [i for i in counts2 if i[1] == 1]

    if len(counts2) == 1:
        name = counts2[0][0]
        index_temp = structure.indices_from_symbol(name)
        if isinstance(ignore_index, (tuple, list)):
            index = [i for i in index_temp if i not in ignore_index][0]
        else:
            index = index_temp[0]
        return name, index
    if len(counts2) == 0:
        if ref_center_index is not None:
            return structure.sites[ref_center_index].specie.name, ref_center_index
        else:
            raise IndexError("Can't find the center. May be there is no single metal element. \n"
                             "For pure base materials, "
                             "please pass the center_index(center_m0) manually, such as: 'center_index=44',\n"
                             "if the number of element with the same with center is more than 1, "
                             "please jump them, such as  by 'ignore_index=[45,]'.")
    else:
        indexes = np.array([structure.indices_from_symbol(i[0])[0] for i in counts2])
        wh = np.argmin(indexes)
        name = counts2[wh][0]
        return name, indexes[wh]


def get_common_name(structure: Structure, jump_atom_type=("C", "O", "N", "H", "P")) -> Tuple:
    """
    Get the base name and indexes, such as for find base metal.

    Args:
        structure: (Structure),
        jump_atom_type: (tuple), jump atom type.

    Returns:
        name:(str), base name.
        index: (tuple), all base indexes.
    """

    names = [i.specie.name for i in structure.sites]
    names = [i for i in names if i not in jump_atom_type]
    counter_name = Counter(names)
    base_metal = counter_name.most_common(1)[0][0]
    return base_metal, structure.indices_from_symbol(base_metal)


def check_random_state(seed):
    """
    Check the random state the same as sklearn.

    Args:
        seed: None, int or instance of RandomState
            If seed is None, return the RandomState singleton used by np.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            Otherwise, raise ValueError.

    Returns:
        RandomState
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


class Interp2dNearest:
    """An interp2d method to get grid interpolation by neighbor points.
    interp2d means the approximate method is just in x, y axes and assess the value of z.
    Return the 'new_z' sites of the input grid (new_x,new_y).

    Examples
    # Get
    >>> x, y = np.meshgrid(np.arange(20), np.arange(10))
    >>> z = np.random.random((20,10))
    >>> iterp = Interp2dNearest(x, y, z)
    # Use
    >>> x = np.arange(-12, 23, 0.2)
    >>> y = np.arange(0, 21, 0.2)
    >>> z = iterp(x,y)
    # Show
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(z.T)
    >>> plt.show()
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, s: float = 2.0):
        """

        Args:
            x: 1D array, initial x data.
            y: 1D array, initial y data.
            z: 1D array, initial z data.
            s: float, smoothness factor.
        """
        assert len(x) == len(y) == len(z)
        self.tz = z
        self.s = s
        self.t_data = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))

    def __call__(self, new_x: np.ndarray, new_y: np.ndarray, mesh: bool = True) -> np.ndarray:
        """
        Interpolate in grid.

        Args:
            new_x: 1D array, x-coordinates of the mesh on which to interpolate.
            new_y: 1D array, y-coordinates of the mesh on which to interpolate.
            mesh: bool, if False, new_x, new_y should be the grid (Each xi, and yi point with same size),
                  please offer by yourself.
                  if True (default), use the x and y to get grid by np.meshgrid function.
        Returns:
            new_z : 2D array, with shape (len(x), len(y)) if not meshed else len(x). The interpolated values.

        """
        t_data = self.t_data
        tz = self.tz

        x = atleast_1d(new_x)
        y = atleast_1d(new_y)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y should both be 1-D arrays")
        if not mesh:
            x_mesh, y_mesh = np.meshgrid(x, y)
        else:
            assert x.shape[0] == y.shape[0]
            x_mesh, y_mesh = x, y

        x_mesh = np.ravel(x_mesh)
        y_mesh = np.ravel(y_mesh)
        data = np.hstack((x_mesh.reshape(-1, 1), y_mesh.reshape(-1, 1)))

        data_ = data[np.newaxis, :, :]
        data_ = np.repeat(data_, t_data.shape[0], axis=0)

        t_data_ = t_data[:, np.newaxis, :]
        t_data_ = np.repeat(t_data_, data.shape[0], axis=1)

        dis = np.sum((t_data_ - data_) ** 2, axis=-1) ** 0.5

        w = np.exp(-dis * self.s)
        w = w / np.sum(w, axis=0)
        new_z = np.sum(w * np.repeat(tz.reshape(-1, 1), w.shape[1], axis=1), axis=0)
        if not mesh:
            new_z = new_z.reshape((y.shape[0], x.shape[0])).T
        return new_z


def fixed_poscar(poscar: str, fixed_type: Union[str, float, None] = "base",
                 fixed_array: np.ndarray = None, coords_are_cartesian=True, cover=False) -> None:
    """
    Fix the atom in poscar with selective_dynamics.

    Args:
        poscar: str, file name.
        fixed_type: str,float,None.
            (1) if is float, the atoms in z axis larger than the float would be fixed.
            (2) if is "base", the last atoms would be fixed.
            (2) if is None, no fixed.
        fixed_array: np.ndarray, array of bool. if fixed array, use this array to fixed directly.
        coords_are_cartesian: bool, The coords is cartesian or not, for fixed_type.
        cover: bool, generate one new file or cover the old.

    Returns:
        None. The result is stored in file.
    """
    if fixed_type is None:
        pass
    else:
        pathi = pathlib.Path(poscar)
        poscar = Poscar.from_file(poscar, check_for_POTCAR=False)
        if fixed_array is None:
            st = poscar.structure
            fixed_array = np.full_like(poscar.structure.frac_coords, False)

            if fixed_type == "base":
                fixed_array[-1] = True
            else:
                if isinstance(fixed_type, float):
                    if coords_are_cartesian is False:
                        index = st.frac_coords[:, -1] > fixed_type
                    else:
                        index = st.cart_coords[:, -1] > fixed_type
                    fixed_array[index] = True
        poscar = Poscar(poscar.structure, selective_dynamics=fixed_array.tolist())
        if cover:
            poscar.write_file(str(pathi))
        else:
            poscar.write_file(str(pathi) + "(1)")
