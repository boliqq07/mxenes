# -*- coding: utf-8 -*-

# @Time     : 2021/10/25 17:23
# @Software : PyCharm
# @Author   : xxx

"""This is a structure extractor for one atom doping 2D MXenes-STM, such as Ti2CO2-STM."""
import functools
import itertools
import numbers
import pathlib
import warnings
from collections import Counter
from copy import deepcopy
from typing import List, Union, Dict, Tuple

import numpy as np
from numpy import atleast_1d
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar


def middle(st: Structure, deprecated=None, tol=0.01):
    """计算核心层，最终中间层"""
    frac = st.frac_coords[:, -1]
    if deprecated:
        mark = np.full_like(frac, True)
        mark[deprecated] = False
        frac_mark = frac[deprecated]
    else:
        frac_mark = frac
    mid = np.mean(frac_mark)
    middle_atoms = [i for i, z in enumerate(frac) if abs(z - mid) < tol]
    return middle_atoms


def coarse_and_spilt_array(array: np.ndarray, tol: float = 0.5, method: str = None, n_cluster: int = 3,
                           reverse: bool = False) -> np.ndarray:
    """
    Split 1D ndarray by distance or group.

    Args:
        array: (np.ndarray) with shape (n,).
        tol: (float) tolerance distance for spilt.
        method:(str) default None. others: "agg", "k_means", "cluster", "k_means_user".
        n_cluster: (int) number of cluster.
        reverse:(bool), reverse the label.

    Returns:
        labels: (np.ndarray) with shape (n,).

    """
    if method in ["agg", "k_means"]:
        if method == "agg":
            from sklearn.cluster import AgglomerativeClustering
            ac = AgglomerativeClustering(n_clusters=None, distance_threshold=tol, compute_distances=True)
        else:
            from sklearn.cluster import KMeans
            ac = KMeans(n_clusters=n_cluster)

        ac.fit(array.reshape(-1, 1))
        labels_ = ac.labels_
        labels_max = np.max(labels_)
        labels = deepcopy(labels_)
        dis = np.array([np.mean(array[labels_ == i]) for i in range(labels_max + 1)])
        dis_index = np.argsort(dis)
        for i in range(labels_max + 1):
            labels[labels_ == i] = dis_index[i]
        if reverse:
            labels = max(labels) - labels
        return labels
    else:
        # use tol directly
        array = array.ravel()
        array_sindex = np.argsort(array)
        array_sort = array[array_sindex]
        i = 0
        label = 0
        labels = []
        while i < len(array_sort):
            if i == 0:
                labels.append(label)
            else:
                if array_sort[i] - array_sort[i - 1] < tol:
                    labels.append(label)
                else:
                    label += 1
                    labels.append(label)
            i += 1
        dis_index = np.argsort(array_sindex)
        labels = np.array(labels)[dis_index]

        if reverse:
            labels = max(labels) - labels

        return labels


def coarse_and_spilt_array_ignore_force_plane(array, ignore_index=None, n_cluster=None, tol=0.5,
                                              method=None, force_plane=True, reverse=True):
    """
    Split sites by distance or group (default z-axis).

    Args:
        tol: (float) tolerance distance for spilt.
        method:(str) default None. others: "agg", "k_means", None.
        n_cluster: (int) number of cluster for "agg", "k_means".
        ignore_index: jump some atoms.
        force_plane: change the single atom to nearest group.
        reverse: reverse the label.

    Returns:
        labels: (np.ndarray) with shape (n,).

    """
    if isinstance(ignore_index, int):
        ignore_index = [ignore_index, ]

    mark = np.full_like(array, True).astype(bool)
    res_label = np.full_like(array, -np.inf)
    if ignore_index is not None:
        mark[ignore_index] = False
        array = array[mark]
    sel = np.where(array)[0]

    label = coarse_and_spilt_array(array, tol=tol, method=method, n_cluster=n_cluster, reverse=reverse)

    # if len(label) < len(array):
    label_dict = Counter(label)
    common_num = Counter(list(label_dict.values())).most_common(1)[0][0]
    # common_num = label_dict.most_common(1)[0][1]

    force_label = []
    for k in label_dict.keys():
        if label_dict[k] >= common_num - 2 and label_dict[k] < common_num:
            force_label.append(k)
    if len(force_label) > 1:
        raise TypeError("just for single doping.")
    elif len(force_label) == 1:
        force_label = force_label[0]
    else:
        force_label = None
    if force_plane and common_num >= 4:
        err = []
        d = list(set(label))
        d.sort()
        for k in d:
            if label_dict[k] <= 2:
                err.append(k)
        if len(err) > 1:
            warnings.warn("Just for single doping, if absorbed, please use ignore_index to jump the absorb atoms",
                          UserWarning)
            ds = np.array([array[:, -1][label == i] for i in err])
            err_index = np.argsort(np.mean(array[:, -1]) - ds)
            k = np.array(err)[err_index][0]
            k_index = np.where(label == k)[0]
            k_array = np.mean(array[k_index])
            if force_label is None:
                warnings.warn("Try to force dispense the single atom to group, please check carefully.", UserWarning)
                force_index = np.argsort(np.abs(array - k_array))[1]
                force_label = label[force_index]
                label[k_index] = force_label
            else:
                label[k_index] = force_label

        elif len(err) == 1:
            k = err[0]
            k_index = np.where(label == k)[0]
            k_array = np.mean(array[k_index])
            if force_label is None:
                warnings.warn("Try to force dispense the single atom to group, please check carefully.")
                force_index = np.argsort(np.abs(array - k_array))[1]
                force_label = label[force_index]
                label[k_index] = force_label
            else:
                label[k_index] = force_label
        else:
            pass

    res_label[sel] = label

    i = 0
    m = max(res_label)
    while i <= m:
        if len(np.where(res_label == i)[0]) == 0:
            s = min(res_label[res_label > i]) - i
            res_label[res_label > i] -= s
        i += 1
        m = max(res_label)
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

    temp_index = coarse_and_spilt_array(np.abs(o_z_coords - center_m0_z_coords), tol=tol, n_cluster=n_cluster,
                                        method=method)

    samez_o_index = np.array(o_index)[temp_index == 0]
    return samez_o_index


def get_nearest_plane_atom_index(structure: Structure, center: int, special_name: Union[str, List, Tuple] = "O",
                                 axis: int = 2,
                                 n_cluster: int = 2, tol: float = 0.5, method: str = "k_means"
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
                                      r=6.0, plane=True, n_cluster=2) -> tuple:
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
        np.ndarray,np.ndarray,np.ndarray
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


def get_plane_neighbors_to_center(st: Structure, center, neighbors_name: Union[List, Tuple, str] = "O",
                                  ignore_index: Union[int, List, Tuple] = None,
                                  r=6.0, top=3, tol=0.6, plane=True, n_cluster=2) -> Dict[int, Dict]:
    """
    Get neighbor to center atom.

    Args:
        st: (Structure),
        center: (str), name of center.
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
    center_m0 = center

    if plane is True:
        samez_o_index = get_nearest_plane_atom_index(st, center, special_name=neighbors_name, n_cluster=n_cluster)
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
                                                                                     sites=[sites[center_m0], ],
                                                                                     numerical_tol=1e-4,
                                                                                     exclude_self=True)
    # Relative to the center
    temp_index2 = np.min(np.abs(points_indices - samez_o_index.reshape(-1, 1)), axis=0) == 0
    points_indices, offset_vectors, distances = points_indices[
                                                    temp_index2], offset_vectors[temp_index2], distances[temp_index2]

    # get top 3 neighbors
    labels = coarse_and_spilt_array(distances, tol=tol, method=None)
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
                    center_index: int = None, ref_center_index=None, ignore_index: [int, List, Tuple] = None) -> Tuple:
    """
    Judge the center and return the name of center, return name and index.

    Args:
        structure: (Structure),
        jump_atom_type: (tuple,), jump atom type.
        center_index: (int), use the centor_index directly.
        ignore_index:(int,list),  ignore_index. refer to all structure atom list.
        ref_center_index:(int), if cant find center, use this ref_center_index.

    Returns:
        name:(str), center name.
        index:(int), centor index.
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
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


class interp2d_nearest:
    def __init__(self, x, y, z, s=2):
        # self.tx=x
        # self.ty=y
        self.tz = z
        self.s = s

        self.t_data = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))

    def __call__(self, x, y, meshed=True):
        """Interpolate the function.

        Parameters
        ----------
        x : 1-D array
            x-coordinates of the mesh on which to interpolate.
        y : 1-D array
            y-coordinates of the mesh on which to interpolate.
        meshed: bool
            if False: x_mesh, y_mesh = np.meshgrid(x,y), else use the x,and y directly.

        Returns
        -------
        z : 2-D array with shape (len(x), len(y)) if not meshed else len(x)
            The interpolated values.
        """
        t_data = self.t_data
        tz = self.tz

        x = atleast_1d(x)
        y = atleast_1d(y)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y should both be 1-D arrays")
        if not meshed:
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
        z = np.sum(w * np.repeat(tz.reshape(-1, 1), w.shape[1], axis=1), axis=0)
        if not meshed:
            z = z.reshape((y.shape[0], x.shape[0])).T
        return z


def fixed_poscar(poscar, fixed_type: Union[str, float, None] = "base", fixed_array=None, coords_are_cartesian=True):
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
        poscar.write_file(str(pathi))
