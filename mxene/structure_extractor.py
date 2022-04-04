import functools
import itertools
import warnings
from collections import Counter
from copy import deepcopy
from multiprocessing import Pool
from typing import List, Union, Dict

import numpy as np
import pandas as pd
from pymatgen.core import Structure, Element

from mgetool.tool import coarse_and_spilt_array
from mxene.functions import get_common_name, get_center_name, get_plane_neighbors_to_center, \
    get_nearest_plane_atom_index


def tetra_sites(st: Structure, base_metal="Ti", s: int = None, r=6.0, tol=0.6, ignore_index=45, center_m0=None,
                return_numpy=True):
    """
    Get tetra sites (4 sites) of one O.

    Args:
        st: (Structure), pymatgen structure.
        base_metal: (str), base metal name.
        center_m0: (int), index of center element. None is automatic judgement.
        s: (int,), specific the index of O manually.
        None is automatic judgement.
        return_numpy: (bool), return type, default numpy. or could be dict.
        r: (float), cut radius.
        ignore_index: (int,list), jump index for judge center. such as adsorbed atoms.
        tol: (float), tolerance to find neighbor.

    Returns:
        sites_msg: (dict), if np.ndarray, shape(20,) [center (1),neighbors(3),distance_c_vs_n (4*1=4),offset(4*3=12)].
    """
    if isinstance(st, str):
        st = Structure.from_file(st)

    if base_metal is None:
        base_metal, _ = get_common_name(st)
    else:
        counts = Counter([i.species_string for i in st.sites])
        assert counts[base_metal] >= 3, f"Please heck your base metal, is '{base_metal}' or not."

    center_m0_name, center_m0 = get_center_name(st, center_index=center_m0, ignore_index=ignore_index)

    points_and_distance_to_si = get_plane_neighbors_to_center(st, center=s, neighbors_name=(base_metal, center_m0_name),
                                                              ignore_index=ignore_index,
                                                              r=r, top=1, tol=tol)

    points_and_distance_to_si = points_and_distance_to_si[0]
    triangular_pyramid_sites = {s: points_and_distance_to_si}
    if return_numpy:
        # np.array type result for use convenience.
        for k, v in triangular_pyramid_sites.items():
            resi = [k, ]
            resi.extend(v.keys())

            for d in zip([0.0, np.array([0.0, 0.0, 0.0])], list(v.values())[0], list(v.values())[1],
                         list(v.values())[2]):
                if isinstance(d[0], float):
                    resi.extend(d)
                else:
                    [resi.extend(i) for i in d]

            return np.array(resi)

            # shape(20,) [center (1),neighbors(3),distance_c_vs_n (4*1=4),offset(4*3=12)]

    else:
        return points_and_distance_to_si


def s012_tetra_sites(st: Structure, base_metal="Ti", center_m0=None, s=None, return_numpy=True, r=6.0,
                     tol=0.6, ignore_index=45):
    """
    Get tetra sites (4 sites) of three O's sites: S0,S1,S2.

    Args:
        st: (Structure), pymatgen structure.
        base_metal: (str), base metal name.
        center_m0: (int), index of center element. None is automatic judgement.
        s: (list), list of all top site indexes of tetra (such as for three O atoms). specific the index of O manually.
        None is automatic judgement.
        return_numpy: (bool), return type, default numpy. or could be dict.
        r: (float), cut radius.
        ignore_index: (int,list), jump index for judge center. such as adsorbed atoms.
        tol: (float), tolerance to find neighbor.

    Returns:
        sites_msg: (dict,np.ndarray), if np.ndarray : for each line with shape(20,) [center (1),neighbors(3),
        distance_c_vs_n (4*1=4),offset(4*3=12)].
    """

    if isinstance(st, str):
        st = Structure.from_file(st)

    if base_metal is None:
        base_metal, _ = get_common_name(st)
    else:
        counts = Counter([i.species_string for i in st.sites])
        assert counts[base_metal] >= 3, f"Please heck your base metal, is '{base_metal}' or not."

    center_m0_name, center_m0 = get_center_name(st, center_index=center_m0, ignore_index=ignore_index)

    if s is None:
        points_and_distance_to_m0 = get_plane_neighbors_to_center(st, center=center_m0, ignore_index=ignore_index,
                                                                  r=6.0, top=3)
        s0 = list(points_and_distance_to_m0[0].keys())[0]
        s1 = list(points_and_distance_to_m0[1].keys())[0]
        s2 = list(points_and_distance_to_m0[2].keys())[0]
        s = [s0, s1, s2]

    triangular_pyramid_sites = {}
    for i, si in enumerate(s):
        points_and_distance_to_si = get_plane_neighbors_to_center(st, center=si,
                                                                  neighbors_name=(base_metal, center_m0_name),
                                                                  ignore_index=ignore_index,
                                                                  r=r, top=1, tol=tol)

        points_and_distance_to_si = points_and_distance_to_si[0]

        if i == 0 and len(points_and_distance_to_si) == 2:
            # for m0-o0, with large deformation, add it manually.
            points_and_distance_to_si[center_m0] = (
                np.sum(((st.cart_coords[center_m0] - st.cart_coords[si]) ** 2)) ** 0.5,
                np.array([0.0, 0.0, 0.0]))
        elif len(points_and_distance_to_si) != 3:
            raise UserWarning("Bad structure, 1. try please pass the top sites to 's', or set point4 manually by "
                              "'set_point4','setp4i'")
        triangular_pyramid_sites[si] = points_and_distance_to_si

    point_array = []
    if return_numpy:
        # np.array type result for use convenience.
        for k, v in triangular_pyramid_sites.items():
            resi = [k, ]
            resi.extend(v.keys())

            for d in zip([0.0, np.array([0.0, 0.0, 0.0])], list(v.values())[0], list(v.values())[1],
                         list(v.values())[2]):
                if isinstance(d[0], float):
                    resi.extend(d)
                else:
                    [resi.extend(i) for i in d]

            point_array.append(np.array(resi))

        # [center,neighbor,*,distance_c_vs_n,offset,distance_c_vs_n,offset,distance_c_vs_n,offset]
        return point_array
    else:
        return triangular_pyramid_sites


class SingleTetra:
    """Single Automic Tetra message extractor."""

    def __init__(self, structure: Structure, n_jobs=1, ref_center_m0=None, ignore_index=None, prefix=None):
        """

        Args:
            structure: (Structure)
            n_jobs: (int), n_jobs
            ref_center_m0: (int), center index.
            ignore_index: (int,list), jump index.
            prefix: (str), prefix for name.
        """
        if isinstance(structure, str):
            structure = Structure.from_file(structure)
        self.st = structure
        self.lattice = self.st.lattice
        self.cart_coords = deepcopy(self.st.cart_coords)
        self.offsets = np.zeros_like(self.cart_coords).astype(float)
        self.period = True
        self.d_c = self.p2_line_c
        self.index = np.array([np.nan, np.nan, np.nan, np.nan])
        self.n_jobs = n_jobs
        self.p4i = None
        self.checked = False
        self.real_cart_coords = None
        self.name_maps = {}
        self.prefix = prefix

        self.ignore_index = ignore_index

        center_m0_name, center_m0 = get_center_name(structure, jump_atom_type=("C", "O", "N", "H", "P"),
                                                    ref_center_index=ref_center_m0, ignore_index=ignore_index)
        self.center_m0_name = center_m0_name
        self.center_m0 = center_m0

    def _maps(self, *a):
        """map the readable name."""
        return [self.name_maps[i] for i in a]

    # def __hash__(self):  # unstable
    #     return hash((list(el.species.keys())[0].Z for el in self.st.sites))

    def p2_axis(self, a, b, axis=0):
        a_, b_ = self.rcc(a), self.rcc(b)

        a, b = self._maps(a, b)
        return {f"p2a{axis}_{a}_{b}": a_[axis] - b_[axis]}

    def p2_line(self, a, b):
        a_, b_ = self.rcc(a), self.rcc(b)

        a, b = self._maps(a, b)
        return {f"p2l_{a}_{b}": (np.sum((a_ - b_) ** 2)) ** 0.5}

    @staticmethod
    def p2_line_c(a_c, b_c):
        return (np.sum((a_c - b_c) ** 2)) ** 0.5

    def p3_face(self, a, b, c):
        abc_c = self.rcc((a, b, c))
        l1, l2, l3 = self.d_c(abc_c[0], abc_c[1]), self.d_c(abc_c[1], abc_c[2]), self.d_c(abc_c[0], abc_c[2])
        p = (l1 + l2 + l3) / 2

        a, b, c = self._maps(a, b, c)
        return {f"p3f_{a}_{b}_{c}": np.sqrt(p * (p - l1) * (p - l2) * (p - l3))}

    def p3_face_c(self, a_c, b_c, c_c):
        l1, l2, l3 = self.d_c(a_c, b_c), self.d_c(b_c, c_c), self.d_c(a_c, c_c)
        p = (l1 + l2 + l3) / 2
        return np.sqrt(p * (p - l1) * (p - l2) * (p - l3))

    def p4_volume(self, a, b, c, d):
        c4 = self.rcc((a, b, c, d))

        a, b, c, d = self._maps(a, b, c, d)
        return {f"p4v_{a}_{b}_{c}_{d}": np.abs(np.linalg.det((c4 - c4[0])[1:, ]) / 6)}

    @staticmethod
    def p4_volume_c(a_c, b_c, c_c, d_c):
        c4 = np.vstack((a_c, b_c, c_c, d_c))
        return np.abs(np.linalg.det((c4 - c4[0])[1:, ]) / 6)

    @functools.lru_cache(10)
    def cc(self, p):
        """for cart-coords"""
        return self.cart_coords[p, :] + np.dot(self.offsets[p, :], self.lattice.matrix)

    @functools.lru_cache(10)
    def rcc(self, p):
        """for real_cart-coords, the offset has been add on it."""
        return self.real_cart_coords[p, :]

    def message(self, p4i=None, return_type='dict'):
        """
        Get Tetra message.

        Args:
            p4i: (np.ndarray), default is None, if specific, the p4i is the 4 index of elements (top,backend1,backend2,backend3).
            return_type: (str), default "dict", or could be "pd".

        Returns:
            message: (dict, pd.Dateframe)
        """

        warnings.filterwarnings("ignore")
        if p4i is not None:  # in default ,don't use it.
            self.set_p4i(p4i)

        assert isinstance(self.p4i, np.ndarray), "please set p4i, by tetra_sites automatically or set_4i manually"
        if self.checked is False:
            warnings.warn("please run self.preprocess first for re-arrange the rank of element index", UserWarning)
        # init
        self.init_point()

        # add 2 site
        # get the most commen o z-axis.
        if self.center_m0 is not None:
            sameo_index = get_nearest_plane_atom_index(self.st, center=self.center_m0, special_name="O")
            z_c = np.mean(self.real_cart_coords[sameo_index][:, -1])
        else:
            z_c = None

        # add 2 site
        init_c_c, init_p_c = self.estimation_init_cp(self.index, z_c)  # with offset
        self.add_point(cart_coord=init_c_c, name="ic", offset=self.p4i[8:11])
        self.add_point(cart_coord=init_p_c, name="ip", offset=self.p4i[11:14])

        mess = {}

        # 2p-axis
        for ai in [0, 1, 2]:
            for a, b in [[0, 4], [1, 5]]:
                # print(a, b)
                a, b = self.index[a], self.index[b]
                mess.update(self.p2_axis(a, b, axis=ai))
        # 2p
        for a, b in itertools.combinations(self.index, 2):
            # print(a, b)
            mess.update(self.p2_line(a, b))
        # 3p
        for a, b, c in itertools.combinations(self.index, 3):
            # print(a, b)
            mess.update(self.p3_face(a, b, c))

        # 4p
        for a, b, c, d in itertools.combinations(self.index, 4):
            # print(a, b)
            mess.update(self.p4_volume(a, b, c, d))
        #
        if return_type == "pd":
            if self.prefix is None:
                prefix_m0_name = self.center_m0_name
            else:
                prefix_m0_name = f"{self.prefix}-{self.center_m0_name}"

            mess = pd.DataFrame.from_dict({f"{prefix_m0_name}-{int(self.p4i[0])}-{int(self.p4i[1])}"
                                           f"-{int(self.p4i[2])}-{int(self.p4i[3])}": mess}).T
        return mess

    def estimation_init_cp(self, p4i_index, z_c=None):
        """estimation init c and init p."""
        # p4i_index point named : [center,special,point1,point2] or [c,s,p1,p2]
        # d_p1_p2_temp = self.d_c(self.cc(p4i_index[2])[:2],self.cc(p4i_index[3])[:2]) # the site if [p1,p2] just(x,y)
        x1, y1, z1 = self.cc(p4i_index[2])
        x2, y2, z2 = self.cc(p4i_index[3])
        if z_c is None:
            _, _, z_c = self.cc(p4i_index[0])

        # # z_c is not the true z axis of init_c !!!

        alpha1 = 0.8660254
        alpha2 = 0.2886751

        init_p1 = np.array([0.5 * x1 + 0.5 * x2 - alpha1 * y1 + alpha1 * y2,
                            y1 / 2 + y2 / 2 + (x1 - x2) * alpha1, (z1 + z2) / 2])

        init_p2 = np.array([0.5 * x1 + 0.5 * x2 + alpha1 * y1 - alpha1 * y2,
                            y1 / 2 + y2 / 2 - (x1 - x2) * alpha1, (z1 + z2) / 2])

        init_c1 = np.array([0.5 * x1 + 0.5 * x2 - alpha2 * y1 + alpha2 * y2,
                            y1 / 2 + y2 / 2 + (x1 - x2) * alpha2, z_c])

        init_c2 = np.array([0.5 * x1 + 0.5 * x2 + alpha2 * y1 - alpha2 * y2,
                            y1 / 2 + y2 / 2 - (x1 - x2) * alpha2, z_c])

        if self.d_c(init_p1, self.cc(p4i_index[0])) < self.d_c(init_p2, self.cc(p4i_index[0])):
            return init_c1, init_p1
        else:
            return init_c2, init_p2

    def init_point(self):
        # >>> for each p4i_index, reset it first!
        self.cart_coords = deepcopy(self.cart_coords)
        self.offsets = np.zeros_like(self.cart_coords).astype(float)
        origin_index4 = self.p4i[:4].astype(int)
        self.offsets[origin_index4] = self.p4i[8:].reshape(-1, 3)
        self.index = origin_index4
        self.real_cart_coords = self.cart_coords + np.dot(self.offsets, self.lattice.matrix)
        self.name_maps = {i: n for i, n in zip(self.index, ["c", "p", "a", "b"])}
        # <<< for each p4i_index, reset it first!

    def add_point(self, cart_coord, name, offset=None, ):
        """
        add new point.

        Args:
            cart_coord: (np.ndarray), with shape (3,)
            name: (str), name of this point
            offset: (np.ndarray), with shape (3,)

        """
        if offset is None:
            offset = np.array([0.0, 0.0, 0.0])
        self.cart_coords = np.concatenate(
            (self.cart_coords, cart_coord.reshape(1, 3) - np.dot(offset, self.lattice.matrix)), axis=0)

        self.offsets = np.concatenate((self.offsets, offset.reshape(1, 3)), axis=0)
        self.index = np.append(self.index, self.cart_coords.shape[0] - 1)
        self.real_cart_coords = self.cart_coords + np.dot(self.offsets, self.lattice.matrix)
        self.name_maps.update({self.cart_coords.shape[0] - 1: name})

    def tetra_sites(self, s: int, base_metal=None, r=6.0, tol=0.6):
        """
        Get neighbor sites of three O's sites.

        Args:
            base_metal: (str), name
            s: (int), specific the index of O.
            r: (float), cut radius.
            tol: (float), tolerance.

        Returns:
            None
        """

        self.p4i = tetra_sites(self.st, base_metal=base_metal, r=r, tol=tol,
                               center_m0=self.center_m0, s=s, return_numpy=True,
                               ignore_index=self.ignore_index)
        self.preprocess()

    def set_p4i(self, p4i, run_check="pymatgen"):
        """
        Set 4 point. please keep the center element in the first!!!.
        if run_check = "", please make sure the tetra msg ndarray manully.

        Args:
            p4i: (ArrayLike), if run_check, if specific, the p4i is the 4 index of elements, else pass tetra_msg ndarray
            directly.
            run_check: ("rank", "pymatgen", ""), method to find the offset.

        """

        if run_check == "rank":
            off = np.array([[0, 0, 0],
                            # 1D
                            [1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
                            # 2D-xy
                            [1, -1, 0], [-1, 1, 0], [1, 1, 0], [-1, -1, 0],
                            # 2D-others
                            [-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 1], [0, 1, -1], [0, 1, 1], [0, -1, -1],
                            [0, -1, 1],
                            # 3D
                            [-1, 1, -1], [-1, 1, 1], [-1, -1, -1], [-1, -1, 1], [1, 1, -1], [1, 1, 1], [1, -1, -1],
                            [1, -1, 1],
                            ])

            def get_distance(i0, j0):
                dis0 = 10
                offi0 = np.array([0, 0, 0])
                for offi0 in off:
                    dis0 = self.st.get_distance(i0, j0, jimage=offi0)
                    if dis0 <= 3.0:
                        break
                if dis0 > 3.0:
                    raise NotImplementedError("The point between {i} and {j} is large than 3.")
                return dis0, offi0

            p4i_temp = []
            i = p4i[0]
            dis_temp = []
            off_temp = []
            for j in p4i[:4]:
                dis, offi = get_distance(i, j)
                dis_temp.append(dis)
                off_temp.extend(offi)
            [p4i_temp.extend(k) for k in [p4i, dis_temp, off_temp]]
            p4i = np.array(p4i_temp, dtype=float)
            self.p4i = np.array(p4i)
            self.preprocess()

        elif run_check == "pymatgen":
            p4i_temp = []
            dis_temp = []
            off_temp = []
            for j in p4i[:4]:
                dis, offi = self.st[0].distance_and_image(self.st[j], None)
                dis_temp.append(dis)
                off_temp.extend(offi)
            [p4i_temp.extend(k) for k in [p4i, dis_temp, off_temp]]
            p4i = np.array(p4i_temp, dtype=float)
            self.p4i = np.array(p4i)
            self.preprocess()

        else:
            self.p4i = np.array(p4i)
            self.checked = True

    def preprocess(self):
        """Re-arrange the rank of 4 points. set the special neighbor point to index 1 (the 2ed site).
        """
        p4i = self.p4i
        assert p4i is not None, "Please get 4 point and set it by self.set_p4i."

        offset = p4i[8:]

        if np.all(offset == 0):
            self.period = False
        else:
            self.period = True

        distance = p4i[4:8]

        # print("Re-arrange the rank of 4 points. set the special neighbor point to index 1 (the 2ed site)")
        labels_ = coarse_and_spilt_array(distance, n_cluster=3, method="k_means_user")
        labels = deepcopy(labels_)
        _ = "shrink" if np.sum(labels) == 5 else "extend"
        if np.sum(labels) == 4:
            labels[labels_ == 1] = 2
            labels[labels_ == 2] = 1
        index = np.argsort(labels)
        p4i[:4] = p4i[:4][index]
        p4i[4:8] = p4i[4:8][index]
        p4i[8:] = (p4i[8:].reshape(-1, 3)[index]).ravel()
        self.p4i = p4i
        self.checked = True


class Tetra:
    """Tetra message extractor for three O's sites."""

    def __init__(self, structure: Structure, n_jobs=1, ref_center_m0=None, ignore_index=None, prefix=None):
        """
        Args:
            structure: (Structure)
            n_jobs: (int), n_jobs.
            ref_center_m0: (int), center index.
        """
        if isinstance(structure, str):
            structure = Structure.from_file(structure)
        self.st = structure
        self.point4 = []
        self.period = True
        self.n_jobs = n_jobs
        self.checked = False

        self.ignore_index = ignore_index

        center_m0_name, center_m0 = get_center_name(self.st, center_index=ref_center_m0, ignore_index=ignore_index)

        self.center_m0_name = center_m0_name
        self.center_m0 = center_m0
        self.prefix = prefix

    def message(self, number: Union[int, List[Union[int, np.ndarray]]] = None, return_type="dict"):
        """
        Get Tetra message.

        Args:
            number: (np.ndarray), index of of tera.
            if specific, the p4i is the 4 index of elements.
            return_type: (str), default "dict", or could be "pd".

        Returns:
            message: (dict, pd.Dateframe)
        """
        assert len(self.point4) > 0, "please set point4,by s012_tetra_sites automatically or set_point4 manually"

        if number is None:
            number = list(range(len(self.point4)))
        elif isinstance(number, int):
            number = [number, ]

        assert isinstance(number, (tuple, list))

        if self.prefix is None:
            prefix_m0_name = self.center_m0_name
        else:
            prefix_m0_name = f"{self.prefix}-{self.center_m0_name}"

        if self.n_jobs == 1:
            mess_all = {}
            for i, p4i in enumerate(number):
                sit = SingleTetra(self.st, self.n_jobs, ref_center_m0=self.center_m0, ignore_index=self.ignore_index)
                if isinstance(p4i, int):  # i==p4ii
                    sit.set_p4i(self.point4[p4i], run_check="")  # jump preprocessing
                else:
                    sit.set_p4i(p4i)
                mess = sit.message()
                mess_all.update({f"{prefix_m0_name}-S{i}": mess})
        else:
            sits = [SingleTetra(self.st, self.n_jobs, ref_center_m0=self.center_m0, ignore_index=self.ignore_index) for
                    _ in
                    number]
            point4 = [self.point4[i] for i in range(len(number))]
            if isinstance(number[0], int):  # i==p4ii
                [sit.set_p4i(p4i, run_check="") for sit, p4i in zip(sits, point4)]
            else:
                [sit.set_p4i(p4i) for sit, p4i in zip(sits, number)]
            pool = Pool(processes=self.n_jobs)
            res_ = []
            for sit in sits:
                res_.append(pool.apply_async(sit.message).get())
            pool.close()
            pool.join()
            mess_all = {f"{prefix_m0_name}-S{i}": mess for i, mess in zip(range(len(number)), res_)}

        if return_type == "pd":
            mess_all = pd.DataFrame.from_dict(mess_all).T

        return mess_all

    def s012_tetra_sites(self, base_metal=None, s=None, r=6.0, tol=0.6):
        """
        Get neighbor sites of three O's sites.

        Args:
            base_metal: (str), name
            s: (list). specific the index of O.
            r: (float), cut radius.
            tol:(float), tolerance.

        Returns:
            None
        """

        self.point4 = s012_tetra_sites(self.st, base_metal=base_metal, r=r, tol=tol,
                                       center_m0=self.center_m0, s=s, return_numpy=True,
                                       ignore_index=self.ignore_index)
        self.preprocess()

    def set_point4(self, point4: List[np.ndarray], run_check="pymatgen"):
        """
        Set 4 point. please keep the center element in the first!!!.

        Args:
            point4: (list of 3 np.ndarray), default is None, if specific, each p4i is the 4 index of elements.
            run_check: (str), method to find the offset.
        """

        if run_check == "rank":
            off = np.array([[0, 0, 0],
                            # 1D
                            [1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
                            # 2D-xy
                            [1, -1, 0], [-1, 1, 0], [1, 1, 0], [-1, -1, 0],
                            # 2D-others
                            [-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 1], [0, 1, -1], [0, 1, 1], [0, -1, -1],
                            [0, -1, 1],
                            # 3D
                            [-1, 1, -1], [-1, 1, 1], [-1, -1, -1], [-1, -1, 1], [1, 1, -1], [1, 1, 1], [1, -1, -1],
                            [1, -1, 1],
                            ])

            def get_distance(i, j):
                dis = 10
                offi = np.array([0, 0, 0])
                for offi in off:
                    dis = self.st.get_distance(i, j, jimage=offi)
                    if dis <= 3.0:
                        break
                if dis > 3.0:
                    raise NotImplementedError("The point between {i} and {j} is large than 3.")
                return dis, offi

            point4_temp = []

            for p4i in point4:
                p4i_temp = []
                i = p4i[0]
                dis_temp = []
                off_temp = []
                for j in p4i:
                    dis, offi = get_distance(i, j)
                    dis_temp.append(dis)
                    off_temp.extend(offi)
                [p4i_temp.extend(k) for k in [p4i, dis_temp, off_temp]]
                point4_temp.append(np.array(p4i_temp))
            self.point4 = point4_temp
            self.preprocess()

        elif run_check == "pymatgen":
            point4_temp = []
            for p4i in point4:
                p4i_temp = []
                dis_temp = []
                off_temp = []
                for j in p4i[:4]:
                    dis, offi = self.st[0].distance_and_image(self.st[j], None)
                    dis_temp.append(dis)
                    off_temp.extend(offi)
                [p4i_temp.extend(k) for k in [p4i, dis_temp, off_temp]]
                point4_temp.append(np.array(p4i_temp))
            self.point4 = point4_temp
            self.preprocess()

        else:
            self.point4 = point4
            self.checked = True

    def preprocess(self):
        """Re-arrange the rank of 4 points. set the special neighbor point to index 1 (the 2ed site).
        """
        point4 = self.point4
        assert point4 != [], "Please get 4 point by function self.s012_triangular_pyramid_sites fucnction()" \
                             " or set it by self.set_p4i."

        offset = np.vstack([i[8:] for i in point4])

        if np.all(offset == 0):
            self.period = False
        else:
            self.period = True

        distance = [i[4:8] for i in point4]
        for i, di in enumerate(distance):
            # print("Re-arrange the rank of 4 points. set the special neighbor point to index 1 (the 2ed site)")
            labels_ = coarse_and_spilt_array(di, tol=0.01, method="k_means_user")
            labels = deepcopy(labels_)
            _ = "shrink" if np.sum(labels) == 5 else "extend"
            if np.sum(labels) == 4:
                labels[labels_ == 1] = 2
                labels[labels_ == 2] = 1
            index = np.argsort(labels)
            point4[i][:4] = point4[i][:4][index]
            point4[i][4:8] = point4[i][4:8][index]
            point4[i][8:] = (point4[i][8:].reshape(4, 3)[index]).ravel()
        self.checked = True


def structure_message(structure: Structure, prefix=None) -> Dict:
    """Get message."""
    # Tetra

    data_same = {}

    m0_name, m0_index = get_center_name(structure, ref_center_index=44, ignore_index=45)
    base_name, _ = get_common_name(structure)
    tpm = Tetra(structure, n_jobs=1, ref_center_m0=44, ignore_index=45, prefix=None)
    tpm.s012_tetra_sites()
    data = tpm.message(return_type="dict")

    # others add
    m0 = Element(m0_name)
    data_same.update({"r_m0": m0.data["Atomic radius calculated"]})
    data_same.update({"r_m_base": Element(base_name).data["Atomic radius calculated"]})
    data_same.update({"r_o": Element("O").data["Atomic radius calculated"]})
    data_same.update({"r_c": Element("C").data["Atomic radius calculated"]})
    data_same.update({"X": m0.X})
    data_same.update({"Z": m0.Z})

    if prefix is None:
        prefix_m0_name = m0_name
    else:
        prefix_m0_name = f"{prefix}-{m0_name}"

    data_same = {f"{prefix_m0_name}-Sall": data_same}

    data.update(data_same)

    return data


# def structure_message_simple_for_Mo(structure: Structure, prefix=None):
#     """For Mo."""
#     cart_coords = structure.cart_coords
#
#     data_same = {}
#     data_sep = {}
#     data = {}
#
#     m0_name = structure.species[44].symbol
#     data_same.update({"same_d_m0_o0": (13, 44)})
#     data_same.update({"same_d_m1_o0": (13, 33)})
#     data_same.update({"same_d_m0_c0": (5, 44)})
#
#     data_sep.update({"ind_d_m0_o0": ((13, 44), (16, 32), (11, 29))})
#     data_sep.update({"ind_d_m1_o0": ((13, 30), (16, 33), (11, 33))})
#
#     if prefix is None:
#         prefix_m0_name = m0_name
#     else:
#         prefix_m0_name = f"{prefix}-{m0_name}"
#
#     for k, v in data_same.items():
#         data_same[k] = structure.get_distance(v[0], v[1])
#
#     for k, v in data_sep.items():
#         for i, vi in enumerate(v):
#             vi = structure.get_distance(vi[0], vi[1])
#             if f"{prefix_m0_name}-S{i}" not in data:
#                 data[f"{prefix_m0_name}-S{i}"] = {k: vi}
#             else:
#                 data[f"{prefix_m0_name}-S{i}"].update({k: vi})
#
#     dz = cart_coords[29][2] - cart_coords[44][2]
#     data_same.update({"same_dz": dz})
#
#     data_same.update({"r_m0": structure.species[44].data["Atomic radius calculated"]})
#     data_same.update({"r_m_base": structure.species[33].data["Atomic radius calculated"]})
#     data_same.update({"r_o": structure.species[11].data["Atomic radius calculated"]})
#     data_same.update({"r_c": structure.species[5].data["Atomic radius calculated"]})
#     data_same.update({"X": structure.species[44].X})
#     data_same.update({"Z": structure.species[44].Z})
#
#     # structure_Mo = Poscar.from_file(r"C:\Users\Administrator\PycharmProjects\samples\Instance\Instance_mo2co2"
#     #                                 r"\MoCMo-O-4\Mo\pure_static\Prim_Mo2CO2_CONTCAR", check_for_POTCAR=False).structure
#     ini_mo = np.array([1.43176042, 4.13308642, 13.86915])  # structure_Mo.cart_coords[44]
#     ini_o = np.array([2.86348217, 4.95971115, 15.0897])  # structure_Mo.cart_coords[17]
#     init_r = np.sum((ini_mo[:2] - ini_o[:2]) ** 2) ** 0.5
#     mo = structure.cart_coords[44]
#     o = structure.cart_coords[17]
#     dr = np.sum((mo[:2] - o[:2]) ** 2) ** 0.5 - init_r
#     dh = mo[2] - ini_mo[2]
#     data_same.update({"dr": dr, "dh": dh})
#
#     data_same = {f"{prefix_m0_name}-Sall": data_same}
#
#     data.update(data_same)
#
#     return data
#
#
# def structure_message_simple(structure: Structure, prefix=None):
#     """For Ti Ta,Zr."""
#     cart_coords = structure.cart_coords
#
#     data_same = {}
#     data_sep = {}
#     data = {}
#
#     m0_name = structure.species[44].symbol
#     data_same.update({"same_d_m0_o0": (13, 44)})
#     data_same.update({"same_d_m1_o0": (13, 33)})
#
#     data_same.update({"same_d_m0_c0": (5, 44)})
#
#     data_sep.update({"ind_d_m0_o0": ((13, 44), (17, 29), (16, 32))})
#     data_sep.update({"ind_d_m1_o0": ((13, 33), (17, 33), (16, 33))})
#
#     if prefix is None:
#         prefix_m0_name = m0_name
#     else:
#         prefix_m0_name = f"{prefix}-{m0_name}"
#
#     for k, v in data_same.items():
#         data_same[k] = structure.get_distance(v[0], v[1])
#
#     for k, v in data_sep.items():
#         for i, vi in enumerate(v):
#             vi = structure.get_distance(vi[0], vi[1])
#             if f"{prefix_m0_name}-S{i}" not in data:
#                 data[f"{prefix_m0_name}-S{i}"] = {k: vi}
#             else:
#                 data[f"{prefix_m0_name}-S{i}"].update({k: vi})
#
#     dz = cart_coords[29][2] - cart_coords[44][2]
#     data_same.update({"same_dz": dz})
#
#     data_same.update({"r_m0": structure.species[44].data["Atomic radius calculated"]})
#     data_same.update({"r_m_base": structure.species[33].data["Atomic radius calculated"]})
#     data_same.update({"r_o": structure.species[11].data["Atomic radius calculated"]})
#     data_same.update({"r_c": structure.species[5].data["Atomic radius calculated"]})
#     data_same.update({"X": structure.species[44].X})
#     data_same.update({"Z": structure.species[44].Z})
#
#     data_same = {f"{prefix_m0_name}-Sall": data_same}
#
#     data.update(data_same)
#
#     return data


class DataSameTep():
    """settle data."""

    def __init__(self, data: Dict, sep="-", sites_name="S", dup=3, prefix=None):
        """
        Make sure the key are formatted by {label}-{Si or Sall} !!! and all values is dict type.

        Args:
            data: (dict), key are formated by {label}{sep}{Si or Sall}.
            sep: (str), default "-".
            sites_name: (str),default "S".
            dup: (int), default 3.
            prefix:(str), the class prefix of one batch data.
        """
        self.data = data
        self.sites_name = sites_name
        self.dup = dup
        self.data2 = {}
        self.sep = sep
        self.data3 = self.data
        self.label = []
        self.mark = [f"{sites_name}{i}" for i in range(dup)] + [f"{sites_name}all"]
        self.check()
        self.prefix = prefix

    def check(self):
        key = list(self.data.keys())
        try:
            labels, marks = list(zip(*[i.split(self.sep) for i in key]))
        except BaseException:
            try:
                prefixs, labels, marks = list(zip(*[i.split(self.sep) for i in key]))
                prefix = list(set(prefixs))
                if len(prefix) >= 1:
                    print(f"There are {len(prefix)} prefix.")
            except BaseException:
                raise ValueError("The key should named '{label}-{Sx}' or '{prefix}-{label}-{Sx}'. ")
        self.label = list(set(labels))
        mark = set(marks)

        assert set(self.mark) == mark

    def __setitem__(self, key, value: Dict):
        assert self.sep in key
        assert key.split(self.sep)[-1] in self.mark
        assert len(key.split(self.sep)) in [2, 3], "The key should named '{label}-{Sx}' or '{prefix}-{label}-{Sx}'."
        if self.prefix:
            if len(key.split(self.sep)) != 3:
                raise UserWarning(f"There are {self.prefix} in definition but the key  {key} is with out prefix."
                                  f"Advise use {{prefix}}-{{label}}-{{Sx}}")
        self.data[key].update(value)

    def update2(self, label: str, site: Union[int, str], value: Dict, prefix=None):
        if isinstance(site, int):
            pass
        else:
            assert f"{self.sites_name}{site}" in self.mark, "Keep site is int or 'all'."
        if prefix:
            key = f"{prefix}{self.sep}{label}{self.sep}{self.sites_name}{site}"
        elif self.prefix:
            key = f"{self.prefix}{self.sep}{label}{self.sep}{self.sites_name}{site}"
        else:
            key = f"{label}{self.sep}{self.sites_name}{site}"
        self.data.update({key: value})

    def settle(self):
        self.check()
        for key in self.data.keys():
            if "all" in key:
                label = key.replace(f"{self.sep}{self.sites_name}all", "")
                for site in range(self.dup):
                    nk = f"{label}{self.sep}{self.sites_name}{site}"
                    if nk in self.data2:
                        self.data2[nk].update(self.data[key])
                    else:
                        self.data2.update({nk: self.data[key]})
            else:
                if key in self.data2:
                    self.data2[key].update(self.data[key])
                else:
                    self.data2.update({key: self.data[key]})
        return self.data2

    def settle_to_pd(self):
        data = self.settle()
        return pd.DataFrame.from_dict(data).T
