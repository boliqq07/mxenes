import copy
import itertools
import pathlib
import warnings
from collections import Counter
from itertools import chain
from typing import Union, Sequence, List

import numpy as np
from ase.visualize import view
from numpy.typing import ArrayLike
from pymatgen.core import Structure, Lattice, SymmOp, PeriodicSite
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from pymatgen.util.typing import CompositionLike
except BaseException:
    CompositionLike = str

from mxene.functions import coarse_and_spilt_array_ignore_force_plane, get_plane_neighbors_to_center, \
    check_random_state, interp2d_nearest
from mxene.disk import make_disk
from mxene.structure_extractor import structure_message

aaa = AseAtomsAdaptor()


class MXene(Structure):
    """One structure for MXene.
    ###下面为常见函数###

    # 0.创建
    从种类创建虚拟结构：from_standard
    读取POSCAR等文件：from_file

    # 1.修改
    插入： insert
    添加： append
    替换为单原子： substitute
    替换为官能团： replace
    删除某个种类的所有原子：remove_species
    删除某个位点：remove_sites
    重排序：sort
    旋转：rotate_sites
    随机扰动：perturb
    超胞：make_supercell
    添加吸附物:add_absorb

    # 3.读取结构信息
    get_structure_message

    # 4.读取结构信息
    add_absorb: 添加吸附原子
    add_face_random_*****: 随机批量添加原子

    """

    nm_list = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I", "Te", "At"]
    tm_list = ["Sc", "Y", "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Mn",
               # "Tc",
               "Re", "Fe", "Ru", "Os", "Co", "Rh", "Ir", "Ni", "Pd", "Pt", "Cu", "Ag", "Au", "Zn", "Cd"]
    am_list = ["Al", "Ca", "Li", "Mg", "Na", "K", "Zn", "H"]
    tem_list = ["O", "F", "OH", "Cl", None]
    cn = ["C", "N"]
    bm = ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"]
    tem_z_axis = {"O": 1.0, "F": 0.9, "Cl": 1.2}
    am_z_axis = {"K": 2.1, "Li": 1.3, "H": 1.0, "Ca": 1.6, "Na": 1.6, "Mg": 1.4, "Al": 1.1, "Zn": 1.2}

    def __init__(self,
                 lattice: Union[ArrayLike, Lattice],
                 species: Sequence[CompositionLike],
                 coords: Sequence[ArrayLike],
                 charge: float = None,
                 validate_proximity: bool = False,
                 to_unit_cell: bool = False,
                 coords_are_cartesian: bool = False,
                 site_properties: dict = None, ):
        """
        Create a periodic structure.

        Args:
            lattice: The lattice, either as a pymatgen.core.lattice.Lattice or
                simply as any 2D array. Each row should correspond to a lattice
                vector. E.g., [[10,0,0], [20,10,0], [0,0,30]] specifies a
                lattice with lattice vectors [10,0,0], [20,10,0] and [0,0,30].
            species: List of species on each site. Can take in flexible input,
                including:

                i.  A sequence of element / species specified either as string
                    symbols, e.g. ["Li", "Fe2+", "P", ...] or atomic numbers,
                    e.g., (3, 56, ...) or actual Element or Species objects.

                ii. List of dict of elements/species and occupancies, e.g.,
                    [{"Fe" : 0.5, "Mn":0.5}, ...]. This allows the setup of
                    disordered structures.
            coords (Nx3 array): list of fractional/cartesian coordinates of
                each species.
            charge (int): overall charge of the structure. Defaults to behavior
                in SiteCollection where total charge is the sum of the oxidation
                states.
            validate_proximity (bool): Whether to check if there are sites
                that are less than 0.01 Ang apart. Defaults to False.
            to_unit_cell (bool): Whether to map all sites into the unit cell,
                i.e., fractional coords between 0 and 1. Defaults to False.
            coords_are_cartesian (bool): Set to True if you are providing
                coordinates in cartesian coordinates. Defaults to False.
            site_properties (dict): Properties associated with the sites as a
                dict of sequences, e.g., {"magmom":[5,5,5,5]}. The sequences
                have to be the same length as the atomic species and
                fractional_coords. Defaults to None for no properties.
        """
        super(MXene, self).__init__(
            lattice, species, coords, charge, validate_proximity,
            to_unit_cell, coords_are_cartesian, site_properties)
        self.out_dir = ""

    def get_similar_layer_atoms(self, z0=0.0, tol=0.005, axis=2, frac=True, frac_coords=None, ):
        """
        Get all same layer atoms by sites.
        获取z位置的整个面上的原子。

        Args:
            z0: z位置
            tol: 容忍因子
            axis: 默认z轴
            frac: 是否为 分数/笛卡尔坐标
            frac_coords: 分数/笛卡尔坐标

        Returns:
            atom_list: 原子位置
        """
        if frac_coords is None:
            if frac is True:
                coords = self.frac_coords if isinstance(self, Structure) else self
            else:
                coords = self.cart_coords if isinstance(self, Structure) else self
        else:
            coords = frac_coords
        coords = coords[:, axis]
        z0_atoms = [i for i, z in enumerate(coords) if abs(z - z0) < tol]
        return z0_atoms

    def __add__(self, other: Structure) -> "MXene":
        """
        Add the 2 MXenes with same Lattice.

        Args:
            other: (MXenes), MXenes.

        Returns:
            result: (MXenes), MXenes.

        """
        assert isinstance(other, Structure)
        assert np.all(other.lattice.matrix == self.lattice.matrix), "Just for same lattice."
        frac = np.concatenate((self.frac_coords, other.frac_coords), axis=0)
        spe = self.species
        spe.extend(other.species)
        return self.__class__(species=spe, coords=frac, lattice=self.lattice)

    def split_layer(self, ignore_index=None, n_cluster=None, tol=0.5, axis=2, method=None,
                    force_plane=True, reverse=True):
        """
        Split sites by distance or group (default z-axis).
        根据划分原子层。

        Args:
            tol: (float) tolerance distance for spilt.
            method:(str) default None. others: "agg", "k_means", None.
            n_cluster: (int) number of cluster for "agg", "k_means".
            axis:(int), z-axis
            ignore_index: jump some atoms.
            force_plane: change the single atom to nearest group.
            reverse: reverse the label.

        Returns:
            labels: (np.ndarray) with shape (n,).

        """
        array = self.cart_coords[:, axis]
        res_label = coarse_and_spilt_array_ignore_force_plane(array, ignore_index=ignore_index,
                                                              n_cluster=n_cluster, tol=tol,
                                                              method=method, force_plane=force_plane,
                                                              reverse=reverse)
        return res_label

    @staticmethod
    def check_terminal_sites_by_3_layer(array, up_down="up", tol=0.5):
        """
        Check the site.
        检查表面三层的堆垛方式类型。

        The up and down are different in some instance.

        Args:
            array: coords of 3 layer atoms.
            up_down: (str) up,down
            tol: (float) tol to split.

        Returns:
            name: (str)
        """

        reverse = True if up_down == "up" else False

        lab = ["fcc", "hcp", "top"]

        layer_label = coarse_and_spilt_array_ignore_force_plane(array[:, -1], tol=tol, reverse=reverse,
                                                                )

        layer1_values = np.max(layer_label)
        layer2_values = layer1_values - 1
        layer3_values = layer1_values - 2
        layer1_index = np.where(layer_label == layer1_values)
        layer2_index = np.where(layer_label == layer2_values)
        layer3_index = np.where(layer_label == layer3_values)
        coords1 = array[layer1_index, :2]
        coords2 = array[layer2_index, :2]
        coords3 = array[layer3_index, :2]
        offset12 = np.mean(coords1 - coords2, axis=0)

        coords_add_fcc = coords1 + offset12
        coords_add_hcp = coords2
        coords_add_top = coords1
        coords_add_fcc = np.sum(coords_add_fcc, axis=0)
        coords_add_hcp = np.sum(coords_add_hcp, axis=0)
        coords_add_top = np.sum(coords_add_top, axis=0)
        coords3 = np.sum(coords3, axis=0)
        r = np.array([coords_add_fcc, coords_add_hcp, coords_add_top])
        dis = np.sum(np.abs(r - coords3))
        s = np.argmin(dis)
        return lab[s]

    @staticmethod
    def get_next_layer_sites_xy(array, site_type: Union[None, str] = "fcc",
                                up_down="up", ):
        """
        按照目前原子层位点，以及堆垛方式，获取下一层原子位置。

        Args:
            array: (np.array), with shape (Nx2), site of all atoms
            site_type: (str), fcc, hcp, top
            up_down: (str),  up and down

        Returns:

        """

        reverse = True if up_down == "up" else False

        layer_label = coarse_and_spilt_array_ignore_force_plane(array[:, -1], tol=0.5, reverse=reverse,
                                                                )

        layer1_values = np.max(layer_label)
        layer2_values = layer1_values - 1
        layer1_index = np.where(layer_label == layer1_values)
        layer2_index = np.where(layer_label == layer2_values)
        coords1 = array[layer1_index, :2]
        coords2 = array[layer2_index, :2]
        offset12 = np.mean(coords1 - coords2, axis=0)

        if site_type == "fcc":

            coords_add = coords1 + offset12
        elif site_type == "hcp":
            coords_add = coords2

        elif site_type == "top":
            coords_add = coords1
        else:
            raise TypeError("Just accept 'top','fcc','hcp' terminal_site.")

        return coords_add

    def get_next_layer_sites(self, site_type: Union[None, str] = "fcc", ignore_index=None,
                             up_down="up", site_atom="O", array=None):
        """
        按照结构中原子层，以及堆垛方式，获取下一层原子位置。

        Args:
            array: np.ndarray, use array directly.
            ignore_index: (list of int), ignore index to split layer
            site_atom: (str), name of next layer atoms
            site_type: (str), fcc, hcp, top
            up_down: (str),  up and down

        Returns:

        """

        reverse = True if up_down == "up" else False
        coef = 1 if up_down == "up" else -1

        if array:
            layer_label = coarse_and_spilt_array_ignore_force_plane(array, ignore_index=ignore_index,
                                                                    reverse=reverse)
        else:

            layer_label = self.split_layer(ignore_index=ignore_index, n_cluster=None, tol=0.5, axis=2,
                                           method=None, reverse=reverse)

        layer1_values = np.min(layer_label)
        layer2_values = layer1_values + 1
        layer1_index = np.where(layer_label == layer1_values)
        layer2_index = np.where(layer_label == layer2_values)
        coords1 = self.cart_coords[layer1_index]
        coords2 = self.cart_coords[layer2_index]
        offset12 = np.mean(coords1 - coords2, axis=0)

        if isinstance(site_atom, str) and site_atom in self.tem_z_axis:
            terminal_z_axis = self.tem_z_axis[site_atom]
        elif isinstance(site_atom, str) and site_atom in self.tem_z_axis:
            terminal_z_axis = self.am_z_axis[site_atom]
        elif isinstance(site_atom, str):
            terminal_z_axis = 1.2
        else:
            terminal_z_axis = None

        if site_type == "fcc":
            coords_add = coords1 + offset12
        elif site_type == "hcp":
            coords_add = coords2
        elif site_type == "top":
            coords_add = coords1
        else:
            raise TypeError("Just accept 'top','fcc','hcp' terminal_site.")
        if terminal_z_axis is not None:
            coords_add[:, -1] = coef * terminal_z_axis + coords1[:, -1]

        return coords_add

    @classmethod
    def from_standard(cls, terminal_site: Union[None, str] = "fcc", doping: Union[None, str] = None,
                      terminal: Union[None, str] = "O",
                      base="Ti", carbide_nitride="C",
                      n_base=2, add_noise=True,
                      super_cell=(3, 3, 1), add_atoms=None, add_atoms_site=None,
                      coords_are_cartesian=True) -> "MXene":
        """
        Generate single atom doping MXenes.
        产生理想的单原子掺杂MXenes结构。

        Args:
            coords_are_cartesian: absorb_site are cartesian or not.
            add_noise: bool, add noise or not.
            doping: None, str, doping atoms.
            terminal_site: str, terminal atom type: ["fcc","bcc","hcp"].
            terminal: None, str, terminal atoms.
            base: str, base atoms.
            carbide_nitride: srt, "N" or "C" type.
            n_base:int, number of base atoms layer.
            super_cell: tuple with szie 3.
            add_atoms: str,list add atoms.
            add_atoms_site:: str,list add atoms site. fractional

        Returns:
            st: pymatgen.core.Strucutre

        """

        if terminal is None:
            terminal_site = None
        if terminal_site is None:
            terminal = None

        if isinstance(base, (tuple, list)):
            assert isinstance(carbide_nitride, (tuple, list)), "terminal and base should be same type, (str or list)."
            assert len(carbide_nitride) < len(base), "terminal should less than 1."
            base_list = list(base)
            carbide_nitride_list = list(carbide_nitride)
            name = {}
            bk = Counter(base_list)
            cnk = Counter(carbide_nitride_list)
            name.update(bk)
            name.update(cnk)

        else:
            assert n_base >= 2
            base_list = [base] * n_base
            carbide_nitride_list = [carbide_nitride] * (n_base - 1)
            name = {}
            name.update({base: n_base})
            name.update({carbide_nitride: n_base - 1})
        if terminal is not None:
            name.update({terminal: 2})
        else:
            name.update({"bare": 2})

        mx_list = list(chain(*zip(base_list[:-1], carbide_nitride_list))) + [base_list[-1]]

        n_layer = len(mx_list)
        d = [[0.00000, 0.00000], [0.33333, 0.66666], [0.66666, 0.33333]]

        z_axis = 2.5 * (n_layer - 3) + 25
        lattice = Lattice.from_parameters(3.0, 3.0, z_axis, 90.0000, 90.0000, 120)
        z_setp = 1.25 / z_axis

        ter_axis = cls.tem_z_axis
        if terminal in ter_axis:
            oz_step = ter_axis[terminal] / z_axis
        else:
            oz_step = 1.2 / z_axis

        fracs = []
        for i in np.arange(0, n_layer):
            index = -round((n_layer - 1) / 2) + i
            fracs.append(d[abs(index % 3)] + [index * z_setp + 0.5])
        if terminal_site is None:
            pass
        else:
            if terminal_site == "fcc":
                sam1, sam2 = 2, -3
            elif terminal_site == "hcp":
                sam1, sam2 = 1, -2
            else:  # top
                sam1, sam2 = 0, -1

            start = copy.copy(fracs[sam1])
            start[-1] = -oz_step - round((n_layer - 1) / 2) * z_setp + 0.5
            fracs.insert(0, start)
            end = copy.copy(fracs[sam2])
            end[-1] = oz_step + round((n_layer - 1) / 2) * z_setp + 0.5
            fracs.append(end)
            mx_list.insert(0, terminal)
            mx_list.append(terminal)

        fracs = np.array(fracs)

        st = cls(lattice=lattice, species=mx_list, coords=fracs)

        st = st.get_sorted_structure(reverse=True)

        if super_cell is not None:
            st.make_supercell(super_cell)

        if add_noise:
            st_frac_coords = st.frac_coords
            st_frac_coords = np.array(st_frac_coords) + np.random.random(st_frac_coords.shape) * 0.001
            st = cls(lattice=st.lattice, species=st.species, coords=st_frac_coords)

        if doping:
            nm_tm = "NM" if doping in cls.nm_list else "TM"

            if terminal_site is None:
                if nm_tm == "NM":
                    raise TypeError("'NM' should be with 'terminal'")
                z0 = fracs[-1][-1]
            else:
                if nm_tm == "TM":
                    z0 = fracs[-2][-1]
                else:
                    z0 = fracs[-1][-1]
            sam_atoms = cls.get_similar_layer_atoms(st, z0=z0, frac=True)
            xys = st.frac_coords[sam_atoms][:, :2]
            xy = np.mean(xys, axis=0)
            xyd = np.sum(abs(xys - xy), axis=1)
            index = int(np.argmin(xyd))
            site = st.frac_coords[sam_atoms[index]]
            st.remove_sites([sam_atoms[index], ])
            st.append(doping, site, coords_are_cartesian=False)
        if add_atoms is not None:
            if isinstance(add_atoms, (list, tuple)):
                for n, s in zip(add_atoms, add_atoms_site):
                    st.append(n, coords=s, coords_are_cartesian=coords_are_cartesian)
            else:
                st.append(add_atoms, coords=add_atoms_site, coords_are_cartesian=coords_are_cartesian)

        return cls.from_sites(sites=list(st.sites))

    def get_structure_message(self):
        """获取结果信息"""
        return structure_message(self)

    def add_absorb(self, center=44, site_type: Union[None, str] = "fcc",
                   add_noise=True, up_down="up",
                   site_name="S0", equivalent="ini_opt", absorb="H", absorb_site=None,
                   ignore_index=None, coords_are_cartesian=True, offset_z=0) -> "MXene":
        """
        添加吸附原子。

        Args:
            up_down: (str), up or down.
            coords_are_cartesian: absorb_site are cartesian or not.
            center: (int), center index.
            site_type: (str), terminal atom type: ["fcc","bcc","hcp","center"].
            absorb: (list,str) name or list of name of absorb atoms.
            absorb_site: (None, list of list), np.ndarray with (Nx3) size.
            add_noise:(bool), add noise to site.
            site_name: (str), equivalent site name
            equivalent: (str), class name of equivalent site
            ignore_index: (list) index of ignore index.
            offset_z: (float), offset z cartesian, manually, not affect by up_down.

        Returns:

        """
        # just for up_down="up"

        if absorb_site is None:
            # 自动确定位置
            if site_type == "center":
                coef = 1 if up_down == "up" else -1
                if isinstance(absorb, str) and absorb in self.tem_z_axis:
                    offset_z = self.tem_z_axis[absorb] * coef + offset_z
                elif isinstance(absorb, str) and absorb in self.tem_z_axis:
                    offset_z = self.am_z_axis[absorb] * coef + offset_z
                elif isinstance(absorb, str):
                    offset_z = 1.2 * coef + offset_z
                else:
                    offset_z = offset_z

                absorb_site = self.cart_coords[center] + np.array([0, 0, offset_z])

                self.append(absorb, coords=absorb_site, coords_are_cartesian=coords_are_cartesian)

            else:

                sites = self.get_next_layer_sites(site_type=site_type, ignore_index=ignore_index,
                                                  up_down=up_down, site_atom=absorb)

                st2 = copy.deepcopy(self)
                [st2.append(species=i, coords=j, coords_are_cartesian=True) for i, j in
                 zip([absorb] * sites.shape[0], list(sites))]

                points_and_distance_to_m0 = get_plane_neighbors_to_center(st2, center, neighbors_name=absorb,
                                                                          plane=False,
                                                                          ignore_index=ignore_index, r=7.0)

                if equivalent == "fin_opt":
                    sel = 0
                else:
                    sel = 1

                si = int(site_name[-1])
                if len(list(points_and_distance_to_m0[0].keys())) == 1:
                    si += 1  # jump center site

                index = list(points_and_distance_to_m0[si].keys())[sel]

                absorb_site = st2.cart_coords[index] + np.array([0, 0, offset_z])

                self.append(absorb, coords=absorb_site, coords_are_cartesian=coords_are_cartesian)

        else:
            if isinstance(absorb_site, (list, tuple)):
                for n, s in zip(absorb, absorb_site):
                    self.append(n, coords=s, coords_are_cartesian=coords_are_cartesian)
            else:
                self.append(absorb, coords=absorb_site, coords_are_cartesian=coords_are_cartesian)

        if add_noise:
            st_frac_coords = self.frac_coords
            st_frac_coords = np.array(st_frac_coords) + np.random.random(st_frac_coords.shape) * 0.001
            st = self.__class__(lattice=self.lattice, species=self.species, coords=st_frac_coords)
            return st
        else:
            return self

    def get_disk(self, disk='.', ignore_index=None, equivalent="ini_opt", site_name="S0", nm_tm="TM",
                 absorb=None, doping=None, terminal=None, carbide_nitride=None,
                 ) -> pathlib.Path:
        """Just for single doping, single。"""
        absorb_ = absorb
        doping_ = doping
        terminal_ = terminal
        carbide_nitride_ = carbide_nitride
        names = [site.specie.name for site in self]
        labels = self.split_layer(ignore_index=ignore_index)
        end = int(max(labels) + 1)
        start = int(max(min(labels), 0))
        layer_name = []
        for i in range(start, end):
            index = np.where(labels == i)[0]
            layer_name.append(np.array(names)[index])

        doping_add = []

        layer_name2 = []
        for name_l in layer_name:
            names_d = Counter(name_l)
            if len(names_d) == 1:
                if list(names_d.values())[0] > 2:
                    layer_name2.append(list(names_d.keys())[0])
                else:
                    doping_add.append(list(names_d.keys())[0])
            else:
                cm = names_d.most_common(1)[0][0]
                layer_name2.append(cm)
                for i in names_d.keys():
                    if i != cm:
                        doping_add.append(i)
        absorb = []
        doping = []
        terminal = []
        base = []
        add_atoms = []
        carbide_nitride = []
        for i in layer_name2:
            if i in self.tem_list:
                terminal.append(i)
            if i in self.cn:
                carbide_nitride.append(i)
            if i in self.tm_list:
                base.append(i)

        for i in doping_add:
            if nm_tm == "TM" and i in self.tm_list:
                doping.append(i)
            elif nm_tm == "NM" and i in self.nm_list[1:]:
                doping.append(i)
            elif i in self.am_list:
                absorb.append(i)
            else:
                add_atoms.append(i)

        assert len(doping) <= 1
        assert len(absorb) <= 1
        assert len(list(set(terminal))) <= 1

        doping = None if len(doping) == 0 else doping[0]
        absorb = None if len(absorb) == 0 else absorb[0]

        terminal = None if len(terminal) == 0 else terminal[0]

        absorb = absorb_ if absorb_ else absorb
        doping = doping_ if doping_ else doping
        terminal = terminal_ if terminal_ else terminal
        carbide_nitride = carbide_nitride_ if carbide_nitride_ else carbide_nitride

        self.out_dir = make_disk(disk, terminal, base, carbide_nitride, n_base=None, doping=doping, absorb=absorb,
                                 equivalent=equivalent,
                                 site_name=site_name, add_atoms=add_atoms)
        return self.out_dir

    # def add_random(self,random_state=0):

    def get_interp2d(self, up_down="up"):
        """Not for 'top' terminals."""
        reverse = True if up_down == "up" else False
        st = copy.deepcopy(self)
        st.make_supercell((3, 3, 1))
        labels = st.split_layer(ignore_index=None, n_cluster=None, tol=0.5, axis=2, method=None,
                                force_plane=True, reverse=reverse)
        marks = []
        for i in range(int(max(labels))):
            marki = labels == i
            if np.sum(marki == True) > 0:
                marks.append(marki)
            if len(marks) == 2:
                break

        mark1, mark2 = marks

        mark = mark1 | mark2
        data = st.cart_coords[mark]
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        # assert np.std(z) > 0.02, "Not for 'top' terminals."
        iterp = interp2d_nearest(x, y, z)
        return iterp

    def extend(self, species: Sequence[CompositionLike], coords, coords_are_cartesian: bool = False, ):
        """Extend, see more MXenes.append."""

        if isinstance(species, Sequence):
            pass
        else:
            species = [species] * coords.shape(-1, 1)

        for s, c in (species, coords):
            self.append(s, c, coords_are_cartesian=coords_are_cartesian)

    @staticmethod
    def _reform_frac(frac_coords):

        for i in range(frac_coords.shape[0]):
            if frac_coords[:, 0][i] > 1:
                frac_coords[:, 0][i] -= 1
            elif frac_coords[:, 0][i] < 0:
                frac_coords[:, 0][i] += 1
            else:
                pass

        for i in range(frac_coords.shape[0]):
            if frac_coords[:, 1][i] > 1:
                frac_coords[:, 1][i] -= 1
            elif frac_coords[:, 1][i] < 0:
                frac_coords[:, 1][i] += 1
            else:
                pass

        for i in range(frac_coords.shape[0]):
            if frac_coords[:, 2][i] > 1:
                frac_coords[:, 2][i] -= 1
            elif frac_coords[:, 2][i] < 0:
                frac_coords[:, 2][i] += 1
            else:
                pass
        return frac_coords

    def reform_structure(self):
        """把结构原子坐标统一到初始晶胞。"""
        frac_coords = copy.copy(self.frac_coords)
        frac_coords = self._reform_frac(frac_coords)
        return self.__class__(self.lattice, self.species, frac_coords, coords_are_cartesian=False)

    def add_face_random(self, number=100, random_state=0, add_atom="H", debug=False, up_down="up", perturb_base=0,
                        offset_z=1.0, alpha=0.5) -> Union["MXene", List["MXene"]]:
        """
        随机添加原子。

        Args:
            number: (str), 添加数量
            random_state: (int), 随机种子。
            add_atom: (str), 添加原子名称
            debug: (bool), 调试，若调试，所有添加原子在一个结构中展示。
            up_down: (str), up and down
            perturb_base:  (str),扰动初始结构。
            offset_z: (str), Z高度。
            alpha: (str), 移动Z波动系数。

        Returns:
            result: (MXenes,list of MXenes)
        """
        rdm = check_random_state(random_state)
        f_interp = self.get_interp2d(up_down=up_down)
        a = self.lattice.a
        b = self.lattice.b
        nn = int(number ** 0.5)

        x = np.linspace(0, a, nn + 1, endpoint=False)
        y = np.linspace(0, b, nn + 1, endpoint=False)
        x_mesh, y_mesh = np.meshgrid(x, y)
        z = f_interp(x, y, meshed=False)
        z = z.ravel()
        x_mesh, y_mesh = x_mesh.ravel(), y_mesh.ravel()
        if up_down == "up":
            car = np.vstack((x_mesh, y_mesh, z + offset_z + alpha * rdm.random(z.shape))).T
        else:
            car = np.vstack((x_mesh, y_mesh, z - offset_z + alpha * rdm.random(z.shape))).T
        add_atom = [add_atom] * car.shape[0]
        if debug:
            print("Just for dedug.")
            st = self.__class__(self.lattice, add_atom, coords=car, coords_are_cartesian=True)
            st = st.reform_structure()
            st = self + st
            return st
        else:
            if perturb_base > 0:
                sts = [self.copy().perturb(distance=0.001, min_distance=0.0001) for _ in range(car.shape[0])]
            else:
                sts = [self.copy() for _ in range(car.shape[0])]
            [si.append(add_atom[0], cari, coords_are_cartesian=True) for si, cari in zip(sts, car)]
            return sts

    def add_face_random_z(self, x, y, number_each=1, random_state=0, add_atom="H", debug=False, up_down="up",
                          perturb_base=0, offset_z=1.0, alpha=1.0, method='random') -> Union["MXene", List["MXene"]]:
        """
        随机添加z原子，x，y使用输入值。

        Args:
            x: (np.array), x
            y: (np.array), y
            number_each: (int)，每对x，y对应多少z样本。
            random_state: (int), 随机种子。
            add_atom: (str), 添加原子名称
            debug: (bool), 调试，若调试，所有添加原子在一个结构中展示。
            up_down: (str), up and down
            perturb_base:  (str),扰动初始结构。
            offset_z: (str), Z高度。
            alpha: (str), 移动Z波动系数。
            method: (str), 'random','uniform','linspace'

        Returns:
            result: (MXenes,list of MXenes)
        """
        rdm = check_random_state(random_state)
        f_interp = self.get_interp2d(up_down=up_down)

        x_mesh, y_mesh = x, y
        z = f_interp(x, y, meshed=True)
        z = z.ravel()
        x_mesh, y_mesh = x_mesh.ravel(), y_mesh.ravel()
        if number_each > 1:
            x_mesh = np.repeat(x_mesh, number_each)
            y_mesh = np.repeat(y_mesh, number_each)
            z = np.repeat(z, number_each)
        if method == "random":
            add = alpha * rdm.random(z.shape)
        elif method == "uniform":
            add = alpha * rdm.uniform(low=0.0, high=1.0, size=z.shape)
        else:
            add = np.concatenate([alpha * np.linspace(0, 1, number_each)] * int(z.shape[0] / number_each))

        if up_down == "up":
            car = np.vstack((x_mesh, y_mesh, z + offset_z + add)).T
        else:
            car = np.vstack((x_mesh, y_mesh, z - offset_z - add)).T
        add_atom = [add_atom] * car.shape[0]
        if debug:
            print("Just for dedug.")
            st = self.__class__(self.lattice, add_atom, coords=car, coords_are_cartesian=True)
            st = st.reform_structure()
            st = self + st
            return st
        else:
            if perturb_base > 0:
                sts = [self.copy().perturb(distance=0.001, min_distance=0.0001) for _ in range(car.shape[0])]
            else:
                sts = [self.copy() for _ in range(car.shape[0])]
            [si.append(add_atom[0], cari, coords_are_cartesian=True) for si, cari in zip(sts, car)]
            return sts

    def show(self):
        """
        展示画图。
        """
        atomss = aaa.get_atoms(self)
        view(atomss)

    def non_equivalent_site(self, center=44, ignore_index=None, base_m=None, terminal=None):
        """
        Just for MXenes with terminals， 获取16个等效位置（fcc，hcp适用。）

        Args:
            center:(int),重心原子位置。
            ignore_index:(list),跳过干扰原子。
            base_m:(str),基础原子名称。
            terminal:(str),终端原子名称。

        Returns:
            res:(np.array)
        """

        label = self.split_layer(ignore_index=ignore_index, n_cluster=None, tol=0.5, axis=2, method=None,
                                 force_plane=True, reverse=True)
        marks = []
        for i in range(int(max(label))):
            marki = label == i
            if np.sum(marki == True) > 0:
                marks.append(marki)
            if len(marks) == 2:
                break

        mark1, mark2 = marks

        if terminal is None:
            terminal = self.species[np.where(mark1)[0][0]].name
        if base_m is None:
            base_m = self.species[np.where(mark2)[0][0]].name

        mark = mark1 | mark2

        m_dict = get_plane_neighbors_to_center(self, center=center, neighbors_name=[base_m],
                                               ignore_index=ignore_index,
                                               r=5.5, plane=True)

        sel = 1
        if len(m_dict[0]) == 1:
            sel = 2
        index = list(m_dict[sel].keys())
        m_dict1 = m_dict[sel]
        assert len(index) >= 2

        site = []
        # offsets =[]
        real_cart = self.cart_coords

        for i, j in itertools.combinations(index, 2):
            if i != j:
                for off2, off3 in [(np.array([0, 0, 0]), np.array([0, 0, 0])), (np.array([0, 0, 0]), m_dict1[j][1]),
                                   (m_dict1[i][1], np.array([0, 0, 0])), (m_dict1[i][1], m_dict1[j][1]), ]:
                    d1 = self.get_distance(i, j, jimage=off3 - off2)
                    d2 = self.get_distance(center, i, jimage=off2)
                    d3 = self.get_distance(center, j, jimage=off3)

                    if np.std([d1, d2, d3]) < 0.35 and d1 > 0.3:
                        site = [center, i, j]

                        real_cart[i] = real_cart[i] + np.dot(off2, self.lattice.matrix)
                        real_cart[j] = real_cart[j] + np.dot(off3, self.lattice.matrix)
                        break
                break

        if len(site) == 0:
            raise ValueError("Cant find the site.")

        core = np.mean(real_cart[site, :2], axis=0)
        top2_0 = real_cart[:, :2][mark]

        core_index_mark = np.argmin(np.sum(np.abs(top2_0 - core), axis=1))
        core_index = np.where(mark)[0][core_index_mark]

        o_dict = get_plane_neighbors_to_center(self, center=core_index, neighbors_name=[terminal],
                                               ignore_index=ignore_index,
                                               r=5, plane=True, )
        sel = 0
        if len(o_dict[0]) == 1:
            sel = 1

        o_index = list(o_dict[sel].keys())
        assert len(o_index) == 3

        for i in o_index:
            off = o_dict[sel][i][1]
            real_cart[i] = real_cart[i] + np.dot(off, self.lattice.matrix)

        real_cart2 = real_cart[:, :2]

        site.append(core_index)
        site.extend(o_index)

        # 6 bond site
        rem = []
        for i, j in itertools.combinations(site, 2):
            if i != j:
                d = np.sum((real_cart2[i] - real_cart2[j]) ** 2) ** 0.5
                if d < 2.3:
                    rem.append((i, j))

        bond = [np.mean(real_cart2[remi, :], axis=0) for remi in rem]

        # 4 metal atoms site
        metal = [real_cart2[i] for i in site[:4]]

        # 6 "3-spilt" site
        sp3 = []
        for i, j in itertools.combinations(site[:3], 2):
            end_site1 = real_cart2[i]
            end_site2 = real_cart2[j]
            x = np.linspace(end_site1[0], end_site2[0], 3, endpoint=False)[1:]
            y = np.linspace(end_site1[1], end_site2[1], 3, endpoint=False)[1:]
            data = np.vstack((x, y)).T
            sp3.extend(data)

        # check "3-spilt" site with O sites
        for i in o_index:
            dis = np.sum((sp3 - real_cart2[i]) ** 2, axis=1) ** 0.5
            o_i = np.argmin(dis)
            if np.min(dis) < 0.4:
                sp3[o_i] = real_cart2[i]

        data_all = np.array(bond + metal + sp3)
        size = data_all.shape[0]
        if size != 16:
            warnings.warn(f"Can't find the 16 non-equvalent sites, just find {size}.", UserWarning)
        return np.array(data_all)

    def add_face_random_z_16_site(self, center=44, ignore_index=None, base_m=None, terminal=None, number_each=1,
                                  random_state=0, add_atom="H", debug=False, up_down="up", perturb_base=0,
                                  offset_z=1.0, alpha=1.0, method="random") -> Union["MXene", List["MXene"]]:
        """
        随机添加z原子，x，y使用输入值。

        Args:
            center:(int),重心原子位置。
            ignore_index:(list),跳过干扰原子。
            base_m:(str),基础原子名称。
            terminal:(str),终端原子名称。
            number_each: (int),每对x，y对应多少z样本。
            random_state: (int), 随机种子。
            add_atom: (str), 添加原子名称
            debug: (bool), 调试，若调试，所有添加原子在一个结构中展示。
            up_down: (str), up and down
            perturb_base:  (str),扰动初始结构。
            offset_z: (str), Z高度。
            alpha: (str), 移动Z波动系数。

        Returns:
            result: (MXenes, list of MNXenes)
        """
        xy = self.non_equivalent_site(center=center, ignore_index=ignore_index, base_m=base_m, terminal=terminal)
        x = xy[:, 0]
        y = xy[:, 1]
        res = self.add_face_random_z(x=x, y=y, number_each=number_each, random_state=random_state, add_atom=add_atom,
                                     debug=debug,
                                     up_down=up_down, perturb_base=perturb_base,
                                     offset_z=offset_z, alpha=alpha, method=method)
        return res

    def apply_operation_no_lattice(self, symmop: SymmOp, fractional: bool = False):
        """
        Apply a symmetry operation to the structure and return the new
        structure. The lattice is operated by the rotation matrix only.
        Coords are operated in full and then transformed to the new lattice.

        Args:
            symmop (SymmOp): Symmetry operation to apply.
            fractional (bool): Whether the symmetry operation is applied in
                fractional space. Defaults to False, i.e., symmetry operation
                is applied in cartesian coordinates.
        """
        if not fractional:
            # self._lattice = Lattice([symmop.apply_rotation_only(row) for row in self._lattice.matrix])

            def operate_site(site):
                new_cart = symmop.operate(site.coords)
                new_frac = self._lattice.get_fractional_coords(new_cart)
                return PeriodicSite(
                    site.species,
                    new_frac,
                    self._lattice,
                    properties=site.properties,
                    skip_checks=True,
                )

        else:
            # new_latt = np.dot(symmop.rotation_matrix, self._lattice.matrix)
            # self._lattice = Lattice(new_latt)

            def operate_site(site):
                return PeriodicSite(
                    site.species,
                    symmop.operate(site.frac_coords),
                    self._lattice,
                    properties=site.properties,
                    skip_checks=True,
                )

        self._sites = [operate_site(s) for s in self._sites]

    def to_ase_atoms(self):

        return aaa.get_atoms(self)
