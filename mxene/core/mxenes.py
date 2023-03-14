# -*- coding: utf-8 -*-

# @Time  : 2022/10/2 13:20
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

import copy
import itertools
import warnings
from collections import Counter
from itertools import chain
from typing import Union, Sequence, List, Optional

import numpy as np
from pymatgen.core import Structure, Lattice, SymmOp, PeriodicSite, Element

from mxene.core.functions import coarse_and_spilt_array_ignore_force_plane, \
    get_plane_neighbors_to_center, Interp2dNearest, coarse_cluster_array, check_random_state
from mxene.utility.typing import CompositionLike, ArrayLike, ListTuple


class MXene(Structure):
    """MXene object.

    ### Bellowing is the common function ###

    1. Create:
        from_standard: Create a virtual structure from the category.

        from_file: Read POSCAR and other files.

    2. Change:
        insert: Insert atoms.

        append: Add atoms.

        substitute:  Replace with a single atom.

        replace:  replace with the functional group.

        remove_species:  Removes all atoms of a species.

        remove_sites:  Delete a site.

        sort:  Resort.

        rotate_sites:  Rotation.

        perturb:  Random disturbance.

        make_supercell:  supercell.

        add_absorb: Adds an adsorbed atom.

        add_face_random_*****: Add atoms in random batch.

        extrusion: add move affect by center atom.

        tuning_layer_space: ture the layer sapce.

    3. Read structural information:
        get_structure_message : get bond, face message.

    Examples:

    >>> mx = MXene.from_standard()
    >>> print(mx)


    Full Formula (Ti18 C9 O18)

    Reduced Formula: Ti2CO2

    abc   :   9.000000   9.000000  25.000000

    angles:  90.000000  90.000000 120.000000

    pbc   :       True       True       True

    Sites (45)

      #  SP           a         b     c

    ---  ----  --------  --------  ----

      0  C     0         0         0.5

    ...

    >>> #mx = MXene.from_file("POSCAR")
    >>> #print(mx)

    """

    _predefined_nm_list = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I", "Te", "At"]
    _predefined_tm_list = ["Sc", "Y", "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Mn",
                           "Re", "Fe", "Ru", "Os", "Co", "Rh", "Ir", "Ni", "Pd", "Pt", "Cu", "Ag", "Au", "Zn", "Cd"]

    _predefined_dop_list = _predefined_nm_list + _predefined_tm_list

    _predefined_am_list = ["Al", "Ca", "Li", "Mg", "Na", "K", "Zn", "H"]
    _predefined_tem_list = ["O", "F", "Cl", "S", None]
    _predefined_cn = ["C", "N"]
    _predefined_bm = ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Sc"]
    _predefined_bm_cell_ab = {"Ti": 3.0, "Zr": 3.2, "Hf": 3.15, "V": 2.93, "Nb": 3.0, "Ta": 3.05, "Cr": 2.83,
                              "Mo": 2.9, "W": 2.87, "Sc": 3.3}
    _predefined_tem_z_axis = {"O": 1.0, "F": 0.9, "Cl": 1.2}
    _predefined_am_z_axis = {"K": 2.1, "Li": 1.3, "H": 0.9, "Ca": 1.6, "Na": 1.6, "Mg": 1.4, "Al": 1.1, "Zn": 1.2}

    def __init__(self,
                 lattice: Union[ArrayLike, Lattice],
                 species: Sequence[CompositionLike],
                 coords: Sequence[ArrayLike],
                 charge: float = None,
                 validate_proximity: bool = False,
                 to_unit_cell: bool = False,
                 coords_are_cartesian: bool = False,
                 site_properties: dict = None):
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
        self.mark_label = None

    @staticmethod
    def _get_real_terminal_site(base, terminal, cn, layer=2):
        bt = f"{terminal}-{base}-{cn}"
        hcp_calc = {
            2: [
                # Cl
                "Cl-Ta-C",
                # F
                "F-Ta-C",
                "F-Ta-N", "F-Hf-N",
                # S
                "S-Cr-C", "S-Mo-C", "S-Sc-C", "S-W-C",
                "S-Cr-N", "S-Mo-N", "S-Nb-N", "S-Ta-N", "S-W-N",
                # O
                "O-Cr-C", "O-Mo-C", "O-Sc-C", "O-W-C",
                "O-Mo-N", "O-Nb-N", "O-Ta-N", "O-W-N", "O-V-N",
                # Se
                "Se-Hf-C", "Se-Mo-C", "Se-W-C",
                "Se-Hf-C", "Se-Mo-C", "Se-W-C",
            ],
            3: [
                # Cl
                "Cl-Mo-C", "Cl-Nb-C", "Cl-Ta-C", "Cl-V-C",
                # F
                "F-Nb-C", "F-Ta-C", "F-V-C",
                # S
                "S-Cr-C", "S-Mo-C", "S-Sc-C", "S-W-C",
                # O
                "O-Cr-C", "O-Mo-C", "O-W-C",
            ],
            4: [
            ]
        }

        top_calc = {
            2: [
                "S-W-N",
            ],
            3: [
                "Cl-W-C",
                "F-Mo-C", "F-W-C",
            ],
            4: [
            ]
        }

        if layer not in hcp_calc:
            warnings.warn("The definition of MXenes is with [2, or 3, or 4] metal layers,"
                          "please check you input.")
        if bt in hcp_calc[layer]:
            return "hcp"
        elif bt in top_calc[layer]:
            return "top"
        else:
            return "fcc"

    def check_single_cell(self, array=False):
        if isinstance(array, np.ndarray):
            l = len(array)
        else:
            l = len(self.atomic_numbers)
        if l < 12:
            single = True
        else:
            single = False
        return single

    def get_similar_layer_atoms(self, z0=0.0, tol=0.005, axis=2, frac=True, array=None):
        """
        Get all same layer atoms by z0 site.

        Args:
            z0: float, site.
            tol: float, tolerance factor.
            axis: int, default the z axis.
            frac: bool, whether it is fractional/Cartesian coordinates.
            array: np.ndarray, use this coordinates directly.

        Returns:
            atom_list: np.ndarray, atom index.
        """
        if array is None:
            if frac is True:
                array = self.frac_coords if isinstance(self, Structure) else self
            else:
                array = self.cart_coords if isinstance(self, Structure) else self
        else:
            array = array
        array = array[:, axis]
        z0_atoms = np.where(np.abs(array - z0) < tol)[0]
        return z0_atoms

    def __add__(self, other: Structure) -> "MXene":
        """
        Add 2 MXenes with same lattice.

        Args:
            other: (MXenes), added MXenes.

        Returns:
            result: (MXenes), new MXenes.

        """
        assert isinstance(other, Structure)
        assert np.all(other.lattice.matrix == self.lattice.matrix), "Just for same lattice."
        frac = np.concatenate((self.frac_coords, other.frac_coords), axis=0)
        spe = self.species
        spe.extend(other.species)
        return self.__class__(species=spe, coords=frac, lattice=self.lattice)

    def is_mirror_sym(self, tol=0.3, axis=2) -> bool:
        try:

            label = self.split_layer(ignore_index=None, tol=tol, force_plane=False, reverse=True,
                                     force_finite=False, array=None, axis=axis)

            label2 = np.max(label) - label

            atm = np.array(self.atomic_numbers)

            for i in range(np.max(label)):
                atm1 = atm[label == i]
                atm2 = atm[label2 == i]
                if not np.all(np.equal(atm1, atm2)):
                    return False

            return True
        except BaseException as e:
            return False

    def split_layer(self, ignore_index=None, tol=0.5, force_plane=True, reverse=True,
                    force_finite=False, array=None, axis=2):
        """
        Split layer.

        'ignore_index' means the atoms is not used to calculated.
        'force_finite' and 'force_plane' are two method to settle the un-grouped atoms.

        See Also:
            :func:`mxene.core.functions.coarse_and_spilt_array_ignore_force_plane`

        (1). For easily compartmentalized array, use the default parameters.

        (2). Used 'ignore_index' and 'force_finite' to drop the add (absorb) atom to make the left atoms to be grouped,
             change the 'tol' to check the result.
        (3). Used 'ignore_index'=None and 'force_plane' to make the 'out-of-station' doped atom to be grouped.
             the 'tol' could be appropriate smaller (less than interlayer spacing).
        (4). For absorb + doped system, 'ignore_index', 'force_finite' and 'force_plane' could be used together,
             change the 'tol' to check the result.But, for large structural deformations array.
             This function is not effective always.
        (5). If all the parameter are failed, please generate the input array by hand or tune
             the value in array.

        Args:
            array:
            axis: (int), axis to be grouped.
            tol: (float), tolerance distance for spilt.(less than interlayer spacing)
            ignore_index: (int, np.ndarray), jump 'out-of-station' atoms.
            force_plane: (bool), change the single atom to the nearest group.
            force_finite: (bool), force to change the index finite.
            reverse: (bool), reverse the label.

        Returns:
            labels: (np.ndarray) with shape (n,).

        """
        if array is None:
            array = self.cart_coords[:, axis]

        if array.shape[0] > 12:
            layer_label = coarse_and_spilt_array_ignore_force_plane(array, ignore_index=ignore_index,
                                                                    tol=tol, force_plane=force_plane,
                                                                    reverse=reverse, force_finite=force_finite)
        else:
            warnings.warn("It could be single cell, and try to spilt layer."
                          "It could result to error. We suggest make super cell first before splitting layer.")
            layer_label = coarse_cluster_array(array, tol=tol, reverse=reverse)

        return layer_label

    def check_terminal_sites_by_3_layer(self, up_down="up", tol=0.3, array=None, **kwargs):
        """

        Check the type of stacking type of the surface three layers.

        The up and down are different in some instance.

        Args:
            array: (np.ndarray), coords atoms.
            up_down: (str), up,down
            tol: (float), tol to split.

        Returns:
            name: (str)
        """
        reverse = True if up_down == "up" else False

        lab = ["hcp", "top"]

        if isinstance(array, np.ndarray):
            assert array.ndim == 2 and array.shape[1] == 3
        else:
            array = self.cart_coords

        layer_label = self.split_layer(array=array[:, -1], tol=tol, reverse=reverse,
                                       **kwargs)

        single = self.check_single_cell(array[:, -1])

        # sss = np.concatenate((array,layer_label.reshape(-1,1)),axis=1)

        ll = []
        i = 0
        while len(ll) < 4:
            if not single:
                if np.sum(layer_label == i) >= 3:
                    ll.append(i)
            else:
                ll.append(i)
            i += 1

        layer1_index = np.where(layer_label == ll[0])[0]
        layer2_index = np.where(layer_label == ll[1])[0]
        layer3_index = np.where(layer_label == ll[2])[0]
        # layer4_index = np.where(layer_label == ll[3])[0]
        coords1 = array[layer1_index, :2]
        coords2 = array[layer2_index, :2]
        coords3 = array[layer3_index, :2]
        # coords4 = array[layer4_index, :2]

        coords_add_top = coords2
        coords_add_hcp = coords3
        # coords_add_fcc = coords4

        rhcp = np.mean(coords_add_hcp, axis=0)
        rtop = np.mean(coords_add_top, axis=0)
        coords1m = np.mean(coords1, axis=0)

        diff = np.abs(np.array([rhcp, rtop]) - coords1m.reshape(1, -1))
        dis = np.sum(diff ** 2, axis=-1) ** 0.5
        s = np.argmin(dis)

        if dis[s] > 0.05:
            return "fcc"
        else:
            return lab[s]

    def get_next_layer_sites_xy(self, array, site_type: Union[None, str] = "fcc",
                                up_down="up", tol=0.5, **kwargs):
        """
        According to the current atomic layer site, and stacking method,
        to obtain the atomic position (x, y) of the next layer.

        Args:
            tol: float, tolerance
            array: (np.array), with shape (N,3), site of all atoms
            site_type: (str), fcc, hcp, top
            up_down: (str),  up and down

        Returns:
            array: (np.array),array.
        """

        reverse = True if up_down == "up" else False

        layer_label = self.split_layer(array=array[:, -1], tol=tol, reverse=reverse, **kwargs)

        single = self.check_single_cell(array)

        # sss = np.concatenate((array,layer_label.reshape(-1,1)),axis=1)

        ll = []
        i = 0
        while len(ll) < 3:
            if not single:
                if np.sum(layer_label == i) >= 3:
                    ll.append(i)
            else:
                ll.append(i)
            i += 1

        layer1_index = np.where(layer_label == ll[0])[0]
        layer2_index = np.where(layer_label == ll[1])[0]
        # layer3_index = np.where(layer_label == ll[2])[0]
        coords1 = array[layer1_index, :2]
        coords2 = array[layer2_index, :2]
        # coords3 = array[layer3_index, :2]

        c = coords1[0]
        moves = coords2 - c
        d = np.sum(moves ** 2, axis=1)
        d_index = np.argmin(d)
        move = moves[d_index]

        coords_add_top = coords1
        coords_add_hcp = coords2
        coords_add_fcc = coords1 + move  # could be negative

        if site_type == "fcc":
            coords_add = coords_add_fcc
        elif site_type == "hcp":
            coords_add = coords_add_hcp
        elif site_type == "top":
            coords_add = coords_add_top
        else:
            raise TypeError("Just accept 'top','fcc','hcp' terminal_site.")

        return coords_add

    def get_next_layer_sites(self, site_type: Union[None, str] = "fcc", ignore_index=None,
                             force_plane=False, force_finite=True, terminal_z_axis=1.2,
                             up_down="up", site_atom="O", array=None, tol=0.5) -> np.ndarray:
        """
        According to the atomic layer in the structure and the stacking method,
        the position of the next layer of atoms is obtained.

        1. For single-atom doping system.
            It is recommended that the `tol` be set to smaller and force_plane.
            Example: tol=0.1, force_plane=False, force_finite=True.
        2. For complex systems.
            It is recommended to redesign the reformed_array so that it can be properly separated.

        Args:
            terminal_z_axis:(float) Add site of z axis.
            tol: (float), tolerance.
            array: (np.ndarray), use this array for spilt directly. (1D).
            ignore_index: (list of int), ignore index to split layer.
            site_atom: (str), name of next layer atoms.
            site_type: (str), fcc, hcp, top.
            up_down: (str),  up and down.
            force_plane: (bool), change the single atom to the nearest group.
            force_finite: (bool), force to change the index finite.

        Returns:
            coord: (np.ndarray),

        """

        reverse = True if up_down == "up" else False
        coef = 1 if up_down == "up" else -1

        if isinstance(array, np.ndarray):
            assert array.ndim == 2 and array.shape[1] == 3
        else:
            array = self.cart_coords

        layer_label = self.split_layer(array=array[:, -1], tol=tol, reverse=reverse,
                                       ignore_index=ignore_index,
                                       force_plane=force_plane, force_finite=force_finite,
                                       )

        single = self.check_single_cell(array[:, -1])

        # sss = np.concatenate((array,layer_label.reshape(-1,1)),axis=1)

        ll = []
        i = 0
        while len(ll) < 3:
            if not single:
                if np.sum(layer_label == i) >= 3:
                    ll.append(i)
            else:
                ll.append(i)
            i += 1

        layer1_index = np.where(layer_label == ll[0])[0]
        layer2_index = np.where(layer_label == ll[1])[0]
        # layer3_index = np.where(layer_label == ll[2])[0]
        coords1 = array[layer1_index, :2]
        coords2 = array[layer2_index, :2]
        # coords3 = array[layer3_index, :2]

        c = coords1[0]
        moves = coords2 - c
        d = np.sum(moves ** 2, axis=1)
        d_index = np.argmin(d)
        move = moves[d_index]

        coords_add_top = coords1
        coords_add_hcp = coords2
        coords_add_fcc = coords1 + move  # could be negative

        if site_type == "fcc":
            coords_add = coords_add_fcc
        elif site_type == "hcp":
            coords_add = coords_add_hcp
        elif site_type == "top":
            coords_add = coords_add_top
        else:
            raise TypeError("Just accept 'top','fcc','hcp' terminal_site.")

        if isinstance(site_atom, str) and site_atom in self._predefined_tem_z_axis:
            terminal_z_axis = self._predefined_tem_z_axis[site_atom]
        elif isinstance(site_atom, str) and site_atom in self._predefined_tem_z_axis:
            terminal_z_axis = self._predefined_am_z_axis[site_atom]
        elif isinstance(site_atom, str):
            terminal_z_axis = 1.2
        else:
            pass
            # terminal_z_axis = 1.2
        coords_top = array[layer1_index, -1]
        zz = np.mean(coords_top)
        coords = np.concatenate((coords_add, np.full(coords_add.shape[0], zz).reshape(-1, 1)), axis=1)

        if terminal_z_axis is not None:
            coords[:, -1] = coef * terminal_z_axis + coords[:, -1]

        return coords

    def change_top_layer_site(self, terminal_site_to: Union[Sequence, str] = "fcc",
                              up_down="down_up",
                              atom_type: Union[None, Sequence, str] = None, tol=0.4,
                              offset_z=0.5) -> "MXene":
        """
        Change the top layer site.

        the ranks is [down, up]

        such as: terminal_site_to = ["fcc","hcp"] means down layer is fcc and up is hcp.

        Args:
            terminal_site_to: str, "fcc","hcp","top","fcc-hcp" ...
            up_down: str, change the ``up`` layer or ``down`` layer or both.
            atom_type: str, new atom type. such as "F","S".
            tol: float, tolerance.

        Returns:
            MXene: MXene, New structure.

        """

        if isinstance(terminal_site_to, str) and "-" in terminal_site_to:
            terminal_site_to = terminal_site_to.split("-")
        if isinstance(terminal_site_to, str):
            terminal_site = [terminal_site_to, terminal_site_to]
        else:
            terminal_site = terminal_site_to

        if isinstance(atom_type, str) and "-" in atom_type:
            atom_type = atom_type.split("-")
        if isinstance(atom_type, str) or atom_type is None:
            atom_type = [atom_type, atom_type]
        else:
            pass

        assert len(atom_type) == len(terminal_site) == 2

        label = self.split_layer(reverse=True)
        max_n = np.max(label)
        min_n = np.min(label)

        max_index = label == max_n
        min_index = label == min_n

        index = ~(max_index | min_index)
        array_select = self.cart_coords[index]

        cart_up = self.get_next_layer_sites_xy(array_select, site_type=terminal_site[1], up_down="up", tol=tol)
        cart_down = self.get_next_layer_sites_xy(array_select, site_type=terminal_site[0], up_down="down", tol=tol)

        coords = self.cart_coords

        if isinstance(offset_z, (float, int)):
            offset_z1 = offset_z * {"hcp": 1, "fcc": 1, "top": 2}[terminal_site[0]]
            offset_z2 = offset_z * {"hcp": 1, "fcc": 1, "top": 2}[terminal_site[1]]
        else:
            offset_z1, offset_z2 = offset_z

        if "up" in up_down:
            coords[min_index, :2] = cart_up
            coords[min_index, -1] = coords[min_index, -1] + abs(offset_z2)
        if "down" in up_down:
            coords[max_index, :2] = cart_down
            coords[max_index, -1] = coords[max_index, -1] - abs(offset_z1)

        sp = np.array(self.species)

        if "up" in up_down:
            if atom_type[0] is not None:
                sp[max_index] = Element(atom_type[0])
        if "down" in up_down:
            if atom_type[1] is not None:
                sp[min_index] = Element(atom_type[1])

        return self.__class__(lattice=self.lattice,
                              species=[i for i in sp],
                              coords=coords,
                              coords_are_cartesian=True, )

    def add_atom_layer(self, site_atom: Union[str, List[str]] = "Ag",
                       site_type: Union[None, str] = "fcc", ignore_index=None,
                       force_plane=False, force_finite=True, terminal_z_axis=1.2,
                       up_down="up_and_down",
                       array=None, tol=0.5):
        """
        Add next layer atoms.

        1. For single-atom doping system.
            It is recommended that the `tol` be set to smaller and force_plane.
            Example: tol=0.1, force_plane=False, force_finite=True.
        2. For complex systems.
            It is recommended to redesign the reformed_array so that it can be properly separated.

        Args:
            terminal_z_axis:(float) Add site of z axis.
            tol: (float), tolerance.
            array: (np.ndarray), use this array for spilt directly. (1D).
            ignore_index: (list of int), ignore index to split layer.
            site_atom: (str), name of next layer atoms.
            site_type: (str), fcc, hcp, top.
            up_down: (str),  up and down.
            force_plane: (bool), change the single atom to the nearest group.
            force_finite: (bool), force to change the index finite.
        """
        arr = []

        if "up" in up_down:
            coords_add1 = self.get_next_layer_sites(site_type=site_type, ignore_index=ignore_index,
                                                    force_plane=force_plane, force_finite=force_finite,
                                                    terminal_z_axis=terminal_z_axis,
                                                    up_down="up", site_atom=site_atom, array=array,
                                                    tol=tol)
            arr.append(coords_add1)
        if "down" in up_down:
            coords_add2 = self.get_next_layer_sites(site_type=site_type, ignore_index=ignore_index,
                                                    force_plane=force_plane, force_finite=force_finite,
                                                    terminal_z_axis=terminal_z_axis,
                                                    up_down="down", site_atom=site_atom, array=array,
                                                    tol=tol)
            arr.append(coords_add2)

        if len(arr) == 2:
            arr = np.concatenate(arr, axis=0)
        else:
            arr = arr[0]

        if isinstance(site_atom, str):
            site_atom = [site_atom] * (arr.shape[0])

        if isinstance(site_atom, (list, tuple)):
            for n, s in zip(site_atom, arr):
                self.append(n, coords=s, coords_are_cartesian=True)

    @classmethod
    def from_standard(cls, terminal_site: Union[None, str, Sequence] = "fcc",
                      doping: Union[None, str] = None, terminal: Union[None, str] = "O",
                      base: Union[Sequence, str] = "Ti", carbide_nitride: Union[Sequence, str] = "C",
                      n_base: int = 2, add_noise: bool = False,
                      super_cell: tuple = (3, 3, 1), add_atoms=None, add_atoms_site=None,
                      coords_are_cartesian=True, lattice: Union[Lattice, tuple, list] = None,
                      layer_space=1.25, terminal_z_axis=None, random_state=None, random_factor=0.001) -> "MXene":
        """
        Generate ideal single atom doping MXenes.

        Examples:
            >>> # Mo2CO2-Ti
            >>> mx = MXene.from_standard(base="Mo",doping="Ti",n_base=3,terminal="O",terminal_site="hcp")

        Args:
            random_factor: float, random factor.
            random_state: (random.RandomState, int),
            terminal_z_axis: float, z axis for terminal.
            layer_space: float, space of layer.
            lattice: (tuple of float,Lattice),  6 float or Lattice
            coords_are_cartesian: absorb_site are cartesian or not.
            add_noise: bool, add noise or not.
            doping: None, str, doping atoms.
            terminal_site: str, terminal atom type: ["fcc","bcc","hcp"].
            terminal: None, str, terminal atoms.
            base: str,tuple,list. base atoms.
            carbide_nitride: srt, "N" or "C" type.
            n_base:int, number of base atoms layer.
            super_cell: tuple with size 3.
            add_atoms: str,list add atoms.
            add_atoms_site:: str,list add atoms site. fractional

        Returns:
            st: pymatgen.core.Structure

        """

        if terminal is None:
            terminal_site = None
        if terminal_site is None:
            terminal = None

        if isinstance(base, (tuple, list)):
            # n_base is not used.
            if isinstance(carbide_nitride, str):
                carbide_nitride = [carbide_nitride] * (len(base) - 1)
            assert isinstance(carbide_nitride, (tuple, list)), "terminal and base should be same type, (str or list)."
            assert len(carbide_nitride) < len(base), "number of carbide_nitride should less than base."
            base_list = list(base)
            carbide_nitride_list = list(carbide_nitride)
            name = {}
            bk = Counter(base_list)
            cnk = Counter(carbide_nitride_list)
            name.update(bk)
            name.update(cnk)

        else:
            assert n_base >= 2, "Just for base atom, more than 2 such as Ti2CO2"
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

        if isinstance(lattice, (tuple, list)):
            lattice = Lattice.from_parameters(*lattice)
        elif isinstance(lattice, Lattice):
            pass
        else:  # default # for atom not predefined
            cc = 2.5 * (n_layer - 3) + 25

            ab = [cls._predefined_bm_cell_ab[i] if i in cls._predefined_bm_cell_ab else 3.0 for i in base_list]
            ab = sum(ab) / len(ab)

            lattice = Lattice.from_parameters(ab, ab, cc, 90.0000, 90.0000, 120)

        z_axis = lattice.c

        z_step = layer_space / lattice.c

        ter_axis = cls._predefined_tem_z_axis
        if terminal in ter_axis:
            if terminal_z_axis:
                oz_step = terminal_z_axis / z_axis
            else:
                oz_step = ter_axis[terminal] / z_axis
        else:
            oz_step = 1.2 / z_axis  # for atom not predefined

        fracs = []
        for i in np.arange(0, n_layer):
            index = -round((n_layer - 1) / 2) + i
            fracs.append(d[abs(index % 3)] + [index * z_step + 0.5])
        if terminal_site is None:
            pass
        else:
            pre_sites = {"fcc": [2, -3], "hcp": [1, -2], "top": [0, -1]}

            if isinstance(terminal_site, str) and "-" in terminal_site:
                terminal_site = terminal_site.split("-")

            if isinstance(terminal_site, (list, tuple)):
                assert len(terminal_site) == 2
                sam1 = pre_sites[terminal_site[0]][0]
                sam2 = pre_sites[terminal_site[1]][1]

            elif terminal_site == "fcc":
                sam1, sam2 = 2, -3
            elif terminal_site == "hcp":
                sam1, sam2 = 1, -2
            elif terminal_site == "top":
                sam1, sam2 = 0, -1

            elif terminal_site == "auto":
                assert len(set(carbide_nitride_list)) == 1, 'auto just accept one type of in ["C","N"], rather both.'
                # experiment site. should check.
                sam = []
                for i in range(2):
                    tm = base_list[[0, -1][i]]

                    tps = cls._get_real_terminal_site(tm, terminal, carbide_nitride_list[0], layer=len(base_list))

                    sam.append(pre_sites[tps][i])
                sam1, sam2 = sam[0], sam[1]
            else:  # top
                raise NotImplementedError("please str name such as 'fcc','hcp','top'.")

            start = copy.copy(fracs[sam1])
            start[-1] = -oz_step - round((n_layer - 1) / 2) * z_step + 0.5
            fracs.insert(0, start)
            end = copy.copy(fracs[sam2])
            end[-1] = oz_step + round((n_layer - 1) / 2) * z_step + 0.5
            fracs.append(end)

            mx_list.insert(0, terminal)
            mx_list.append(terminal)

        fracs = np.array(fracs)

        st = cls(lattice=lattice, species=mx_list, coords=fracs)

        st = st.get_sorted_structure(key=lambda x: x.specie.Z)

        if super_cell is not None or super_cell == (1, 1, 1):
            st.make_supercell(super_cell)

        if add_noise:
            st = st.add_noise(random_state=random_state, random_factor=random_factor)

        if doping:

            nm_tm = "NM" if doping in cls._predefined_nm_list else "TM"

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
            xyd = np.sum(abs(xys - xy), axis=1)  # find the center location.
            index = int(np.argmin(xyd))
            site = st.frac_coords[sam_atoms[index]]
            st.remove_sites([sam_atoms[index], ])
            st.append(doping, site, coords_are_cartesian=False)
            st.num_doping = len(st) - 1
        if add_atoms is not None:
            if isinstance(add_atoms, (list, tuple)):
                for n, s in zip(add_atoms, add_atoms_site):
                    st.append(n, coords=s, coords_are_cartesian=coords_are_cartesian)
            else:
                st.append(add_atoms, coords=add_atoms_site, coords_are_cartesian=coords_are_cartesian)

        return cls.from_sites(sites=list(st.sites))

    def pure_add_doping(self, doping, up_down="up"):
        """From pure to center face doping"""
        nm_tm = "NM" if doping in self._predefined_nm_list else "TM"
        assert up_down in ["up", "down"]
        reverse = True if up_down == "up" else False

        label = self.split_layer(ignore_index=None, tol=0.5, axis=2,
                                 force_plane=True, reverse=reverse, force_finite=True)

        if nm_tm == "NM":
            index = np.argmin(label)
        else:
            index = np.where(label == (np.min(label) + 1))[0][0]

        assert isinstance(index, np.int64)

        z0 = self.frac_coords[index][-1]
        sam_atoms = self.get_similar_layer_atoms(z0=z0, frac=True)
        xys = self.frac_coords[sam_atoms][:, :2]
        xy = np.mean(xys, axis=0)
        xyd = np.sum(abs(xys - xy), axis=1)  # find the center location.
        index = int(np.argmin(xyd))

        doping_index = sam_atoms[index]
        old_atom = self.species[sam_atoms[index]].name

        site = self.frac_coords[doping_index]
        self.remove_sites([doping_index, ])

        if doping is None:
            doping = old_atom
        self.append(doping, site, coords_are_cartesian=False)
        self.num_doping = len(self) - 1
        return doping_index, old_atom

    def get_structure_message(self):
        """Obtaining bond, face Information"""
        from mxene.extract.structure_extractor import structure_message
        return structure_message(self)

    @classmethod
    def from_structure(cls, structrue: Structure):
        """From structure to mxenes object."""
        return cls(lattice=structrue.lattice,
                   species=structrue.species,
                   coords=structrue.frac_coords,
                   charge=structrue.charge,
                   validate_proximity=False,
                   to_unit_cell=False,
                   coords_are_cartesian=False,
                   site_properties=structrue.site_properties)

    def extrusion(self, center=-1, factors=(0.01, 0.001), affect_radius=6.0,
                  center_move=0, center_move_is_frac=False, axis=2):
        """Get structure defects for doping system.
         Just for the one axis is vertical with the 2 other axis.(cubic, square, orthogonality).

        Args:
            center: (int), center index.
            factors: (float), move factor for nearst, second nearst, ... atoms  .
            affect_radius: (float), radius.
            center_move: (float), center atoms move in z axis or 3 axis.
            center_move_is_frac: (bool), center move is fractional or not.
            axis: (int), axis, the same size as center_move.
         """

        st2 = self.copy()

        if axis is None:
            axis = (0, 1, 2)

        (center_indices, points_indices, offset_vectors, distances) = \
            self.get_neighbor_list(affect_radius, numerical_tol=1e-3,
                                   sites=[self.sites[center]], exclude_self=True)
        index = np.sum(np.abs(offset_vectors), axis=1) == 0
        points_indices = points_indices[index]
        distances = distances[index]
        label = coarse_cluster_array(distances, tol=0.02)

        zer = np.where(label == 0)
        if distances[zer[0]] < 0.001:
            s = 1
        else:
            s = 0

        mes = []
        for i in range(len(factors)):
            second = np.where(label == (i + s))
            point_se = points_indices[second]
            mes.append(point_se)

        center_cart = self.cart_coords[center]
        if center_move_is_frac:
            center_move = center_move * self.lattice.abc[axis]  # to Cartesian coordinates.
        center_cart[axis] = center_cart[axis] + center_move
        for mesi, faci in zip(mes, factors):
            neighbor_cart = self.cart_coords[mesi]
            base = neighbor_cart - center_cart
            base_norm = base / (np.sum(base ** 2, axis=1).reshape(-1, 1) ** 0.5)
            neighbor_cart2 = neighbor_cart + faci * base_norm
            for mesii, neii in zip(mesi, neighbor_cart2):
                st2.replace(mesii, species=self.species[mesii], coords=neii, coords_are_cartesian=True)
        st2.replace(center, species=self.species[center], coords=center_cart, coords_are_cartesian=True)

        return st2

    @property
    def doped(self):
        """Check it is doped or not."""
        label = coarse_cluster_array(self.cart_coords[:, 2], tol=0.1)
        label[label == 0] = max(label) + 1

        typess = set(self.atomic_numbers)
        label2 = np.array(list(self.atomic_numbers))
        index = [label2 == ti for ti in typess]
        for i, ti in enumerate(index):
            label2[ti] = i

        label = label * label2 + 0.01 * (label - label2)
        label_dict = Counter(label)
        numbers = list(label_dict.values())

        if len(set(numbers)) != 1:
            return True
        else:
            return False

    def add_noise(self, random_state=None, random_factor=0.001, axis_factor=(1, 1, 1)) -> "MXene":
        """The same with self.perturb but change in new object."""
        rdm = check_random_state(random_state)
        st_frac_coords = self.frac_coords
        axis_factor = np.array(list(axis_factor)).reshape(1, -1)
        st_frac_coords = np.array(st_frac_coords) + (
                rdm.random(st_frac_coords.shape) - 0.5) * random_factor * axis_factor
        st = self.__class__(lattice=self.lattice, species=self.species, coords=st_frac_coords)
        return st

    def append_noise(self, random_state=None, random_factor=0.001, axis_factor=(1, 1, 1)) -> None:
        """The same with self.perturb."""
        rdm = check_random_state(random_state)
        st_frac_coords = self.frac_coords
        axis_factor = np.array(list(axis_factor)).reshape(1, -1)

        ve_frac_coords = rdm.random(st_frac_coords.shape) * random_factor * axis_factor
        for i in range(len(self._sites)):
            self.translate_sites([i], ve_frac_coords[i], frac_coords=True)

    def tuning_layer_space(self, random_state=None, max_turing=0.05):
        """For the structure is centered in z axis !!!"""
        rdm = check_random_state(random_state)

        layer_label = self.split_layer(tol=0.02, force_plane=False, reverse=False).astype(int)

        num = int(max(layer_label) + 1)

        frac_coords_z = self.frac_coords[:, 2].copy()

        frac_coords_z_05 = frac_coords_z - 0.5

        c0 = layer_label[np.where(frac_coords_z_05 < 0, frac_coords_z_05, -np.inf).argmax()]
        c1 = layer_label[np.where(frac_coords_z_05 >= 0, frac_coords_z_05, np.inf).argmin()]
        if abs(c1 - c0) == 1:
            dis = np.mean(frac_coords_z[layer_label == c1]) - np.mean(frac_coords_z[layer_label == c0])
            diss = 0.5 - np.mean(frac_coords_z[layer_label == c0])
            p = diss / dis
        else:
            p = 0.0

        p = 1.0 if p > 0.99 else p
        p = 0.0 if p < 0.01 else p

        _, index = np.unique(layer_label, return_index=True)

        space = frac_coords_z[index[1:]] - frac_coords_z[index[:-1]]

        fc = (rdm.random(num - 1) - 0.5) * max_turing
        dis_space = space * (1 + fc)
        dis_sum = np.cumsum(dis_space)
        dis_sum = np.concatenate((np.array(0.0).reshape(1, ), dis_sum))
        dis_space = np.concatenate((np.array(0.0).reshape(1, ), dis_space))

        dis_sum2 = dis_sum - dis_sum[c1] + p * dis_space[c1] + 0.5

        new_frac_coords_z = dis_sum2[layer_label]

        frac_coords = np.copy(self.frac_coords)
        frac_coords[:, 2] = new_frac_coords_z

        st = self.__class__(lattice=self.lattice, species=self.species, coords=frac_coords,
                            coords_are_cartesian=False)
        return st

    def surface_pure_center(self, up_down="up", atom_type="O"):
        """Get the surface center index for no-doped mxene."""
        if self.doped is True:
            warnings.warn("Just for no doped mxene.")

        frac_coords = self.frac_coords
        frac_coords_xy = frac_coords[:, (0, 1)]
        frac_coords_z = frac_coords[:, 2]
        frac_coords_xy_temp = np.zeros_like(frac_coords_xy)

        index = np.array([True if i.specie.name == atom_type else False for i in self.sites])

        labels = coarse_cluster_array(frac_coords_z, tol=0.02).astype(float)
        labels[~index] = np.nan

        s = max(labels) if up_down == "up" else min(labels)

        index2 = labels == s

        frac_coords_xy_temp[index2] = frac_coords_xy[index2]

        l = np.sum((frac_coords_xy_temp - 0.5) ** 2, axis=1) ** 0.5

        center = np.argmin(l)

        return center

    def add_absorb(self, center: Union[int, None] = -1, site_type: Union[None, str] = "top",
                   offset_z: float = 0,
                   add_noise: bool = True, up_down: str = "up", site_name: str = "S0",
                   equivalent: str = "ini_opt",
                   absorb: str = "H", absorb_site: Union[np.ndarray, ListTuple] = None,
                   array: np.ndarray = None,
                   ignore_index: Union[np.ndarray, tuple, int, List] = None,
                   coords_are_cartesian: bool = True, tol: float = 0.3, random_state=None,
                   random_factor: float = 0.001, pure: bool = False,
                   ) -> "MXene":
        """
        Add adsorbent atoms. Change the original data!!!

        Args:
            random_state: random_state seed.
            tol: group layer tolerance.
            random_factor: (float), factor for random size.
            up_down: (str), up or down.
            array: np.ndarray, use this array for spilt directly. (1D)
            coords_are_cartesian: absorb_site are cartesian or not.
            center: (int,None), center index.
            site_type: (str), terminal atom type: ["fcc","bcc","hcp","center"]. such as 'top' for H and 'fcc' for metals.
            absorb: (list,str) name or list of name of absorb atoms.
            absorb_site: (None, np.ndarray), np.ndarray with (Nx3) or np.ndarray s in list with (3,) size .
            add_noise:(bool), add noise to site.
            site_name: (str), equivalent site name
            equivalent: (str), class name of equivalent site
            ignore_index: (list) index of ignore index.
            offset_z: (float), offset z cartesian, manually, not affected by up_down.
            pure: (bool),  base structures without doping.

        Returns:

        """

        if ignore_index is not None:
            ignore_index = list(range(len(self)))[ignore_index]

        if pure and center is None:
            center = self.surface_pure_center(up_down=up_down)

        center = list(range(len(self)))[center]

        rdm = check_random_state(random_state)

        if absorb_site is None:
            # 自动确定位置
            if site_type == "center":
                coef = 1 if up_down == "up" else -1
                if isinstance(absorb, str) and absorb in self._predefined_tem_z_axis:
                    offset_z = (self._predefined_tem_z_axis[absorb] + offset_z) * coef
                elif isinstance(absorb, str) and absorb in self._predefined_tem_z_axis:
                    offset_z = (self._predefined_am_z_axis[absorb] + offset_z) * coef
                elif isinstance(absorb, str):
                    offset_z = coef * (offset_z + 1.0)
                else:
                    offset_z = offset_z * coef

                absorb_site = self.cart_coords[center] + np.array([0, 0, offset_z]) + rdm.random(
                    size=3) * 50 * random_factor

                self.append(absorb, coords=absorb_site, coords_are_cartesian=coords_are_cartesian)

            else:

                sites = self.get_next_layer_sites(site_type=site_type, ignore_index=ignore_index,
                                                  force_finite=True, force_plane=True,
                                                  up_down=up_down, site_atom=absorb, tol=tol,
                                                  array=array)

                st2 = copy.deepcopy(self)
                [st2.append(species=i, coords=j, coords_are_cartesian=True) for i, j in
                 zip([absorb] * sites.shape[0], list(sites))]

                points_and_distance_to_m0 = get_plane_neighbors_to_center(st2, center_index=center,
                                                                          neighbors_name=absorb,
                                                                          plane=False,
                                                                          ignore_index=ignore_index,
                                                                          r=7.0)

                if equivalent == "fin_opt":
                    sel = 0
                else:
                    sel = 1

                si = int(site_name[-1])  # just for S0,S1,S2
                if len(list(points_and_distance_to_m0[0].keys())) == 1:
                    si += 1  # jump out to center site

                index = list(points_and_distance_to_m0[si].keys())[sel]

                absorb_site = st2.cart_coords[index] + np.array([0, 0, offset_z]) + rdm.random(
                    size=3) * 50 * random_factor

                self.append(absorb, coords=absorb_site, coords_are_cartesian=coords_are_cartesian)

        else:
            if isinstance(absorb_site, np.ndarray) and len(absorb_site) >= 6:  # multi atoms
                absorb_site = [i for i in absorb_site]
            if isinstance(absorb_site, (list, tuple)):
                if isinstance(absorb, str):
                    absorb = [absorb] * len(absorb_site)
                for n, s in zip(absorb, absorb_site):  # keep match
                    self.append(n, coords=s, coords_are_cartesian=coords_are_cartesian)
            else:
                self.append(absorb, coords=absorb_site, coords_are_cartesian=coords_are_cartesian)  # single

        if isinstance(absorb, str):
            self.num_absorb = len(self) - 1
        else:
            self.num_absorb = list(range(len(self) - len(absorb), len(self)))

        if add_noise:
            self.append_noise(random_state=random_state, random_factor=random_factor)

        return self

    def get_disk(self, disk='.', site_name: Optional[str] = "ter_site-dop_site-add_site",
                 equ_name="opt", ignore_index=None, add_atoms=None, tol=0.4, terminal_site=None,
                 absorb=None, doping=None, terminal=None, carbide_nitride=None, force_plane=True,
                 old_type=False, nm_tm: Optional[str] = None, super_cell: Optional[tuple] = None,
                 force_finite=False,
                 ):
        """Just for single doping, single absorb, single type terminal.
        for some name, the code could judge the parameter automatically, and for others, you should offer it.
        such as site_name="S0", equ_name="ini_opt", nm_tm="TM",

        **(1)不吸附** 6层 8层

        MXenes -> 基底层数 -> 基底名称 -> 负载物 -> 搀杂物 -> pure/pure_static

        例子 6层  （兼容以前）

        MXenes -> M2C -> Ti2NO2 -> no_add -> no_doping -> pure/pure_static

        >>> self.get_disk(disk='.', site_name=None, equ_name="pure",
        ...          add_atoms=None, absorb=None, doping=None, old_type=True,)

        MXenes -> M2N -> Ti2NO2 -> no_add -> Mo        -> pure/pure_static

        例子 8层

        MXenes -> 基底层数 -> 基底名称 -> 负载物 -> 搀杂物 -> 吸附物     -> 位点 -> 标签

        >>> self.get_disk(disk='.', site_name="up", equ_name="pure", terminal_site="fcc",
        ...          add_atoms=None, absorb=None, doping=None, old_type=False)

        MXenes -> M2C -> Ti2NO2 -> no_add -> no_doping -> no_absorb -> fcc -> pure_opt

        **(2)吸附** 8 层

        MXenes -> 基底层数  -> 基底名称 -> 负载物 -> 搀杂物 -> 吸附物  -> 位点 -> 标签

        例子

        MXenes -> M2N  -> Ti2NO2      -> no_add -> no_doping -> H/add_H   -> up-S0 -> ini_opt

        MXenes -> M3N2 -> TiNCrNTi-O2 -> Hf     -> C         -> Li        -> down-S0  -> ini_opt

        MXenes -> M3N2 -> TiNCrNTi-O2 -> Hf     -> C         -> no_absorb -> down-S0  -> ini_opt

        >>> self.get_disk(disk='.', site_name="up", equ_name="ini_opt",
        ...          add_atoms=None, absorb="Li", doping=None)

        **(3)NEB** 8 层

        MXenes -> 基底层数  -> 基底名称 -> 负载物 -> 搀杂物 -> 吸附物 -> 等效位点 -> 路径名称

        例子

        MXenes -> M2N -> Ti2NO2 -> no_add -> no_doping -> H -> S0-S1 -> 00/01/01/03/04/ini/fin/...

        >>> self.get_disk(disk='.', site_name="up-S0-S1", equ_name="ini_opt", nm_tm="TM",
        ...          add_atoms=None, absorb="Li", doping=None)

        """
        absorb_ = absorb
        doping_ = doping
        add_atoms_ = add_atoms
        terminal_ = terminal
        carbide_nitride_ = carbide_nitride
        _ = nm_tm

        names = [site.specie.name for site in self]
        if len(names) <= 12:
            mx = self.copy()
            mx.make_supercell((2, 2, 1))
            names = [site.specie.name for site in mx]
        else:
            mx = self
        labels = mx.split_layer(ignore_index=ignore_index, tol=tol, force_plane=force_plane,
                                reverse=False,  # to make rank is corresponding with down-up
                                force_finite=force_finite)
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
            if i in self._predefined_tem_list:
                terminal.append(i)
            if i in self._predefined_cn:
                carbide_nitride.append(i)
            if i in self._predefined_tm_list:
                base.append(i)

        for i in doping_add:
            # just for doping
            if i in base:
                base.append(i)
            elif i in self._predefined_tm_list and i not in base:
                doping.append(i)
            elif i in self._predefined_nm_list[1:] and i not in base:
                doping.append(i)
            elif i in self._predefined_am_list:
                absorb.append(i)
            else:
                add_atoms.append(i)

        if absorb_:
            absorb = absorb_
        else:
            if absorb is None:
                pass
            else:
                assert len(absorb) <= 1
                absorb = None if len(absorb) == 0 else absorb[0]

        if doping_:
            doping = doping_
        else:
            if doping is None:
                pass
            else:
                assert len(doping) <= 1
                doping = None if len(doping) == 0 else doping[0]

        if add_atoms_:
            add_atoms = add_atoms_
        else:
            if add_atoms is None:
                pass
            else:
                add_atoms = None if len(add_atoms) == 0 else add_atoms

        if len(list(terminal)) == 0:
            terminal = terminal_ if terminal_ else None
        if len(list(set(terminal))) <= 1:
            terminal = terminal_ if terminal_ else terminal[0]
        else:
            warnings.warn("This is just for same terminal. Different terminals could be error path.")

            terminal = terminal_ if terminal_ else terminal

        carbide_nitride = carbide_nitride_ if carbide_nitride_ else carbide_nitride

        if isinstance(base, (tuple, list)):
            assert isinstance(carbide_nitride, (tuple, list)), "terminal and base should be same type, (str or list)."
            assert len(carbide_nitride) < len(base), "carbide_nitride should less than base."

        from mxene.organize.disk import make_disk

        if terminal_site == "auto" and terminal is not None:
            try:
                terminal_site_up = mx.check_terminal_sites_by_3_layer(up_down="up", tol=0.4)
                terminal_site_down = mx.check_terminal_sites_by_3_layer(up_down="down", tol=0.4)
                if terminal_site_up != terminal_site_down:
                    terminal_site = f"{terminal_site_down}-{terminal_site_up}"
                else:
                    terminal_site = terminal_site_down
            except BaseException as e:
                terminal_site = None

        self.out_dir = make_disk(disk, terminal, base, carbide_nitride,
                                 n_base=None, doping=doping, site_name=site_name,
                                 absorb=absorb, equ_name=equ_name, base_num_cls=None,
                                 add_atoms=add_atoms, super_cell=super_cell,
                                 terminal_site=terminal_site,
                                 old_type=old_type, )
        return self.out_dir

    def relax_base(self, random_state=None, lr=1, extrusion=True, strain=True, tun_layer=True, add_noise=True,
                   kwarg_extrusion=None, kwarg_strain=None, ):
        """
        Change the structure.

        Args:
            random_state: randomstate.
            lr: bool, learning rate.
            extrusion: bool, extrusion.
            strain: bool, strain.
            tun_layer: bool, tune the layer spacing.
            add_noise: bool, add noise.
            kwarg_extrusion: dict, the extrusion parameter.
            kwarg_strain: dict, the extrusion parameter.

        Returns:
            mxi: MXene, Changed structure.
        """

        if kwarg_extrusion is None:
            kwarg_extrusion = {}

        if kwarg_strain is None:
            kwarg_strain = {}

        rdm = check_random_state(random_state)
        mxi = self.copy()

        if extrusion:
            if "center_move" in kwarg_extrusion:
                kwarg_extrusion["center_move"] = kwarg_extrusion["center_move"] * lr
            mxi = mxi.extrusion(**kwarg_extrusion)

        if strain:
            if "strain" in kwarg_strain and kwarg_strain["strain"] != (0, 0, 0):
                kwarg_strain["strain"] = [i * lr for i in kwarg_strain["strain"]]
            mxi = mxi.adjust_lattice(**kwarg_strain)

        if tun_layer:
            mxi = mxi.tuning_layer_space(random_state=rdm, max_turing=0.05 * lr)

        if add_noise:
            mxi = mxi.add_noise(random_state=rdm, random_factor=0.005, )

        return mxi

    # def relax_by_predictor(self,predictor):

    def adjust_lattice(self, strain: ArrayLike = None):
        """
        Apply a strain to the lattice.

        Args:
            strain (float or list): Amount of strain to apply. Can be a float,
                or a sequence of 3 numbers. E.g., 0.01 means all lattice
                vectors are increased by 1%. This is equivalent to calling
                modify_lattice with a lattice with lattice parameters that
                are 1% larger.
        """
        if strain is None:
            strain = [0, 0, 0]
        if isinstance(strain, tuple):
            strain = list(strain)

        s = (1 + np.array(strain)) * np.eye(3)
        self.lattice = Lattice(np.dot(self._lattice.matrix.T, s).T)
        return self

    def get_interp2d(self, up_down="up"):
        """Not for 'top' terminals."""
        reverse = True if up_down == "up" else False
        st = copy.deepcopy(self)
        st.make_supercell((3, 3, 1))
        labels = st.split_layer(ignore_index=None, tol=0.5, axis=2,
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
        # assert np.std(z) > 0.02,
        iterp = Interp2dNearest(x, y, z)
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
        """The atomic coordinates of the structure are unified to the initial cell."""
        frac_coords = copy.copy(self.frac_coords)
        frac_coords = self._reform_frac(frac_coords)
        return self.__class__(self.lattice, self.species, frac_coords, coords_are_cartesian=False)

    def add_face_random(self, number=100, random_state=0, random_xy=True, add_atom="H", debug=False,
                        up_down="up", perturb_base=0,
                        offset_z=1.0, alpha=0.5) -> Union["MXene", List["MXene"]]:
        """
        Add atoms at randomly.

        Args:
            number: (str), samples.
            random_state: (int), randomstate.
            random_xy:(bool),Whether x and y are random.
            add_atom: (str), Add atom name
            debug: (bool), Debug. If debugged, all added atoms are displayed in one structure.
            up_down: (str), up and down
            perturb_base:  (str),Perturbed the initial structure.
            offset_z: (str),  height along Z
            alpha: (str), Shift coefficient in of surface.

        Returns:
            result: (MXenes,list of MXenes)
        """
        rdm = check_random_state(random_state)
        f_interp = self.get_interp2d(up_down=up_down)
        a = self.lattice.a
        b = self.lattice.b
        nn = int(number ** 0.5)

        if random_xy:
            x = rdm.randint(0, a, nn)
            y = rdm.randint(0, b, nn)
        else:
            x = np.linspace(0, a, nn + 1, endpoint=False)
            y = np.linspace(0, b, nn + 1, endpoint=False)

        x_mesh, y_mesh = np.meshgrid(x, y)
        z = f_interp(x, y, mesh=False)
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
        Add z atoms randomly, x, y using the input values.

        Args:
            x: np.ndarray, x values of point.
            y: np.ndarray, x values of point.
            method:(str), method to add.
            number_each: (str), samples.
            random_state: (int), randomstate.
            add_atom: (str), Add atom name
            debug: (bool), Debug. If debugged, all added atoms are displayed in one structure.
            up_down: (str), up and down
            perturb_base:  (str),Perturbed the initial structure.
            offset_z: (str),  height along Z
            alpha: (str), Shift coefficient in of surface.

        Returns:
            result: (MXenes,list of MXenes)
        """
        rdm = check_random_state(random_state)
        f_interp = self.get_interp2d(up_down=up_down)

        x_mesh, y_mesh = x, y
        z = f_interp(x, y, mesh=True)
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

    def non_equivalent_site(self, center=44, ignore_index=None, base_m=None, terminal=None):
        """
        Just for MXene with terminals. Obtain 16 equivalent positions (fcc, hcp applicable.)

        Args:
            center:(int), Center of gravity atomic position.
            ignore_index:(list),Skip the interfering atoms.
            base_m:(str), Name of the base atom.
            terminal:(str), Terminal atom name.

        Returns:
            res:(np.array)
        """

        label = self.split_layer(ignore_index=ignore_index, tol=0.5, axis=2,
                                 force_plane=True, reverse=True, force_finite=True)
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

        m_dict = get_plane_neighbors_to_center(self, center_index=center, neighbors_name=[base_m],
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

        o_dict = get_plane_neighbors_to_center(self, center_index=core_index, neighbors_name=[terminal],
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
        Add z atoms randomly, x, y using the input values.

        Args:
            center:(int), Center of gravity atomic position.
            ignore_index:(list),Skip the interfering atoms.
            base_m:(str), Name of the base atom.
            terminal:(str), Terminal atom name.
            number_each: (int), How many z samples are there for each pair of x and y.
            random_state: (int), random state.
            add_atom: (str), Add atom name.
            debug: (bool), Debug. If debugged, all added atoms are displayed in one structure.
            up_down: (str), up and down
            perturb_base:  (str), Perturbed the initial structure.
            offset_z: (str), height along Z
            alpha: (str), Shift the Z-wave coefficient.

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
        structure.

        The lattice is operated by the rotation matrix only!!!

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
        from pymatgen.io.ase import AseAtomsAdaptor
        aaa = AseAtomsAdaptor()
        return aaa.get_atoms(self)

    def show(self):
        """
        Plot by ase.
        """
        from pymatgen.io.ase import AseAtomsAdaptor
        aaa = AseAtomsAdaptor()
        atoms_s = aaa.get_atoms(self)
        from ase.visualize import view
        view(atoms_s)

    def view(self):
        """The same as show. plot by ase."""
        self.show()


if __name__ == "__main__":
    mx = MXene.from_standard(terminal_site=["hcp", "hcp"], base=["W", "Ti"],
                             carbide_nitride="C", n_base=3, terminal="O")
    assert "hcp" == mx.check_terminal_sites_by_3_layer(up_down="up")
    assert "hcp" == mx.check_terminal_sites_by_3_layer(up_down="down")
    mx = MXene.from_standard(terminal_site=["hcp", "fcc"], base=["W", "Ti"],
                             carbide_nitride="C", n_base=3, terminal="O")
    assert "fcc" == mx.check_terminal_sites_by_3_layer(up_down="up")
    assert "hcp" == mx.check_terminal_sites_by_3_layer(up_down="down")
    mx = MXene.from_standard(terminal_site=["top", "hcp"], base=["W", "Ti"],
                             carbide_nitride="C", n_base=3, terminal="O")
    assert "hcp" == mx.check_terminal_sites_by_3_layer(up_down="up")
    assert "top" == mx.check_terminal_sites_by_3_layer(up_down="down")

    # po = Poscar(mx)
    # po.write_file("POSCAR")
    # mx2 = mx.change_top_layer_site(terminal_site_to="fcc-top", up_down="updown",
    #                                atom_type="S-F")
    # po = Poscar(mx2)
    # po.write_file("new-POSCAR")
