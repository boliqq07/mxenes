# -*- coding: utf-8 -*-

# @Time  : 2022/10/2 13:20
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

import copy
import itertools
import pathlib
import warnings
from collections import Counter
from itertools import chain
from typing import Union, Sequence, List

import numpy as np
from numpy.linalg import eigh
from pymatgen.core import Structure, Lattice, SymmOp, PeriodicSite

from mxene.utility.typing import CompositionLike, ArrayLike, ListTuple

from mxene.core.functions import coarse_and_spilt_array_ignore_force_plane, \
    get_plane_neighbors_to_center, Interp2dNearest, coarse_and_spilt_array, check_random_state


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
    _predefined_tem_list = ["O", "F", "OH", "Cl", None]
    _predefined_cn = ["C", "N"]
    _predefined_bm = ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"]
    _predefined_tem_z_axis = {"O": 1.0, "F": 0.9, "Cl": 1.2}
    _predefined_am_z_axis = {"K": 2.1, "Li": 1.3, "H": 1.0, "Ca": 1.6, "Na": 1.6, "Mg": 1.4, "Al": 1.1, "Zn": 1.2}

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

    def get_similar_layer_atoms(self, z0=0.0, tol=0.005, axis=2, frac=True, coords=None):
        """
        Get all same layer atoms by z0 site.

        Args:
            z0: float, site.
            tol: float, tolerance factor.
            axis: int, default the z axis.
            frac: bool, whether it is fractional/Cartesian coordinates.
            coords: np.ndarray, use this coordinates directly.

        Returns:
            atom_list: np.ndarray, atom index.
        """
        if coords is None:
            if frac is True:
                coords = self.frac_coords if isinstance(self, Structure) else self
            else:
                coords = self.cart_coords if isinstance(self, Structure) else self
        else:
            coords = coords
        coords = coords[:, axis]
        z0_atoms = np.array([i for i, z in enumerate(coords) if abs(z - z0) < tol])
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

    def split_layer(self, ignore_index=None, tol=0.5, axis=2,
                    force_plane=True, reverse=True, force_finite=False):
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
            axis: (int), axis to be grouped.
            tol: (float), tolerance distance for spilt.(less than interlayer spacing)
            ignore_index: (int, np.ndarray), jump 'out-of-station' atoms.
            force_plane: (bool), change the single atom to the nearest group.
            force_finite: (bool), force to change the index finite.
            reverse: (bool), reverse the label.

        Returns:
            labels: (np.ndarray) with shape (n,).

        """
        array = self.cart_coords[:, axis]

        layer_label = coarse_and_spilt_array_ignore_force_plane(array, ignore_index=ignore_index,
                                                                    tol=tol, force_plane=force_plane,
                                                                    reverse=reverse, force_finite=force_finite)

        return layer_label

    @staticmethod
    def check_terminal_sites_by_3_layer(array, up_down="up", tol=0.5):
        """
        Check the type of stacking type of the surface three layers.

        The up and down are different in some instance.

        Args:
            array: (np.ndarray), coords of 3 layer atoms.
            up_down: (str), up,down
            tol: (float), tol to split.

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
        According to the current atomic layer site, and stacking method,
        to obtain the atomic position (x, y) of the next layer.

        Args:
            array: (np.array), with shape (Nx2), site of all atoms
            site_type: (str), fcc, hcp, top
            up_down: (str),  up and down

        Returns:
            array: (np.array),array.
        """

        reverse = True if up_down == "up" else False

        layer_label = coarse_and_spilt_array_ignore_force_plane(array[:, -1], tol=0.5,
                                                                reverse=reverse, )

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
                             force_plane=False, force_finite=True, terminal_z_axis=1.2,
                             up_down="up", site_atom="O", reformed_array=None, tol=0.5)->np.ndarray:
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
            reformed_array: (np.ndarray), use this array for spilt directly. (1D).
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

        if reformed_array:
            layer_label = coarse_and_spilt_array_ignore_force_plane(reformed_array, ignore_index=ignore_index,
                                                                    force_plane=True,
                                                                    reverse=reverse, tol=tol)
        else:
            layer_label = self.split_layer(ignore_index=ignore_index, reverse=reverse, tol=tol,
                                           axis=2, force_plane=force_plane, force_finite=force_finite)

        layer1_values = np.min(layer_label)
        layer2_values = layer1_values + 1 if layer1_values + 1 in layer_label else layer1_values + 2
        layer1_index = np.where(layer_label == layer1_values)
        layer2_index = np.where(layer_label == layer2_values)
        coords1 = self.cart_coords[layer1_index]
        coords2 = self.cart_coords[layer2_index]

        if layer1_index[0].shape[0] != layer2_index[0].shape[0]:
            raise IndexError("The two layer is not with same size. This could be split error, "
                             "due to large deformation for some atoms. "
                             "Try to use 'reformed_array' directly. "
                             "One another solution: use 'ignore_index' to jump it.")

        offset12 = np.mean(coords1 - coords2, axis=0)

        if isinstance(site_atom, str) and site_atom in self._predefined_tem_z_axis:
            terminal_z_axis = self._predefined_tem_z_axis[site_atom]
        elif isinstance(site_atom, str) and site_atom in self._predefined_tem_z_axis:
            terminal_z_axis = self._predefined_am_z_axis[site_atom]
        elif isinstance(site_atom, str):
            terminal_z_axis = 1.2
        else:
            pass
            # terminal_z_axis = 1.2

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

    def add_next_layer_atoms(self,site_atom:Union[str, List[str]] = "Ag",
                             site_type: Union[None, str] = "fcc", ignore_index = None,
                             force_plane=False, force_finite = True, terminal_z_axis=1.2,
                             up_down="up_and_down",
                             reformed_array = None, tol = 0.5):
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
            reformed_array: (np.ndarray), use this array for spilt directly. (1D).
            ignore_index: (list of int), ignore index to split layer.
            site_atom: (str), name of next layer atoms.
            site_type: (str), fcc, hcp, top.
            up_down: (str),  up and down.
            force_plane: (bool), change the single atom to the nearest group.
            force_finite: (bool), force to change the index finite.
        """
        arr = []

        if "up" in up_down:
            coords_add1 = self.get_next_layer_sites(site_type=site_type, ignore_index = ignore_index,
                                 force_plane=force_plane, force_finite = force_finite,terminal_z_axis=terminal_z_axis,
                                 up_down="up", site_atom = site_atom, reformed_array = reformed_array, tol = tol)
            arr.append(coords_add1)
        if "down" in up_down:
            coords_add2 = self.get_next_layer_sites(site_type=site_type, ignore_index = ignore_index,
                                 force_plane=force_plane, force_finite = force_finite,terminal_z_axis=terminal_z_axis,
                                 up_down="down", site_atom=site_atom, reformed_array = reformed_array, tol=tol)
            arr.append(coords_add2)

        if len(arr)==2:
            arr = np.concatenate(arr,axis=0)
        else:
            arr = arr[0]

        if isinstance(site_atom, str):
            site_atom = [site_atom]*(arr.shape[0])

        if isinstance(site_atom, (list, tuple)):
            for n, s in zip(site_atom, arr):
                self.append(n, coords=s, coords_are_cartesian=True)

    @classmethod
    def from_standard(cls, terminal_site: Union[None, str] = "fcc",
                      doping: Union[None, str] = None, terminal: Union[None, str] = "O",
                      base="Ti", carbide_nitride="C", n_base=2, add_noise=False,
                      super_cell=(3, 3, 1), add_atoms=None, add_atoms_site=None,
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
            lattice = Lattice.from_parameters(3.0, 3.0, cc, 90.0000, 90.0000, 120)

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
            if terminal_site == "fcc":
                sam1, sam2 = 2, -3
            elif terminal_site == "hcp":
                sam1, sam2 = 1, -2
            else:  # top
                sam1, sam2 = 0, -1

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

        if super_cell is not None:
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
            self.get_neighbor_list(affect_radius, numerical_tol=1e-3, sites=[self.sites[center]], exclude_self=True)
        index = np.sum(np.abs(offset_vectors), axis=1) == 0
        points_indices = points_indices[index]
        distances = distances[index]
        label = coarse_and_spilt_array(distances, tol=0.02)

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
        label = coarse_and_spilt_array(self.cart_coords[:, 2], tol=0.1)
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

        st = self.__class__(lattice=self.lattice, species=self.species, coords=frac_coords, coords_are_cartesian=False)
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

        labels = coarse_and_spilt_array(frac_coords_z, tol=0.02).astype(float)
        labels[~index] = np.nan

        s = max(labels) if up_down == "up" else min(labels)

        index2 = labels == s

        frac_coords_xy_temp[index2] = frac_coords_xy[index2]

        l = np.sum((frac_coords_xy_temp - 0.5) ** 2, axis=1) ** 0.5

        center = np.argmin(l)

        return center

    def add_absorb(self, center: Union[int, None] = -1, site_type: Union[None, str] = "top", offset_z: float = 0,
                   add_noise: bool = True, up_down: str = "up", site_name: str = "S0", equivalent: str = "ini_opt",
                   absorb: str = "H", absorb_site: Union[np.ndarray, ListTuple] = None, reformed_array: np.ndarray = None,
                   ignore_index: Union[np.ndarray, tuple, int, List] = None,
                   coords_are_cartesian: bool = True, tol: float = 0.3, random_state=None,
                   random_factor: float = 0.001,  pure: bool = False) -> "MXene":
        """
        Add adsorbent atoms. Change the original data!!!

        Args:
            random_state: random_state seed.
            tol: group layer tolerance.
            random_factor: (float), factor for random size.
            up_down: (str), up or down.
            reformed_array: np.ndarray, use this array for spilt directly. (1D)
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
            center = self.surface_pure_center(up_down="up")

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
                                                  up_down=up_down, site_atom=absorb, tol=tol,
                                                  reformed_array=reformed_array)

                st2 = copy.deepcopy(self)
                [st2.append(species=i, coords=j, coords_are_cartesian=True) for i, j in
                 zip([absorb] * sites.shape[0], list(sites))]

                points_and_distance_to_m0 = get_plane_neighbors_to_center(st2, center_index=center,
                                                                          neighbors_name=absorb,
                                                                          plane=False, ignore_index=ignore_index, r=7.0)

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

    def get_disk(self, disk='.', site_name="S0", equ_name="ini_opt", nm_tm="TM",
                 ignore_index=None,add_atoms=None, tol=0.4,
                 absorb=None, doping=None, terminal=None, carbide_nitride=None,
                 ) -> pathlib.Path:
        """Just for single doping, single absorb, single type terminal.
        for some name, the code could judge the parameter automatically, and for others, you should offer it.
        such as site_name="S0", equ_name="ini_opt", nm_tm="TM",

        **(1)不吸附**

        MXenes -> 基底层数 -> 基底名称 -> 负载物 -> 搀杂物 -> pure/pure_static

        例子

        MXenes -> M2C -> Ti2NO2 -> no_add -> no_doping -> pure/pure_static

        MXenes -> M2N -> Ti2NO2 -> no_add -> Mo        -> pure/pure_static

        **(2)吸附**

        MXenes -> 基底层数  -> 基底名称 -> 负载物 -> 搀杂物 -> 吸附物  -> 标签

        例子
        MXenes -> M2N  -> Ti2NO2      -> no_add -> no_dopin -> H/add_H  -> top -> 00

        MXenes -> M3N2 -> TiNCrNTi-O2 -> Hf     -> C         -> Li -> S0  -> 00

        **(3)NEB**

        MXenes -> 基底层数  -> 基底名称 -> 负载物 -> 搀杂物 -> 吸附物 -> 等效位点 -> 路径名称

        例子

        MXenes -> M2N -> Ti2NO2 -> no_add -> no_doping -> H -> S0-S1 -> 00/01/01/03/04/ini/fin/...

        """
        absorb_ = absorb
        doping_ = doping
        add_atoms_ = add_atoms
        terminal_ = terminal
        carbide_nitride_ = carbide_nitride

        names = [site.specie.name for site in self]
        if len(names) <= 12:
            mx = self.copy()
            mx.make_supercell((2, 2, 1))
            names = [site.specie.name for site in mx]
        else:
            mx = self
        labels = mx.split_layer(ignore_index=ignore_index, tol=tol)
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
            if i in base:
                base.append(i)
            elif nm_tm == "TM" and i in self._predefined_tm_list and i not in base:
                doping.append(i)
            elif nm_tm == "NM" and i in self._predefined_nm_list[1:] and i not in base:
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

        assert len(list(set(terminal))) <= 1, "just for same terminal."
        terminal = terminal_ if terminal_ else terminal[0]
        carbide_nitride = carbide_nitride_ if carbide_nitride_ else carbide_nitride

        if isinstance(base, (tuple, list)):
            assert isinstance(carbide_nitride, (tuple, list)), "terminal and base should be same type, (str or list)."
            assert len(carbide_nitride) < len(base), "carbide_nitride should less than base."

        from mxene.organize.disk import make_disk

        self.out_dir = make_disk(disk, terminal, base, carbide_nitride, n_base=None, doping=doping, absorb=absorb,
                                 equ_name=equ_name, base_num_cls=None,
                                 site_name=site_name, add_atoms=add_atoms)
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

    def relax_by_predictor(self,predictor):


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
