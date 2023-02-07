# -*- coding: utf-8 -*-
import os

from monty.os.path import zpath
# @Time  : 2022/11/4 16:50
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

from pymatgen.io.vasp import VaspInput, Incar, Kpoints, Potcar, Poscar


class MXVaspInput(VaspInput):
    """
    Class to contain a set of vasp input objects corresponding to a run.
    """

    def __init__(self, incar, kpoints, poscar, potcar, optional_files=None, **kwargs):
        """
        Args:
            incar: Incar object.
            kpoints: Kpoints object.
            poscar: Poscar object.
            potcar: Potcar object.
            optional_files: Other input files supplied as a dict of {
                filename: object}. The object should follow standard pymatgen
                conventions in implementing as_dict() and from_dict method.
        """
        super().__init__(incar, kpoints, poscar, potcar, optional_files=optional_files, **kwargs)
        if hasattr(poscar.structure, "out_dir"):
            self.out_dir = poscar.structure.out_dir

    def write_input(self, output_dir="auto", make_dir_if_not_present=True):
        """
        Write VASP inputs to a directory.

        Args:
            output_dir (str): Directory to write to. Defaults to current
                directory (".").
            make_dir_if_not_present (bool): Create the directory if not
                present. Defaults to True.
        """
        if output_dir == "auto" and hasattr(self, "out_dir"):
            output_dir = self.out_dir
        elif output_dir == "auto" and not hasattr(self, "out_dir"):
            raise NotImplementedError("No default dir to store. please set ``output_dir``.")
        super().write_input(output_dir=output_dir, make_dir_if_not_present=make_dir_if_not_present)

    @staticmethod
    def from_directory(input_dir, optional_files=None, structure_file="CONTCAR"):
        """
        Read in a set of VASP input from a directory. Note that only the
        standard INCAR, POSCAR, POTCAR and KPOINTS files are read unless
        optional_filenames is specified.

        Args:
            input_dir (str): Directory to read VASP input from.
            optional_files (dict): Optional files to read in as well as a
                dict of {filename: Object type}. Object type must have a
                static method from_file.
        """
        sub_d = {}
        for fname, ftype in [
            ("INCAR", Incar),
            ("KPOINTS", Kpoints),
            (structure_file, Poscar),
            ("POTCAR", Potcar),
        ]:
            try:
                fullzpath = zpath(os.path.join(input_dir, fname))
                sub_d[fname.lower()] = ftype.from_file(fullzpath)
            except FileNotFoundError:  # handle the case where there is no KPOINTS file
                sub_d[fname.lower()] = None

        sub_d["optional_files"] = {}
        if optional_files is not None:
            for fname, ftype in optional_files.items():
                sub_d["optional_files"][fname] = ftype.from_file(os.path.join(input_dir, fname))
        return VaspInput(**sub_d)