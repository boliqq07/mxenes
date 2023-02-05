# -*- coding: utf-8 -*-

# @Time  : 2022/11/4 16:50
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

from pymatgen.io.vasp import VaspInput


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
