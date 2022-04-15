import pathlib
from collections import Counter

from pymatgen.io.vasp import VaspInput


def make_disk(disk, terminal, base, carbide_nitride, n_base, doping, absorb=None, equ_name=None,
              site_name=None, add_atoms=None) -> pathlib.Path:
    nm_list = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I", "Te", "At"]
    if isinstance(base, (tuple, list)):
        assert isinstance(carbide_nitride, (tuple, list)), "terminal and base should be same type, (str or list)."
        # assert len(carbide_nitride) < len(base), "terminal should less than 1."
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
        name = {}
        name.update({base: n_base})
        name.update({carbide_nitride: n_base - 1})
    if terminal is not None:
        name.update({terminal: 2})
    else:
        name.update({"bare": 2})

    if doping is None:
        nm_tm = None
    else:
        nm_tm = "NM" if doping in nm_list else "TM"

    if doping is None:
        if nm_tm == "TM":
            if isinstance(base, (tuple, list)):
                raise ValueError("doping should be offered")
            else:
                doping = base_list[-1]
        elif nm_tm == "NM":
            doping = terminal
        else:
            doping = None

    if disk is None:
        disk = "."

    ll = []
    for k, v in name.items():
        if k == "bare":
            pass
        elif v != 1:
            ll.append(f"{k}{v}")
        else:
            ll.append(f"{k}")
    base_mx = "".join(ll)

    if doping is None:
        dop = "no_doping"
    else:
        dop = doping

    if add_atoms is None:
        add_atoms = "no_add"
    elif isinstance(add_atoms, (list, tuple)):
        add_atoms = "-".join(add_atoms)
    elif isinstance(add_atoms, str):
        pass
    else:
        raise TypeError("add_atoms just accept list of str or str.")

    if absorb is None:
        absorb = "no_absorb"

    disk = pathlib.Path(disk) / "MXenes" / base_mx / absorb / add_atoms / dop

    if site_name is not None:
        disk = disk / site_name

    if equ_name is not None:
        disk = disk / equ_name


    return disk


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
                conventions in implementing a as_dict() and from_dict method.
        """
        super().__init__(incar, kpoints, poscar, potcar, optional_files=optional_files, **kwargs)
        self.out_dir = "."
        if hasattr(poscar, "out_dir"):
            self.out_dir = poscar.out_dir

    def write_input(self, output_dir="auto", make_dir_if_not_present=True):
        """
        Write VASP input to a directory.

        Args:
            output_dir (str): Directory to write to. Defaults to current
                directory (".").
            make_dir_if_not_present (bool): Create the directory if not
                present. Defaults to True.
        """
        if output_dir == "auto":
            output_dir = self.out_dir
            super().write_input(output_dir=output_dir, make_dir_if_not_present=make_dir_if_not_present)
