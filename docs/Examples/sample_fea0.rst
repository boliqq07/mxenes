Batch generation of VASP calculation files
============================================

    >>> from mxene.core.mxenes import MXene
    >>> from pymatgen.io.vasp import Poscar, Kpoints, Incar
    >>> from mxene.prepare.conf_files import opt_incar
    >>> from mxene.prepare import pre_potcar
    >>> from mxene.prepare.vaspinput import MXVaspInput
    >>>
    >>> kpoints = Kpoints(kpts=((3, 3, 1),))
    >>> incar = Incar.from_string(opt_incar)
    >>> pos = MXene.from_file('POSCAR')
    >>>
    >>> for i in ["Mo", "Sc", "Ag"]:
    >>>     pos.replace(44, i)
    >>>     poscar = Poscar(pos)
    >>>     sym_potcar_map = pre_potcar.check_potcar(potpath=r"POT-database")
    >>>     potcar = pre_potcar.get_potcar(poscar, sym_potcar_map=sym_potcar_map)
    >>>     mxinput = MXVaspInput(incar, kpoints, poscar, potcar, optional_files=None)
    >>>     mxinput.write_input(output_dir='mxene_data' + '\\' + i, make_dir_if_not_present=True)