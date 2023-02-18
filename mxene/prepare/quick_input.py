# -*- coding: utf-8 -*-
"""
Notes:

    # 0. generate structure.
    >>> pure_space = {"super_cell": [(1, 1, 1), ],
    ...              "base": ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"],
    ...              "carbide_nitride": ["C","N"]}
    >>> pgs = list(ParameterGrid(pure_space))

    >>> mxs = [MXene.from_standard(**i, random_state=0) for i in tqdm(pgs)]

    # 1. generate vasp input.
    >>> mx_input_batch_parallelize(pgs, potpath=r"POT-database", log_file="path.csv",
    ...                           relax=True, n_jobs=10)

    # 2. read vasp (CONTCAR, ...) results, make supercell, doping, and make new input.

    >>> paths = find_leaf_path(".") # filter the needed
    >>> supercell_and_doping_mx_input_batch_parallelize(paths, pgs = None, potpath="POT-database", n_jobs=8)


    # 3. read vasp (CONTCAR, ...) results, make absorb, and make new input.

    >>> paths = find_leaf_path(".") # filter the needed
    >>> absorb_mx_input_batch_parallelize(paths, pgs= None, potpath="POT-database", n_jobs=8)

"""

import multiprocessing
import pathlib
from typing import List, Dict, Iterable

import numpy as np
import pandas as pd
from pymatgen.io.vasp import Potcar, Poscar, Incar
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from mxene.core.mxenes import MXene
from mxene.prepare.conf_files import opt_incar
from mxene.prepare.pre_kpoints import kp_dict
from mxene.prepare.pre_potcar import check_potcar
from mxene.prepare.vaspinput import MXVaspInput


# 1. generate vasp input

def mx_input(pg: Dict, potpath="POT-database", incar=opt_incar, terminal="O", disk_num=0,
             run_file=None, relax=True, **kwargs):
    doping = pg["doping"] if "doping" in pg else None
    super_cell = pg["super_cell"] if "super_cell" in pg else None
    terminal_site = pg["terminal_site"] if "terminal_site" in pg else None

    if isinstance(potpath, str):
        potpath = check_potcar(potpath=potpath)

    structure = MXene.from_standard(**pg)

    if relax:
        try:
            # try to relax by M3GNet
            structure = structure.relax()
            structure = MXene.from_structure(structure)
        except BaseException:
            pass

    root = "."
    try:
        disk = structure.get_disk(root, tol=0.5, doping=doping, terminal=terminal,
                                  ignore_index=-1,
                                  force_plane=True,
                                  )
    except BaseException as e:
        disk = f"{root}/{disk_num}"

    if terminal_site is None:
        disk = f"{disk}/{super_cell[0]}{super_cell[1]}{super_cell[2]}"
    else:
        disk = f"{disk}/{terminal_site}-{super_cell[0]}{super_cell[1]}{super_cell[2]}"

    kpoints = kp_dict[super_cell]
    poscar = Poscar(structure)
    incar = Incar.from_string(incar) if isinstance(incar, str) else incar
    potcar = Potcar(tuple(poscar.site_symbols), sym_potcar_map=potpath)
    mxin = MXVaspInput(incar, kpoints, poscar, potcar, optional_files=run_file)

    mxin.write_input(output_dir=str(disk), make_dir_if_not_present=True)
    return disk


def mx_input_batch(pgs: Iterable[Dict], potpath=r"POT-database", start=0,
                   relax=True, log_file="path.csv", **kwargs):
    pgs = pgs.tolist() if isinstance(pgs, np.ndarray) else list(pgs)

    if pgs is None:
        pgs = [{}]

    potpath = check_potcar(potpath=potpath)

    log = {}

    for i, pg in tqdm(enumerate(pgs)):

        try:
            logi = mx_input(pg, disk_num=i + start, potpath=potpath, incar=opt_incar, terminal="O",
                            run_file=None, relax=relax,
                            **kwargs)
        except BaseException as e:
            logi = f"Error(num:{i},err_msg:{e})"
        log.update({i: logi})

    log = {"log": log}

    log = pd.DataFrame.from_dict(log)
    log.to_csv(log_file)
    # return log


def _func2(iterable):
    mx_input_batch(iterable[0], log_file=iterable[1],
                   potpath=iterable[2], start=iterable[3], relax=iterable[4])


def mx_input_batch_parallelize(pgs: Iterable[Dict], potpath=r"POT-database", log_file="path.csv",
                               relax=True, n_jobs=10):
    assert n_jobs>1

    pgs = list(pgs)
    l = len(pgs)
    step = l // (n_jobs - 1)
    indices_or_sections = [step * i for i in range(1, n_jobs)]

    if indices_or_sections == l:
        n_jobs = n_jobs - 1
        step = l // (n_jobs - 1)
        indices_or_sections = [step * i for i in range(1, n_jobs)]

    msgs = np.split(np.array(pgs), indices_or_sections=indices_or_sections)
    log_files = [f"part-0-{log_file}"] + [f"part-{i}-{log_file}" for i in indices_or_sections]
    potpaths = [potpath] * n_jobs
    relaxs = [relax] * n_jobs
    starts = [step * i for i in range(n_jobs)]


    pool = multiprocessing.Pool(processes=n_jobs)

    tqdm(pool.map(func=_func2, iterable=zip(msgs, log_files, potpaths, starts, relaxs)))
    pool.close()
    pool.join()


# 2. make supercell and doping


def supercell_and_doping(structure, doping, super_cell, pathi, incar=None,
                         potpath="POT-database"):
    if incar is None:
        incar = Incar.from_string(opt_incar)

    if isinstance(potpath, str):
        potpath = check_potcar(potpath=potpath)

    kpoints = kp_dict[super_cell]
    sti = MXene.from_structure(structure)

    sti.make_supercell(super_cell, )
    sti.pure_add_doping(doping=doping)

    poscar_new = Poscar(sti)
    if "no_doping" in pathi:
        pi_new = pathi.replace("no_doping", doping)
        pi_new = pathlib.Path(pi_new)
        if str(pi_new.name) == "111":
            pi_new = pi_new.parent / f"{super_cell[0]}{super_cell[1]}{super_cell[2]}"
        else:
            pi_new = pi_new / f"{super_cell[0]}{super_cell[1]}{super_cell[2]}"
    else:
        try:
            pi_new = sti.get_disk(doping=doping, ignore_index=-1)
        except BaseException:
            pi_new = pathlib.Path(pathi)
            if str(pi_new.name) == "111":
                pi_new = pi_new.parent / f"{super_cell[0]}{super_cell[1]}{super_cell[2]}"
            else:
                pi_new = pi_new / f"{super_cell[0]}{super_cell[1]}{super_cell[2]}"

    out_dir = pi_new

    potcar = Potcar(tuple(poscar_new.site_symbols), sym_potcar_map=potpath)
    mxin = MXVaspInput(incar, kpoints, poscar_new, potcar, optional_files=None)
    mxin.write_input(output_dir=out_dir)
    return out_dir


def supercell_and_doping_mx_input_batch_parallelize(paths: List, pgs: Iterable[Dict] = None,
                                                    potpath="POT-database",
                                                    n_jobs=8,
                                                    read_file="CONTCAR"
                                                    ):
    if isinstance(potpath, str):
        potpath = check_potcar(potpath=potpath)

    pool = multiprocessing.Pool(processes=n_jobs)
    for pi in tqdm(paths):
        pool.apply(func=supercell_and_doping_mx_input, args=(pi,),
                   kwds={"pgs": pgs, "potpath": potpath, "read_file": read_file})
    pool.close()
    pool.join()


def supercell_and_doping_mx_input(pi, pgs: Iterable[Dict] = None, potpath="POT-database", read_file="CONTCAR"):
    if isinstance(potpath, str):
        potpath = check_potcar(potpath=potpath)

    if pgs is None:
        nm_list = ["C", "N", "F", "P", "S", "Cl"]
        tm_list = ["Sc", "Y", "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Mn",
                   "Re", "Fe", "Ru", "Os", "Co", "Rh", "Ir", "Ni", "Pd", "Pt", "Cu", "Ag",
                   "Au", "Zn", "Cd"]

        doping_list = nm_list + tm_list

        dop_space = {
            "super_cell": [(2, 2, 1), (3, 3, 1)],
            "doping": doping_list, }

        pgs = ParameterGrid(dop_space)

    try:
        mxvi = MXVaspInput.from_directory(pi, structure_file=read_file)
        poscar = mxvi["POSCAR"]
        incar = mxvi["INCAR"]
        if poscar is None:
            return
    except:
        print(f"The needed 4 files is not enough in:\n {pi}")
    else:
        for pgi in pgs:
            super_cell = pgi["super_cell"]
            doping = pgi["doping"]
            supercell_and_doping(poscar.structure, doping, super_cell, pi,
                                 incar=incar,
                                 potpath=potpath)


# 3. absorb

def absorb_atom(structure, absorb, site_name, site_type, pathi, kpoints, incar=None,
           potpath="POT-database"):
    if incar is None:
        incar = Incar.from_string(opt_incar)

    if isinstance(potpath, str):
        potpath = check_potcar(potpath=potpath)

    sti = MXene.from_structure(structure)

    sti.add_absorb(site_name=site_name, site_type=site_type, absorb=absorb, up_down="up")

    poscar_new = Poscar(sti)

    if "no_absorb" in pathi:
        pi_new = pathi.replace("no_absorb", absorb)
        pi_new = pathlib.Path(pi_new)
    else:
        try:
            pi_new = sti.get_disk(doping=absorb, ignore_index=-1)
        except BaseException:
            pi_new = pathlib.Path(pathi)

    out_dir = pi_new

    potcar = Potcar(tuple(poscar_new.site_symbols), sym_potcar_map=potpath)
    mxin = MXVaspInput(incar, kpoints, poscar_new, potcar, optional_files=None)
    mxin.write_input(output_dir=out_dir)
    return out_dir


def absorb_mx_input_batch_parallelize(paths: List, pgs: Iterable[Dict] = None, potpath="POT-database",
                                      n_jobs=8, read_file="CONTCAR"):
    if isinstance(potpath, str):
        potpath = check_potcar(potpath=potpath)

    pool = multiprocessing.Pool(processes=n_jobs)

    for pi in tqdm(paths):
        pool.apply(func=absorb_mx_input, args=(pi,),
                   kwds={"pgs": pgs, "potpath": potpath, "read_file": read_file})
    pool.close()
    pool.join()


def absorb_mx_input(pi, pgs: Iterable[Dict] = None, potpath="POT-database", read_file="CONTCAR"):
    if isinstance(potpath, str):
        potpath = check_potcar(potpath=potpath)

    if pgs is None:
        _predefined_am_list = ["Al", "Ca", "Li", "Mg", "Na", "K", "Zn", "H"]

        dop_space = {"site_name": ["S0"],
                     "absorb": _predefined_am_list, }

        pgs = ParameterGrid(dop_space)

    try:
        mxvi = MXVaspInput.from_directory(pi, structure_file=read_file)
        poscar = mxvi["POSCAR"]
        incar = mxvi["INCAR"]
        kpoints = mxvi["KPOINTS"]

        if poscar is None:
            return
    except NameError as e:
        print(e)
        print(f"The needed 4 files is not enough in:\n {pi}")
    else:
        for pgi in pgs:
            site_name = pgi["site_name"]
            absorb = pgi["absorb"]
            site_type = "top" if absorb == "H" else "fcc"
            absorb_atom(poscar.structure, absorb, site_name, site_type, pi, kpoints,
                   incar=incar,
                   potpath=potpath)
