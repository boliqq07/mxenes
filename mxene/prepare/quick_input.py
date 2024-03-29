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
    >>> supercell_and_doping_mx_input_batch(paths, pgs = None, potpath="POT-database", n_jobs=8)


    # 3. read vasp (CONTCAR, ...) results, make absorb, and make new input.

    >>> paths = find_leaf_path(".") # filter the needed
    >>> absorb_mx_input_batch(paths, pgs= None, potpath="POT-database", n_jobs=8)

"""

import multiprocessing
import pathlib
from typing import List, Dict, Iterable

import numpy as np
import pandas as pd
import path
from pymatgen.io.vasp import Potcar, Poscar, Incar
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from mxene.core.mxenes import MXene
from mxene.prepare.conf_files import opt_big_incar, opt_small_incar, opt_big_incar_isym
from mxene.prepare.pre_kpoints import kp_dict
from mxene.prepare.pre_potcar import check_potcar
from mxene.prepare.vaspinput import MXVaspInput


# 1. generate vasp input

def mx_input(pg: Dict, potpath="POT-database", incar=opt_small_incar, terminal="O", disk_num=0,
             run_file=None, relax=True, test=False, **kwargs):
    assert "base" in pg
    assert "carbide_nitride" in pg
    assert "terminal_site" in pg

    # not assert "doping" in pg
    # not assert "up_down" in pg
    # not assert "super_cell" in pg

    doping = pg["doping"] if "doping" in pg else None
    super_cell = pg["super_cell"] if "super_cell" in pg else None
    if "terminal_site" in pg:
        if pg["terminal_site"] is None:
            pg["terminal_site"] = "fcc"

    if doping is None:
        up_down = None
        equ_name = "pure_opt"
    else:
        up_down = pg["up_down"] if "up_down" in pg else "up"
        equ_name = "ini_opt"

    if isinstance(potpath, str):
        potpath = check_potcar(potpath=potpath)

    sym = False
    base = pg["base"]
    if len(base) == 2 and base[0] == base[1]:
        sym = True
    elif len(base) == 3 and base[0] == base[2]:
        sym = True
    elif len(base) == 4 and base[0] == base[1]:
        sym = True

    if sym:
        up_down = "up"
        terminal_site = pg["terminal_site"]
        terminal_site = terminal_site.split("-")[0] if "-" in terminal_site else terminal_site
        pg["terminal_site"] = terminal_site

    pg["up_down"] = up_down

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

        # decrease calculation (等位置覆盖)
        terminal_site = "auto"
        disk = structure.get_disk(root, tol=0.4, doping=doping,
                                  terminal=terminal,
                                  site_name=up_down,
                                  equ_name=equ_name,
                                  terminal_site=terminal_site,
                                  ignore_index=-1,
                                  force_plane=True,
                                  old_type=False,
                                  super_cell=super_cell,

                                  )
    except BaseException as e:
        disk = f"{root}/{disk_num}"

    if test:
        print(f"Path:{disk}")
    else:
        kpoints = kp_dict[super_cell]
        poscar = Poscar(structure)
        incar = Incar.from_string(incar) if isinstance(incar, str) else incar
        potcar = Potcar(tuple(poscar.site_symbols), sym_potcar_map=potpath)
        mxin = MXVaspInput(incar, kpoints, poscar, potcar, optional_files=run_file)

        mxin.write_input(output_dir=str(disk), make_dir_if_not_present=True)
    return disk


def mx_input_batch(pgs: Iterable[Dict], potpath=r"POT-database", start=0,
                   relax=True, log_file="path.csv", test=False, incar=opt_small_incar, **kwargs):
    pgs = pgs.tolist() if isinstance(pgs, np.ndarray) else list(pgs)

    if pgs is None:
        pgs = [{}]

    potpath = check_potcar(potpath=potpath)

    log = {}

    for i, pg in tqdm(enumerate(pgs)):

        try:
            logi = mx_input(pg, disk_num=i + start, potpath=potpath, incar=incar, terminal="O",
                            run_file=None, relax=relax, test=test,
                            **kwargs)
        except BaseException as e:
            logi = f"Error(num:{i},err_msg:{e})"
        log.update({i: logi})

    log = {"log": log}

    log = pd.DataFrame.from_dict(log)
    log.to_csv(log_file)
    # return log


# def _func2(iterable):
#     mx_input_batch(iterable[0], log_file=iterable[1],
#                    potpath=iterable[2], start=iterable[3], relax=iterable[4],test=iterable[5])


def mx_input_batch_parallelize(pgs: Iterable[Dict], potpath=r"POT-database", log_file="path.csv",
                               relax=True, n_jobs=10, test=False, incar=opt_small_incar):
    assert n_jobs > 1

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
    starts = [step * i for i in range(n_jobs)]

    ##### old #############
    # potpaths = [potpath] * n_jobs
    # relaxs = [relax] * n_jobs
    # tests = [test] * n_jobs
    # pool = multiprocessing.Pool(processes=n_jobs)
    #
    # tqdm(pool.map(func=_func2, iterable=zip(msgs, log_files, potpaths, starts, relaxs, tests)))
    # pool.close()
    # pool.join()

    ###### new #################
    pool = multiprocessing.Pool(processes=n_jobs)

    for pi, log, start in tqdm(zip(msgs, log_files, starts)):
        pool.apply(func=mx_input_batch, args=(pi,),
                   kwds={"log_file": log, "potpath": potpath, "start": start, "relax": relax,
                         "test": test, "incar": incar})
    pool.close()
    pool.join()


# 2. make supercell and doping


def supercell_and_doping(structure, doping, super_cell, pathi, incar=None,
                         potpath="POT-database", up_down="up", no_doping_name="no_doping", test=False):
    assert up_down in ["up", "down"]

    if isinstance(potpath, str):
        potpath = check_potcar(potpath=potpath)

    kpoints = kp_dict[super_cell]

    sti = MXene.from_structure(structure)

    if super_cell == (1, 1, 1):
        pass
        if incar is None:
            incar = Incar.from_string(opt_small_incar)
    else:
        sti.make_supercell(super_cell)
        if incar is None:
            incar = Incar.from_string(opt_big_incar_isym)

    if sti.is_mirror_sym():
        up_down = "up"

    doping_index, atom = sti.pure_add_doping(doping=doping, up_down=up_down)

    if doping is None:
        if no_doping_name == "atom":
            no_doping_name = str(atom)
    else:
        no_doping_name = doping

    poscar_new = Poscar(sti)
    if "no_doping" in pathi:
        pi_new = pathi.replace("no_doping", no_doping_name)
        pi_new = pi_new.replace("pure", "ini")
        pi_new = pathlib.Path(pi_new)
        end = pi_new.name
        terminal_site2 = path.Path(pi_new).parent.name
        site_name = "-".join([i for i in [terminal_site2, up_down] if i is not None])
        pi_new = pi_new.parent.parent / site_name / end

        if "111" in str(pi_new):
            pi_new = str(pi_new).replace("111", f"{super_cell[0]}{super_cell[1]}{super_cell[2]}")
        else:
            pi_new = pi_new / f"{super_cell[0]}{super_cell[1]}{super_cell[2]}"

    else:
        try:
            if doping is None and no_doping_name == "atom":
                doping = str(atom)

            pi_new = sti.get_disk(doping=doping, ignore_index=-1, site_name=up_down,
                                  equ_name="ini_opt", tol=0.4, terminal_site="auto", super_cell=super_cell,
                                  old_type=False)
        except BaseException:
            pi_new = pathlib.Path(pathi)
            if "111" in str(pi_new):
                pi_new = str(pi_new).replace("111", f"{super_cell[0]}{super_cell[1]}{super_cell[2]}")
            else:
                pi_new = pi_new / f"{super_cell[0]}{super_cell[1]}{super_cell[2]}"

    out_dir = path.Path(str(pi_new))

    if len(sti.atomic_numbers) > 50:
        incar = opt_big_incar_isym

    if test:
        print(f"Old:{pathi}")
        print(f"New:{out_dir}")
    else:

        potcar = Potcar(tuple(poscar_new.site_symbols), sym_potcar_map=potpath)
        mxin = MXVaspInput(incar, kpoints, poscar_new, potcar, optional_files=None)
        mxin.write_input(output_dir=out_dir)
        # out_dir.rmtree_p()
        pass
    return out_dir


def supercell_and_doping_mx_input_batch(paths: List, pgs: Iterable[Dict] = None,
                                        potpath="POT-database",
                                        n_jobs=8,
                                        read_file="CONTCAR",
                                        test=False
                                        ):
    if isinstance(potpath, str):
        potpath = check_potcar(potpath=potpath)

    pool = multiprocessing.Pool(processes=n_jobs)
    for pi in tqdm(paths):
        pool.apply(func=supercell_and_doping_mx_input, args=(pi,),
                   kwds={"pgs": pgs, "potpath": potpath, "read_file": read_file, "test": test})
    pool.close()
    pool.join()


def supercell_and_doping_mx_input(pi, pgs: Iterable[Dict] = None, potpath="POT-database",
                                  read_file="CONTCAR", no_doping_name="no_doping", test=False):
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
        try:
            for pgi in pgs:
                super_cell = pgi["super_cell"]
                doping = pgi["doping"]

                if "up_down" in pgi:
                    if pgi["up_down"] is None:
                        up_down = "up"
                    else:
                        up_down = pgi["up_down"]
                else:
                    up_down = "up"

                supercell_and_doping(poscar.structure, doping, super_cell, pi,
                                     incar=incar,
                                     potpath=potpath, up_down=up_down, no_doping_name=no_doping_name,
                                     test=test)
        except IndexError:
            print(f"Error for '{pi}' and store to doping_fail_path.temp")
            with open("doping_fail_path.temp", "a+") as f:
                f.write(str(pi))


# 3. absorb

def absorb_atom(structure, absorb, site_name, site_type, pathi, kpoints, incar=None, up_down=None,
                potpath="POT-database", test=True):
    if incar is None:
        incar = Incar.from_string(opt_big_incar)

    if len(structure.atomic_numbers) > 50:
        incar = opt_big_incar_isym

    if isinstance(potpath, str):
        potpath = check_potcar(potpath=potpath)

    sti = MXene.from_structure(structure)

    pathi = path.Path(pathi)

    if up_down is None or up_down == "auto":
        if any([True for i in pathi.parts() if "down" in str(i)]):
            up_down = "down"
        else:
            up_down = "up"
    else:
        up_down = str(up_down)

    sti.add_absorb(site_name=site_name, site_type=site_type, absorb=absorb, up_down=up_down,
                   ignore_index=-1)

    poscar_new = Poscar(sti)

    pi_new = str(pathi).replace("no_absorb", absorb)
    if "up" in str(pi_new) or "down" in str(pi_new):
        pi_new = str(pi_new).replace(up_down, f"{up_down}-{site_name}-{site_type}")
    else:
        old = str(pathi.parts()[-2])
        pi_new = str(pi_new).replace(old, f"{old}-{site_name}-{site_type}")

    pi_new = pathlib.Path(pi_new)

    out_dir = pi_new

    if test:
        print(f"Old:{pathi}")
        print(f"New:{pi_new}")
    else:
        print(f"Old:{pathi}")
        print(f"New:{pi_new}")
        potcar = Potcar(tuple(poscar_new.site_symbols), sym_potcar_map=potpath)
        mxin = MXVaspInput(incar, kpoints, poscar_new, potcar, optional_files=None)
        mxin.write_input(output_dir=str(out_dir))
        pass
    return out_dir


def absorb_mx_input_batch(paths: List, pgs: Iterable[Dict] = None, potpath="POT-database",
                          n_jobs=8, read_file="CONTCAR", test=True):
    if isinstance(potpath, str):
        potpath = check_potcar(potpath=potpath)

    pool = multiprocessing.Pool(processes=n_jobs)

    for pi in tqdm(paths):
        pool.apply(func=absorb_mx_input, args=(pi,),
                   kwds={"pgs": pgs, "potpath": potpath, "read_file": read_file, "test": test})
    pool.close()
    pool.join()


def absorb_mx_input(pi, pgs: Iterable[Dict] = None, potpath="POT-database", read_file="CONTCAR", test=True):
    tm_list = ["Sc", "Y", "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Mn",
               "Re", "Fe", "Ru", "Os", "Co", "Rh", "Ir", "Ni", "Pd", "Pt", "Cu", "Ag",
               "Au", "Zn", "Cd"]

    if isinstance(potpath, str):
        potpath = check_potcar(potpath=potpath)

    if pgs is None:
        _predefined_am_list = ["H", "Al", "Ca", "Li", "Mg", "Na", "K", "Zn"]

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
        try:
            for pgi in pgs:
                site_name = pgi["site_name"]
                absorb = pgi["absorb"]

                if "site_type" in pgi:
                    site_type = pgi["site_type"]
                else:
                    site_type = "top" if absorb == "H" else "fcc"

                if site_type == "top":
                    if poscar.structure.symbol_set[-1] in tm_list:
                        site_name = "S0"  # some `center` for non-mental

                if "up_down" in pgi:
                    if pgi["up_down"] == "auto":
                        up_down = None
                    else:
                        up_down = pgi["up_down"]
                else:
                    up_down = None

                absorb_atom(poscar.structure, absorb, site_name, site_type, pi, kpoints,
                            incar=incar, test=test, up_down=up_down,
                            potpath=potpath)
        except (IndexError, ValueError):
            print(f"Error for {pi} and store to absorb_fail_path.temp")
            with open("absorb_fail_path.temp", "a+") as f:
                f.write(str(pi))
