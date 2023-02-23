# -*- coding: utf-8 -*-

# @Time  : 2022/10/2 13:20
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

import path
from collections import Counter
from typing import Sequence

import path


def make_disk(disk, terminal, base, carbide_nitride, n_base, doping, site_name, absorb=None, equ_name=None,
              add_atoms=None, base_num_cls=None, super_cell=None, terminal_site=None,
              old_type=True) -> path.Path:
    """Organize the name to one path."""
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
        carbide_nitride_list = [base] * (n_base - 1)
        name = {}
        name.update({base: n_base})
        name.update({carbide_nitride: n_base - 1})
    if terminal is not None:
        if not isinstance(terminal, (list, tuple)):
            name.update({terminal: 2})
        else:
            name.update({i: 1 for i in terminal})
    else:
        name.update({"bare": 2})

    if base_num_cls is None:
        if len(set(carbide_nitride_list)) > 1:
            base_num_cls = "".join([f"M{len(base_list)}"] + carbide_nitride_list)
        elif len(set(carbide_nitride_list)) == 1:
            if len(carbide_nitride_list) == 1:
                base_num_cls = f"M{len(base_list)}{carbide_nitride_list[0]}"
            else:
                base_num_cls = f"M{len(base_list)}{carbide_nitride_list[0]}{len(carbide_nitride_list)}"
        else:
            base_num_cls = "need_to_redefine"

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

    if absorb is None:
        absorb = "no_absorb"

    if add_atoms is None:
        add_atoms = "no_add"
    elif isinstance(add_atoms, (list, tuple)):
        add_atoms = "-".join(add_atoms)
    elif isinstance(add_atoms, str):
        pass
    else:
        raise TypeError("add_atoms just accept list of str or str.")

    assert equ_name is not None

    if isinstance(super_cell, tuple):
        super_cell = "".join([str(i) for i in super_cell])

    if isinstance(super_cell, str):
        base_num_cls = f"{base_num_cls}_{super_cell}"

    ############################################################

    disk = path.Path(disk) / "MXenes" / base_num_cls / base_mx / add_atoms / dop

    site_name = "-".join([i for i in [terminal_site, site_name] if i is not None])

    if old_type is True:
        if absorb == "no_absorb" and add_atoms == "no_add":  # pure
            pass
        else:
            if site_name =="":
                raise NotImplementedError("site_name must offered.")
            disk = disk / absorb / site_name  # doping or absorb
    else:
        if site_name == "":
            raise NotImplementedError("site_name must offered.")
        disk = disk / absorb / site_name  # doping or absorb

    disk = disk / equ_name

    return disk
