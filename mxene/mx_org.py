"""

****此脚本用来重新组织 MXenes文件夹****

0. 此代码可以作脚本单独使用，需要安装python3，若需要获取推荐路径，需要安装pymatgen,ase,mxene等包。

1. 此代码不更改文件内容, 仅调整目录（注意先数据备份，确保无误再删除原文件夹）。

2. 若经过此代码分析，经过整理仍然无法通过，将在路径下添加输出 un_mark.txt文件 ,输出必要信息。

3. 若优化结构文件夹及静态计算文件夹同时存在，默认仅保存静态计算（未添加，开发中）。

4. 整理后数据，不能出现同一文件夹下，即出现计算文件又出现子文件夹情况，（辅助文件例外）。

5. 所有未计算，或者 un_mark 数据不建议被上传到数据库。

6. 此脚本可自由分发并可自由更改使用，以mxene包中版本为源头版本。若bug或重要内容，需要更改源头版本，请联系我。

最终统一的文件夹格式为:

(1).不吸附

MXenes -> 基底名称 -> 负载物 -> 搀杂物 -> 标签

例子

MXenes -> Ti2NO2   -> no_add -> no_doping -> pure_opt/pure_static

MXenes -> TiNCr-O2 -> no_add -> Mo       -> pure_optpure_static

(2).吸附

MXenes -> 基底名称 -> 负载物 -> 搀杂物 -> 吸附物  -> 等效位点 -> 标签

例子

MXenes -> Ti2NO2     -> no_add -> no_dopin -> H/add_H -> top -> opt/static

MXenes -> TiNCrNTi-O2 -> Hf    -> C        -> Li       -> S0 -> 00


(3).NEB
MXenes -> 基底名称 -> 负载物 -> 搀杂物 -> 吸附物 -> 等效位点路径名称 -> 标签

例子
MXenes -> Ti2NO2 -> no_add -> no_doping -> H -> S0-S1/neb_S0-S1 -> 00/01/01/03/04/ini/fin/...

``->`` 代表下一层
``/`` 代表或者


"""
import copy
import os
import pathlib
import shutil
import warnings
from typing import Union, List
import path

nm_list = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I", "Te", "At"]
tm_list = ["Sc", "Y", "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Mn",
           # "Tc",
           "Re", "Fe", "Ru", "Os", "Co", "Rh", "Ir", "Ni", "Pd", "Pt", "Cu", "Ag", "Au", "Zn", "Cd"]
am_list = ["Al", "Ca", "Li", "Mg", "Na", "K", "Zn", "H", "OH", "O", "OOH"]
tem_list = ["O", "F", "OH", "Cl"]
cn_list = ["C", "N"]
bm_list = ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"]
tnm_list = tm_list + nm_list

no_absorb_list = ["no_absorb", "pure", "static", "pure_static", "pure_opt", "scf", "opt"]
site_list = ["S0", "S1", "S2", "neb_S0-S0", "neb_S0-S1", "neb_S0-S2", "neb", "S0-S0", "S0-S1", "S0-S2", "top"]
label_list = ["00", "01", "02", "03", "04", "ini", "fin", "ini_opt", "fin_opt"]


def path_regroup(pt: Union[str, path.Path, os.PathLike, pathlib.Path],
                 base_name: Union[tuple, int, None, str] = 1,
                 add: Union[tuple, int, None, str] = 2,
                 doping: Union[tuple, int, None, str] = 3,
                 absorb: Union[tuple, int, None, str] = 4,
                 site: Union[tuple, int, None, str] = 5,
                 label: Union[tuple, int, None, str] = 6,
                 **kwargs
                 ) -> path.Path:
    """
    Re-group the disk.

    重新组织文件夹,使用序号标记文件层，使得最终为:

    MXenes -> 基底名称 -> 负载物 -> 搀杂物 -> 吸附物/未吸附/pure_* -> 等效位点/路径 -> 标签
    MXenes -> base_name -> add -> doping -> absorb -> site -> lable

    1.若输入序号n,则代表使用旧文件夹的第n层作为名称.
    2.若输入None,则代表跳过 不（掺杂no_doping，负载no_add，吸附no_absorb）.
    3.若输入字符串,则该层直接使用该名称.


    Examples:
    ----------

    path_regroup(pt=r"E:/x/MXenes/M2XO2-TM/Hf/C/Au/2",
                 base_name=1, add=2,doping=3,absorb=4,site=5,label=6)

    path_regroup(pt=r"E:/x/MXenes/M2XO2-TM/Hf/C/Au/2",
                 base_name=(1,2,3),add=4, doping=None,absorb=None,site=5,label="00",)

    path_regroup(pt=r'E:/x/MXenes/M2XO2/Hf/C/Au/H/neb/ini',
                 base_name=(1,2,3),add=4, doping=None,absorb=None,site="neb_S0-S0",label="ini",)

    path_regroup(pt=r'E:/x/MXenes/M2XO2/Hf/C/Au/H/neb/ini',
                 base_name=(1,2,3),add=4, doping=None, absorb=None,site="neb_S0-S0",label="ini",
                 fun_base_name=lambda x:f"S{x}")

    Args:
        pt: (str, path.Path, os.PathLike,pathlib.Path), path of leaf node.
        base_name: (tuple,int,None,str), layer of base structure name.
        add: (tuple,int,None,str), layer of add atom.
        doping: (tuple,int,None,str), layer of doping atom.
        absorb: (tuple,int,None,str), layer of absorb atom.
        site: (tuple,int,None,str), layer of site name.
        label: (tuple,int,None,str), layer of label name.
        **kwargs: keywords are formed "func_*", such as "fun_base_name", and the value are one function.

    Returns:
        new_pt:(path.Path), new path
    """
    ls = [i for i in [base_name, add, doping, absorb, site, label] if i is not None]
    ls = [i if isinstance(i, int) else max(i) for i in ls]
    dir_max = max([i for i in ls if isinstance(i, int)])

    absorb_ = copy.copy(absorb)
    site_ = copy.copy(site)
    label_ = copy.copy(label)

    if not isinstance(pt, path.Path):
        pt = path.Path(pt)
    parents = pt.splitall()

    assert "MXenes" in parents
    mx_site = parents.index("MXenes")
    pre_parents = parents[:mx_site]

    real_parents = parents[mx_site:]
    assert len(real_parents) > dir_max, "The deep of path should large than max number of marks,(label,site,...)"

    if isinstance(add, tuple):
        add = [real_parents[i] for i in add]
    elif isinstance(add, int):
        add = real_parents[add]
    elif isinstance(add, str):
        pass
    else:
        add = "no_add"

    if isinstance(doping, tuple):
        doping = [real_parents[i] for i in doping]
    elif isinstance(doping, int):
        doping = real_parents[doping]
    elif isinstance(doping, str):
        pass
    else:
        doping = "no_doping"

    if isinstance(absorb, tuple):
        absorb = [real_parents[i] for i in absorb]
    elif isinstance(absorb, int):
        absorb = real_parents[absorb]
    elif isinstance(absorb, str):
        pass
    else:
        absorb = "no_absorb"

    if isinstance(base_name, tuple):
        base_name = [real_parents[i] for i in base_name]
    elif isinstance(base_name, int):
        base_name = real_parents[base_name]
    elif isinstance(base_name, str):
        pass
    else:
        base_name = "MnXn-1Tx"

    add = "-".join(add) if isinstance(add, list) else add
    doping = "-".join(doping) if isinstance(doping, list) else doping
    absorb = "-".join(absorb) if isinstance(absorb, list) else absorb
    base_name = "-".join(base_name) if isinstance(base_name, list) else base_name

    if "add_" in absorb:
        absorb = absorb.replace("add_", "")
    if "add-" in absorb:
        absorb = absorb.replace("add-", "")

    if absorb_ is None:
        if site_ is not None or label_ is not None:
            warnings.warn("abosrb is None means not absorb and the site_path and label should be None.", UserWarning)

    if isinstance(site, tuple):
        site = [real_parents[i] for i in site]
    elif isinstance(site, int):
        site = real_parents[site]
    elif isinstance(site, str):
        pass
    else:
        site = "S0"

    if isinstance(label, tuple):
        label = [real_parents[i] for i in label]
    elif isinstance(label, int):
        label = real_parents[label]
    elif isinstance(label, str):
        pass
    else:
        label = "00"

    mxene_ = kwargs["func_MXene"](real_parents[0]) if "func_MXene" in kwargs else real_parents[0]
    base_name = kwargs["func_base_name"](base_name) if "func_base_name" in kwargs else base_name
    add = kwargs["func_add"](add) if "fun_add" in kwargs else add
    absorb = kwargs["func_absorb"](absorb) if "func_absorb" in kwargs else absorb
    doping = kwargs["func_doping"](doping) if "func_doping" in kwargs else doping

    new_pt = pre_parents
    new_pt.extend([mxene_, base_name, add, doping, absorb])
    if site_ is not None:
        site = kwargs["func_site"](site) if "func_site" in kwargs else site
        new_pt.append(site)
    if label_ is not None:
        label = kwargs["func_label"](label) if "func_label" in kwargs else label
        new_pt.append(label)

    new_pt = path.Path.joinpath(*new_pt)
    return new_pt


def _rmtree(path, protect=None):
    with os.scandir(path) as scandir_it:
        entries = list(scandir_it)

    for entry in entries:
        fullname = os.path.join(path, entry.name)
        if entry.name == protect or fullname == protect:
            pass
        else:
            is_dir = entry.is_dir(follow_symlinks=False)

            if is_dir:

                _rmtree(fullname, protect)

                os.rmdir(fullname)
            else:
                os.remove(fullname)


def copy_disk(old_pt: Union[str, path.Path], new_pt: Union[str, path.Path], file=True, disk=False,
              cover=False, remove=False):
    """
    Copy files,disks of old path to new path.
    复制旧路径下的子文件，子文件夹到新的路径下。

    Args:
        old_pt: (str, path.Path, os.PathLike,pathlib.Path), old path
        new_pt: (str, path.Path, os.PathLike,pathlib.Path), new path
        file: (bool), copy file.
        disk: (bool), copy sub-disk.
        cover: (bool), cover the exist data.
        remove: (bool), remove the old path.

    """
    if not isinstance(old_pt, path.Path):
        old_pt = path.Path(old_pt)
    if not isinstance(new_pt, path.Path):
        new_pt = path.Path(new_pt)
    ds = old_pt.dirs()

    if file and disk:
        shutil.copytree(old_pt, new_pt, dirs_exist_ok=cover)
        if remove:
            if old_pt in new_pt:
                _rmtree(old_pt, new_pt)
            else:
                shutil.rmtree(old_pt, )
    elif file:
        if len(ds) > 0:
            warnings.warn(f"{old_pt} is not the leaf node directory.", UserWarning)
            if cover:
                for i in old_pt.files():
                    shutil.copyfile(old_pt / i, new_pt / i)
                    if remove:
                        os.remove(old_pt / i)
            else:
                for i in old_pt.files():
                    if new_pt.isfile():
                        raise FileExistsError(f"{new_pt / i} is exist.")
                    else:
                        shutil.copyfile(old_pt / i, new_pt / i)
                        if remove:
                            os.remove(old_pt / i)
        else:
            shutil.copytree(old_pt, new_pt, dirs_exist_ok=cover)
            if remove:
                if old_pt in new_pt:
                    _rmtree(old_pt, new_pt)
                else:
                    shutil.rmtree(old_pt, )
    else:
        if len(ds) > 0:
            for i in ds:
                shutil.copytree(old_pt / i, new_pt / i, dirs_exist_ok=cover)
                if remove:
                    if old_pt in new_pt:
                        _rmtree(old_pt, new_pt)
                    else:
                        shutil.rmtree(old_pt, )
        else:
            pass


def check_structure_contcar(pt: Union[str, path.Path, os.PathLike, pathlib.Path], msg=None):
    """
    Check structure of contar and poscar are consistent.

    检查POSCAR，CONTCAR结构是否对应。

    Args:
        pt: (str, path.Path, os.PathLike,pathlib.Path), path
        msg:(list of str), message.

    Returns:
        res:(tuple), bool and msg list

    """
    if msg is None:
        msg = []
    msg.append("\nCheck Structure:")
    if not isinstance(pt, path.Path):
        pt = path.Path(pt)
    contcar = pt / "CONTCAR"
    poscar = pt / "POSCAR"
    if contcar.isfile() and poscar.isfile():
        try:
            pc = []
            for con in [contcar, poscar]:
                with open(con) as fc:
                    f1 = fc.readlines()
                    k = f1[5].replace(" ", "")
                    v = f1[6].replace(" ", "")
                    pc.append(k + v)
            if pc[0] == pc[1]:
                res = True, msg
            else:
                warnings.warn(f"contcar or poscar are different.",
                              UnicodeWarning)
                msg.append(f"contcar or poscar are different.")
                res = False, msg
        except IndexError:
            warnings.warn(f"contcar or poscar are empty.",
                          UnicodeWarning)
            msg.append(f"contcar or poscar are empty.")
            res = False, msg

    else:
        warnings.warn(f"Can't find contcar or poscar in path.",
                      UnicodeWarning)
        msg.append(f"Can't find contcar or poscar in path.")
        res = False, msg
    return res


def check_convergence(pt: Union[str, path.Path, os.PathLike, pathlib.Path], msg=None):
    """
    Check final energy.

    检查结构是否收敛。

    Args:
        pt: (str, path.Path, os.PathLike,pathlib.Path), path
        msg:(list of str), message.

    Returns:
        res:(tuple), bool and msg list

    """
    if msg is None:
        msg = []
    msg.append("\nCheck Convergence:")
    key_sentence0 = ' reached required accuracy - stopping structural energy minimisation\n'
    key_sentence1 = ' General timing and accounting informations for this job:\n'
    if not isinstance(pt, path.Path):
        pt = path.Path(pt)
    try:
        with open(pt / 'OUTCAR') as c:
            outcar = c.readlines()

        if key_sentence0 not in outcar[-40:] and key_sentence1 not in outcar[-20:]:
            warnings.warn(f"Not converge and not get the final energy.",
                          UnicodeWarning)
            msg.append(f"Not converge and not get the final energy..")
            res = False, msg
        else:
            res = True, msg

    except BaseException:
        warnings.warn(f"Error to read OUTCAR.",
                      UnicodeWarning)
        msg.append(f"Error to read OUTCAR.")
        res = False, msg

    return res


def get_recommend_path(pt: Union[str, path.Path, os.PathLike, pathlib.Path]):
    """
    Get recommend path.

    根据结构，获取推荐路径（不一定是对的，需要检查）。
    需要安装 mxene 包:
    pip install mxene

    Args:
        pt: (str, path.Path, os.PathLike,pathlib.Path), path
        msg:(list of str), message.

    Returns:
        res:(tuple), bool and msg list

    """
    if not isinstance(pt, path.Path):
        pt = path.Path(pt)

    msg = ["\nGET PATH:",
           f"Standard PATH:           .../MXenes/基底名称/   负载/  搀杂/   吸附/   等效位点(路径名)/ 标签",
           f"Standard PATH:           .../MXenes/base_name/add/doping/absorb/site_or_move_path/label\n", ]

    contcar = pt / "CONTCAR"
    if contcar.isfile():
        try:
            from mxene.mxenes import MXene
            mx = MXene.from_file(contcar)
            parents = pt.splitall()

            assert "MXenes" in parents
            mx_site = parents.index("MXenes")
            pre_parents = parents[:mx_site]

            new_pt = mx.get_disk(disk=path.Path.joinpath(*pre_parents), site_name="xx", equ_name="xx")
            new_pt = path.Path(new_pt)
            msg.append(f"Now PATH:                 {pt}")
            msg.append(f"Recommend PATH:    {new_pt}      (May Be Wrong, Use Carefully!)")
            res = True, msg
        except BaseException:
            msg.append("Can't to define path for this structure.")
            warnings.warn("Can't to define path for this structure.", UnicodeWarning)
            res = False, msg
    else:
        msg.append("Can't to define path for this structure, due to no CONTCAR.")
        warnings.warn("Can't to define path for this structure, due to no CONTCAR.", UnicodeWarning)
        res = False, msg
    return res


def check_pt(pt: Union[str, path.Path, os.PathLike, pathlib.Path], msg=None):
    """
    Check the path is in standard.
    检查路径是否符合标准

    Args:
        msg: (list), message.
        pt: (str, path.Path, os.PathLike,pathlib.Path), path of leaf node.

    Returns:
        res:(tuple), bool and msg list

    """
    if msg is None:
        msg = []
    msg.append("\nCheck PATH:")
    if not isinstance(pt, path.Path):
        pt = path.Path(pt)
    parents = pt.splitall()
    assert "MXenes" in parents
    mx_site = parents.index("MXenes")
    # pre_parents = parents[:mx_site]
    real_parents = parents[mx_site:]

    # nm_list = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I", "Te", "At"]
    # tm_list = ["Sc", "Y", "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Mn",
    #            # "Tc",
    #            "Re", "Fe", "Ru", "Os", "Co", "Rh", "Ir", "Ni", "Pd", "Pt", "Cu", "Ag", "Au", "Zn", "Cd"]
    # am_list = ["Al", "Ca", "Li", "Mg", "Na", "K", "Zn", "H", "OH", "O", "OOH"]
    # tem_list = ["O", "F", "OH", "Cl", None]
    # cn_list = ["C", "N"]
    # bm_list = ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"]

    t1 = any([i for i in bm_list if i in real_parents[1]])
    if not t1:
        msg.append(f"Error to not find the base metal in base structure name. Check: {real_parents[1]} ")
        warnings.warn(f"Error to not find the base metal in base structure name. Check: {real_parents[1]} ",
                      UnicodeWarning)
    t2 = any([i for i in cn_list if i in real_parents[1]])
    if not t2:
        msg.append(f"Error to not find the C/N in base structure name. Check: {real_parents[1]} ")
        warnings.warn(f"Error to find the C/N in base structure name. Check: {real_parents[1]} ", UnicodeWarning)

    t3 = any([i for i in tem_list if i in real_parents[1]])
    if not t3:
        warnings.warn(f"Warning to not find the terminal atoms in base structure name, "
                      f"check and make sure it is bare structure without terminal. Check: {real_parents[1]} ",
                      UnicodeWarning)
        msg.append(f"Not find the terminal atoms in base structure name, "
                   f"check and make sure it is bare structure without terminal. . Check: {real_parents[1]} ")

    t4 = any([i for i in tm_list if i in real_parents[2]]) if real_parents[2] != "no_add" else True
    if not t4:
        msg.append(f"Error to not find the add atoms in transition metal. Check: {real_parents[2]} ")
        warnings.warn(f"Error to not find the add atoms in transition metal. Check: {real_parents[2]} ", UnicodeWarning)

    t5 = any([i for i in tnm_list if i in real_parents[3]]) if real_parents[3] != "no_doping" else True
    if not t5:
        msg.append(
            f"Error to not find the doping atoms in non-transition or transition  metal. Check: {real_parents[3]} ")
        warnings.warn(
            f"Error to not find the doping atoms in non-transition or  transition metal. Check: {real_parents[3]} ",
            UnicodeWarning)

    t6 = any([i for i in am_list if i in real_parents[4]]) if real_parents[4] not in no_absorb_list else True
    if not t6:
        msg.append(f"Error to not find the absorb atoms/group in absorb atoms. Check: {real_parents[4]} ")
        warnings.warn(f"Error to not find the absorb atoms/group in absorb atoms. Check: {real_parents[4]} ",
                      UnicodeWarning)

    if len(real_parents) >= 6:
        t7 = any([i for i in site_list if i in real_parents[5]])
        msg.append(f"Warning to not find the standard site name. Check: {real_parents[5]} ")
        warnings.warn(f"Warning to not find the standard site name. Check: {real_parents[5]} ",
                      UnicodeWarning)
    else:
        ds = pt.dirs()
        if len(ds) == 0:
            t7 = True
        else:
            t7 = False
            msg.append(f"Error the disk contain sub-disk, which is illegal. Check: {real_parents[4]} ")
            warnings.warn(f"Error the disk contain sub-disk, which is illegal. Check: {real_parents[4]} ",
                          UnicodeWarning)

    if not t1 or not t2 or not t5 or not t6:
        res = False, msg
    else:
        res = True, msg

    return res


def find_leaf_path(root_pt: Union[str, path.Path, os.PathLike, pathlib.Path]) -> List[path.Path]:
    """
    Find the leaf path.
    获取所有叶节点路径.

    Args:
        root_pt: pt: (str, path.Path, os.PathLike,pathlib.Path), path.

    Returns:
        paths: (list), list of sub leaf path.

    """

    if not isinstance(root_pt, path.Path):
        root_pt = path.Path(root_pt)

    sub_disk = list(root_pt.walkdirs())

    par_disk = [i.parent for i in sub_disk]
    par_disk = list(set(par_disk))

    res = [i for i in sub_disk if i not in par_disk]
    return res


def check_mx_data(pt, ck_pt=True, ck_conver=True, ck_st=True, get_rcmd_pt=True, out_file="un_mark.txt"):
    """
    Check MXene data in total.

    总检查MXene数据.

    Args:
        pt: pt: (str, path.Path, os.PathLike,pathlib.Path), path.
        ck_pt: (bool), check path.
        ck_conver: (bool),check convergence.
        ck_st: (bool), check structure.
        get_rcmd_pt: (bool), check recommend path.
        out_file: (str), out file name.

    """
    msg = ["### Check MXene Data ###"]

    if ck_pt:
        c1, msg1 = check_pt(pt)
    else:
        c1, msg1 = True, []

    if ck_conver:
        c2, msg2 = check_convergence(pt)
    else:
        c2, msg2 = True, []

    if ck_st:
        c3, msg3 = check_structure_contcar(pt)
    else:
        c3, msg3 = True, []

    if get_rcmd_pt:
        c4, msg4 = get_recommend_path(pt)
    else:
        c4, msg4 = True, []

    if c1 and c2 and c3:
        if (pi / out_file).isfile():
            os.remove(pi / out_file)
    else:
        if out_file is None:
            pass
        else:
            msg.extend(msg1)
            msg.extend(msg2)
            msg.extend(msg3)
            msg.extend(msg4)
            with open(pt / out_file, "w") as f:
                msg = "\n".join(msg)
                f.write(msg)
            print("Error report are stored in un_mark.txt")


if __name__ == "__main__":

    # 移动替换部分,用于原始数据初始路径移动（谨慎使用）
    # pt = r"E:\MXenes_raw_data\MXenes"
    # #
    # paths = find_leaf_path(pt)
    #
    # for pi in paths:
    #
    #     npi = path_regroup(pi,base_name=(1,2,3),add=4,doping=None,absorb=None,site=5,label="00",
    #                     func_site= lambda x:f"AS{x}",
    #                     func_base_name =  lambda x: f"{x.split('-')[-2]}2{x.split('-')[-1]}O2",)
    #
    #     npi = npi.replace("MXene", "MXene_temp/MXene")
    #     print(pi)
    #     print(npi)
    #     copy_disk(old_pt=pi, new_pt=npi, file = True, disk = False, cover = False,remove=False)
    #     break

    # 检查路径部分，用于检查路径是否合格 ###
    pt = r"E:\MXenes_raw_data\MXenes"
    paths = find_leaf_path(pt)

    for pi in paths:
        try:
            npi = path_regroup(pi, base_name=1,add=2,doping=3,absorb=4,site=5,label=6,)
        except AssertionError:
            try:
                npi = path_regroup(pi, base_name=1,add=2,doping=3,absorb=4,site=5,label=None)
                npi = npi/"00"
            except AssertionError:
                npi = path_regroup(pi,base_name=1, add=2,doping=3,absorb=4,site=None,label=None,)

        if pi==npi:
            pass
        else:
            print(pi)
            print(npi)

    # 标记部分(可以重复运行)，用于检查数据是否有效 ###
    pt = r"E:\MXenes_raw_data\MXenes\Mo2CO2"
    paths = find_leaf_path(pt)

    for pi in paths:
        check_mx_data(pi, ck_pt=True, ck_conver=True, ck_st=True, get_rcmd_pt=True, out_file="un_mark.txt")

    temp = []
    for pi in paths:
        if (pi / "un_mark.txt").isfile():
            print(pi)
            temp.append(pi)
            # if 'pure_static' in pi:
            #     shutil.copyfile(pi / "CONTCAR" ,pi / "POSCAR")
    with open("paths.temp", "w") as f:
        f.write("\n".join(temp))
