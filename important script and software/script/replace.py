# -*- coding: utf-8 -*-

# @Time  : 2023/3/15 14:43
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

# !/bin/bash
import os.path
import re
import shutil
from pathlib import Path
from shutil import move

with open("paths.temp") as f:
    ls = f.readlines()
ls = [Path(i.rstrip()) for i in ls]

old_cmd = 'NCORE = *'
new_cmd = 'NCORE = 16'

old_cmd = old_cmd.replace("*", ".*")
match_patten = re.compile(old_cmd)

for i in ls:
    if (i / "INCAR").is_file():
        shutil.copyfile(i / "INCAR", i / "INCAR_bak")
        with open(i / "INCAR", "r") as f:
            w = f.readlines()
        w = "".join(w)
        res = re.findall(match_patten, w)
        print(res)
        if res is None or len(res) == 0:
            w = "\n".join([w, new_cmd])
        elif len(res)==1:
            old_cmd_2 = res[0]
            w = w.replace(old_cmd_2, new_cmd)
        elif len(res)>1:
            old_cmd_2 = res
            for oi in old_cmd_2:
                w = w.replace(oi, "")
            w = "\n".join([w, new_cmd])

        with open(i / "INCAR", "w") as f2:
            f2.write(w)
        print(f"Refresh INCAR with ({new_cmd}) for {i}")