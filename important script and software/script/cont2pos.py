#!/bin/bash
import os.path
from pathlib import Path
from shutil import copyfile, move

with open("paths.temp") as f:
    ls = f.readlines()
ls = [Path(i.rstrip()) for i in ls]

for i in ls:
    if (i/"CONTCAR").is_file():
        if os.path.getsize(i/"CONTCAR")>0:
            try:
                print(f"Move CONTCAR to POSCAR for {i}")
                move(i/"POSCAR", i/"OLD_POSCAR")
                copyfile(i/"CONTCAR", i/"POSCAR")
            except:
                print(f"Error for: {i}")

    if not (i/"POSCAR").is_file() and not (i/"CONTCAR").is_file() and (i / "OLD_POSCAR").is_file() and os.path.getsize(i/"OLD_POSCAR")>0:
        move(i/"OLD_POSCAR", i/"POSCAR")
        print(f"Restore OLD_POSCAR to POSCAR for {i}")
