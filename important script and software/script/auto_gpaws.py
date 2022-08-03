import os.path
from shutil import copyfile

path= '.'

gpaw_relax_all = "gpaw_relax_all.py"

temp = "paths_gpaw.temp"

run_str = '''
#!/bin/sh
#JSUB -J gpaw       
#JSUB -n 64
#JSUB -R span[ptile=64]        
#JSUB -q cpu
#JSUB -o out.%J                  
#JSUB -e err.%J

source ~/intel/oneapi/mkl/2022.0.2/env/vars.sh intel64
source ~/intel/oneapi/compiler/2022.0.2/env/vars.sh intel64
source ~/intel/oneapi/mpi/2021.5.1/env/vars.sh intel64

ulimit -s 5120000

source /beegfs/jhinno/unischeduler/conf/unisched
########################################################
#   $JH_NCPU:         Number of CPU cores              #
#   $JH_HOSTFILE:     List of computer hostfiles       #
########################################################

mpirun -np $JH_NCPU -machinefile $JH_HOSTFILE gpaw python gpaw_relax_all.py
'''

if os.path.isfile(gpaw_relax_all) and os.path.isfile(temp):
    os.system(f"split -l 100 -d {temp} paths_gpawi.temp")
    fs = os.listdir()
    fs = [i for i in fs if "paths_gpawi.temp" in i ]
    lfs = len(fs)
    assert lfs<100
    for i in range(lfs):
        if i>=10:
            newpy = f"gpaw_relax_all{i}.py"
            newrun = f"gjj_wang_gpaw_relax_cpu{i}.run"
        else:
            newpy = f"gpaw_relax_all0{i}.py"
            newrun = f"gjj_wang_gpaw_relax_cpu0{i}.run"

        new_run_str=run_str.replace("gpaw_relax_all.py", newpy)
        with open(newrun, "w") as f:
            f.write(new_run_str)
        copyfile(gpaw_relax_all, newpy)




