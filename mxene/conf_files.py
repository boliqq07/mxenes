import functools
import itertools
import os
from typing import Union

from pymatgen.core import Structure
from pymatgen.io.vasp import Kpoints, Potcar, Poscar

# 1. INCAR file

opt_incar = """
# Basic
PREC = Normal
ISPIN = 2
#LORBIT=11

# Initialization
ISTART = 0
ICHARG = 2

# Electronic relaxation
ENCUT = 600
NELMIN = 5
NELM = 200
ALGO = F
EDIFF = 1E-04
EDIFFG = -0.001

# Ionic relaxation
IBRION = 2
ISIF = 3
NSW = 200

# Density of states related
ISMEAR = 0
SIGMA = 0.02
NEDOS = 301

# Output
LCHARG = .FALSE.
LWAVE = .FALSE.

# Specific
IVDW = 12

IOPTCELL = 1 1 0 1 1 0 0 0 0"""

neb_incar = """
# Basic
PREC = Normal
ISPIN = 2

# Initialization
ISTART = 0
ICHARG = 2

# Electronic relaxation
ENCUT = 500
NELMIN = 5
NELM = 200
ALGO = Fast
EDIFF = 1E-04

# Ionic relaxation
IBRION = 3
ISIF = 2
POTIM = 0
NSW = 300
EDIFFG = -0.001

# Density of states related
ISMEAR = 0
SIGMA = 0.02
#NEDOS = 301

# Output
LCHARG = .FALSE.
LWAVE = .FALSE.

# Specific
IVDW = 12
#NCORE = 4
#NPAR = 8
ISYM = 0
LREAL = Auto

#CI-NEB caculations (Append "#" before these flags for other caculations):
ICHAIN = 0             # 0-NEB,1-Dynamic matrix,2-Dimer,3-Lanczos
IMAGES = 3             # Number of images,excluding endpoints.
SPRING = -5            # Spring force between images
LCLIMB = .TRUE.        # Flag to ture on the climbing image algorithm
LTANGENTOLD = .FALSE.  # Flag to turn on the old central difference tangent
LDNEB  = .FALSE.       # Flag to turn on modified doubble nudging
#
##Aggressive optimizers:
IOPT   = 3            # 0-IBRION(Default),1-LBFGS,2-CG,3-QM,4-SD,7-FIRE; Set IBRION=3,POTIM=0 
                      # when turn on it.
                      # After the force < 1 eV/ang, an aggressive optimizer con
                      # verges better.



"""

static_incar = """
# Basic
PREC = Normal
ISPIN = 2
LORBIT=11

# Initiaization
ISTART = 0
ICHARG = 2

# Electronic relaxation
ENCUT = 600
NELMIN = 5
NELM = 200
ALGO = N
EDIFF = 1E-05
EDIFFG = -0.001

# Ionic relaxation
IBRION = -1
ISIF = 2
NSW = 1

# Density of states related
ISMEAR = 0
SIGMA = 0.02
NEDOS = 301

# Output
LCHARG = .FALSE.
LWAVE = .FALSE.
LAECHG= .FALSE.

# Specific
IVDW = 12

IOPTCELL = 1 1 0 1 1 0 0 0 0
"""

# 2. run file

run_192_168_3_6_wang = """
#PBS -S /bin/bash
#PBS -N vasp
#PBS -l nodes=1:ppn=12
#PBS -l walltime=120:00:00
#PBS -q master

source /opt/intel/oneapi/mkl/latest/env/vars.sh intel64
source /opt/intel/oneapi/compiler/latest/env/vars.sh intel64
source /opt/intel/oneapi/mpi/2021.4.0/env/vars.sh intel64

export PATH=/opt/software/vasp.5.4.4/bin:$PATH

cd ${PBS_O_WORKDIR}

mpirun -np 12 vasp_std


"""


run_gjj_wang = """

#!/bin/sh
#JSUB -J CX_W       
#JSUB -n 60
#JSUB -R span[ptile=60]        
#JSUB -q cpu
#JSUB -o out.%J                  
#JSUB -e err.%J

source ~/intel/oneapi/mkl/2022.0.2/env/vars.sh intel64
source ~/intel/oneapi/compiler/2022.0.2/env/vars.sh intel64
source ~/intel/oneapi/mpi/2021.5.1/env/vars.sh intel64
export PATH=~/app/vasp.5.4.4/bin/:$PATH

ulimit -s 5120000

source /beegfs/jhinno/unischeduler/conf/unisched
########################################################
#   $JH_NCPU:         Number of CPU cores              #
#   $JH_HOSTFILE:     List of computer hostfiles       #
########################################################

mpirun -np $JH_NCPU -machinefile $JH_HOSTFILE vasp_std  > vasp.log

"""

run_240_all = """
#BSUB -q normal
#BSUB -n 24
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -R "span[ptile=24]"
PATH=/share/app/vasp.5.4.4-fix-neb/bin/:$PATH
source /share/intel/intel/bin/compilervars.sh intel64
mpirun vasp_std > log.log
"""

run_240_all_neb = """
#BSUB -q normal
#BSUB -n 72
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -R "span[ptile=24]"
PATH=/share/app/vasp.5.4.4-fix-neb/bin/:$PATH
source /share/intel/intel/bin/compilervars.sh intel64
mpirun vasp_std > log.log
"""

run_gjj_sk = """
#!/bin/sh
#JSUB -J skk       
#JSUB -n 64
#JSUB -R span[ptile=64]        
#JSUB -q cpu 
#JSUB -o out.%J                  
#JSUB -e err.%J                  

source /beegfs/home/kesong/intel/bin/compilervars.sh intel64
source /beegfs/home/kesong/intel/mkl/bin/mklvars.sh intel64

ulimit -s 5120000

source /beegfs/jhinno/unischeduler/conf/unisched
########################################################
#   $JH_NCPU:         Number of CPU cores              #
#   $JH_HOSTFILE:     List of computer hostfiles       #
########################################################
export PATH=/beegfs/home/kesong/app/vasp.5.4.4-vtst/bin/:$PATH
mpirun -np $JH_NCPU -machinefile $JH_HOSTFILE vasp_std  > vasp.log

"""

gpaw_relax_py = '''
from ase.optimize import QuasiNewton
from gpaw import GPAW, PW
from mxene.function2 import fixed_atoms
from mxene.mxene import MXene, aaa

mx = MXene.from_file("POSCAR")

atoms = mx.to_ase_atoms()
atoms = fixed_atoms(atoms, fixed_type=0.58, doping_fixed_type="line", doping_direction=(0, 0, 1),
                    coords_are_cartesian=False)

convergence={'energy': 0.001,  # eV / electron
 'density': 5.0e-4,  # electrons / electron
 'bands': 'occupied',
             }

atoms.calc = GPAW(xc='PBE',
                  mode=PW(300),
                  convergence=convergence,
                  )

relax = QuasiNewton(atoms)

try:
    relax.run(fmax=0.1)

    st = aaa.get_structure(atoms)
    st.to("poscar","POSCAR")
    mx.to("poscar","old_POSCAR")
except BaseException:
    pass
'''

run_gjj_wang_gpaw_relax = """
#!/bin/sh
#JSUB -J gpaw       
#JSUB -n 60
#JSUB -R span[ptile=60]        
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

$ mpirun -np $JH_NCPU -machinefile $JH_HOSTFILE gpaw python gpaw_relax.py

"""


def cmd_sys(cmds):
    for i in cmds:
        os.system(i)


def write_batch():
    with open("static_INCAR", "w") as f:
        f.writelines(static_incar)
    with open("opt_INCAR", "w") as f:
        f.writelines(opt_incar)
    with open("neb_INCAR", "w") as f:
        f.writelines(neb_incar)

    with open("gjj_skk_cpu.run", "w") as f:
        f.writelines(run_gjj_sk)
    with open("gjj_wang.run", "w") as f:
        f.writelines(run_gjj_wang)

    with open("gjj_wang_gpaw_relax.run", "w") as f:
        f.writelines(run_gjj_wang_gpaw_relax)
    with open("gpaw_relax.py", "w") as f:
        f.writelines(gpaw_relax_py)

    path = os.getcwd()

    cmd_sys(["mgetool makebatch -cmd 'jsub < $(find -name *.run)'",
             f"mgetool makebatch -cmd 'cd .. \nmv fin_static \ncp -r fin_opt fin_static \ncp fin_opt/CONTCAR fin_static/Y-Cr-Mo-POSCAR \ncp {path}/static_INCAR fin_static/INCAR' -o static_fin.sh",
             f"mgetool makebatch -cmd 'cd .. \nmv ini_static \ncp -r ini_opt ini_static \ncp ini_opt/CONTCAR ini_static/Y-Cr-Mo-POSCAR \ncp {path}/static_INCAR ini_static/INCAR' -o static_ini.sh",
             f"mgetool makebatch -cmd 'cd .. \nmv pure_static \ncp -r pure_opt pure_static \ncp pure_opt/CONTCAR pure_static/Y-Cr-Mo-POSCAR \ncp {path}/static_INCAR pure_static/INCAR' -o static_pure.sh",
             f"mgetool makebatch -cmd 'cp {path}/cpu.run ../ini_opt' -o cpbatch.sh",
             f"mgetool makebatch -cmd 'cd .. \nnebmake.pl ini_static/CONTCAR fin_static/CONTCAR 3 \ncp ini_static/OUTCAR 00/OUTCAR \ncp fin_static/OUTCAR 04/OUTCAR \ncp ini_static/KPOINTS KPOINTS "
             f"\ncp ini_static/POTCAR POTCAR \ncp {path}/neb_cpu.run neb_cpu.run' -o nebbatch.sh"
             ])


# 3.POT-database

potpath = r"POT-database"  # POT-database should be offered


def check_potcar(potpath):
    try:
        potpaths = os.listdir(potpath)
        sym_potcar_map = {}
        for i in potpaths:
            with open(potpath + "/" + i) as f:
                te = f.readlines()
                text = "".join(te)
                i = i.split("-")[0]
                i = i.split("_")[0]
                sym_potcar_map.update({i: text})
    except BaseException:
        sym_potcar_map = None
    return sym_potcar_map


sym_potcar_map = check_potcar(potpath=r"POT-database")


@functools.lru_cache(200)
def get_potcar_lru(sym):
    return Potcar(sym, sym_potcar_map=sym_potcar_map, )


def get_potcar(poscar: Union[Poscar, Structure], sym_potcar_map=sym_potcar_map):
    if isinstance(poscar, Structure):
        syms = [site.specie.symbol for site in poscar]
        site_symbols = [a[0] for a in itertools.groupby(syms)]
        sym = tuple(site_symbols)
    else:
        sym = tuple(poscar.site_symbols)

    if sym_potcar_map is not None:
        POTCAR = Potcar(sym, sym_potcar_map=sym_potcar_map, )
    else:
        POTCAR = Potcar(sym)
    return POTCAR


"""

sym_potcar_map = check_potcar(potpath=r"POT-database")
potcar = get_potcar(poscar, sym_potcar_map=sym_potcar_map)


# or quick method

sym_potcar_map = check_potcar(potpath=r"POT-database")

@functools.lru_cache(200)
def get_potcar_lru(sym):
    return Potcar(sym, sym_potcar_map=sym_potcar_map)
potcar = get_potcar_lru(tuple(poscar.site_symbols), sym_potcar_map=sym_potcar_map)

"""

# 4.kpoints

kpoints = Kpoints(kpts=((3, 3, 1),))
