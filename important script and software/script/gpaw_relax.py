

from ase.constraints import ExpCellFilter
from ase.io import Trajectory
from ase.optimize import QuasiNewton
from gpaw import GPAW, PW
from mxene.functions2 import fixed_atoms
from mxene.mxenes import MXene, aaa
import numpy as np

def search_space(*arg):

    meshes = np.meshgrid(*arg)
    meshes = [_.ravel() for _ in meshes]
    meshes = np.array(meshes).T
    return meshes

mx = MXene.from_file("POSCAR")

atoms = mx.to_ase_atoms()
atoms = fixed_atoms(atoms, fixed_type=0.00, doping_fixed_type="line", doping_direction=(0, 0, 1),
                    coords_are_cartesian=False)

convergence = {'energy': 0.005,  # eV / electron
               'density': 5.0e-4,  # electrons / electron
               'bands': 'occupied',
               }

# kpts={'size': (3, 3, 1)}

atoms.calc = GPAW(xc='PBE',
                  # kpts=kpts,
                  mode=PW(340),
                  convergence=convergence,
                  )

ecf = ExpCellFilter(atoms)

relax = QuasiNewton(ecf)

try:
    traj = Trajectory('struc.traj', 'w', atoms)
    relax.attach(traj)

    relax.run(fmax=0.05)

    st = aaa.get_structure(atoms)
    st.to("poscar", "POSCAR")
    mx.to("poscar", "old_POSCAR")
except BaseException:
    pass
