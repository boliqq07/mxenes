import gc
import os
import time

import copy
from ase.cell import Cell
from ase.constraints import ExpCellFilter
from ase.optimize import QuasiNewton
from ase.parallel import paropen
from gpaw import GPAW, PW
from mxene.functions2 import fixed_atoms
from mxene.mxenes import MXene, aaa
import numpy as np


def search_space(*arg):
    meshes = np.meshgrid(*arg)
    meshes = [_.ravel() for _ in meshes]
    meshes = np.array(meshes).T
    return meshes


nu = str(__file__)
file = f"paths_gpawi.temp{nu[-5:-3]}"
# file = "paths_gpaw.temp"

with open(file) as f:
    words = f.readlines()

words = [i.replace("\n", "") for i in words]
words.reverse()
old_path = os.getcwd()

while len(words) > 0:

    i = words.pop()
    os.chdir(i)
    try:
        mx = MXene.from_file("POSCAR")

        atoms = mx.to_ase_atoms()
        atoms = fixed_atoms(atoms, fixed_type=0.0, doping_fixed_type="line", doping_direction=(0, 0, 1),
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
            relax.run(fmax=0.05)

            # relax lattice
            # ei = atoms.get_potential_energy()
            # cell0 = atoms.cell
            # cellpar0 = cell0.cellpar()
            # a, b, c, alpha, beta, gamma = cellpar0
            # cells = []
            # energys = []
            # with paropen('gpaw.log', 'w') as ff:
            #     ff.writelines([f"{time.time()}\n", f"{ei}\n"])
            # abps = search_space(np.linspace(-0.02, 0.02, 3), np.linspace(-0.02, 0.02, 3))
            # for i, j in abps:
            #     celli = Cell.fromcellpar([(1 + i) * a, (1 + j) * b, c, alpha, beta, gamma])
            #     cells.append(celli)
            #     atoms.cell = celli
            #     enei = atoms.get_potential_energy()
            #     energys.append(enei)
            #     with paropen('gpaw.log', 'a+') as ff:
            #         ff.writelines([f"{time.time()}\n", f"{enei}\n", f"{i},{j}\n"])
            # index = np.argmin(np.array(energys))
            # atoms.cell = cells[index]
            # end relax lattice

            st = aaa.get_structure(atoms)
            st.to("poscar", "POSCAR")
            mx.to("poscar", "old_POSCAR")

            del atoms
            del st
            del mx
            del relax
            del ecf

            gc.collect()

            to_word = i
            left_word = words

            os.chdir(old_path)
            with paropen(f"Succeed_{file}", 'a+') as ff:
                ff.write(i + "\n")

        except BaseException:
            os.chdir(old_path)
            with paropen(f"Failed_{file}", 'a+') as ff:
                ff.write(i + "\n")

    except BaseException:
        os.chdir(old_path)

        with paropen(f"Failed_{file}", 'a+') as ff:
            ff.write(i + "\n")

    os.chdir(old_path)
    words2 = copy.deepcopy(words)
    words2.reverse()
    with paropen(file, 'w') as ff:
        ff.write("\n".join(words2))
