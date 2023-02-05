# import warnings
#
# import numpy as np
# from numpy.linalg import eigh
# from pymatgen.core import Structure
#
#
# class Optimizer:
#     """Base-class for all structure optimization classes."""
#
#     # default maxstep for all optimizers
#     defaults = {'maxstep': 0.2}
#
#     def __init__(
#         self,
#         atoms,
#         restart,
#         logfile,
#         trajectory,
#         master=None,
#         append_trajectory=False,
#         force_consistent=False,
#     ):
#         """Structure optimizer object.
#
#         Parameters:
#
#         atoms: Atoms object
#             The Atoms object to relax.
#
#         restart: str
#             Filename for restart file.  Default value is *None*.
#
#         logfile: file object or str
#             If *logfile* is a string, a file with that name will be opened.
#             Use '-' for stdout.
#
#         trajectory: Trajectory object or str
#             Attach trajectory object.  If *trajectory* is a string a
#             Trajectory will be constructed.  Use *None* for no
#             trajectory.
#
#         master: boolean
#             Defaults to None, which causes only rank 0 to save files.  If
#             set to true,  this rank will save files.
#
#         append_trajectory: boolean
#             Appended to the trajectory file instead of overwriting it.
#
#         force_consistent: boolean or None
#             Use force-consistent energy calls (as opposed to the energy
#             extrapolated to 0 K).  If force_consistent=None, uses
#             force-consistent energies if available in the calculator, but
#             falls back to force_consistent=False if not.
#         """
#
#         # initialize attribute
#         self.fmax = None
#
#
#     def irun(self, fmax=0.05, steps=None):
#         """ call Dynamics.irun and keep track of fmax"""
#         self.fmax = fmax
#         if steps:
#             self.max_steps = steps
#         return self.irun(self)
#
#     def run(self, fmax=0.05, steps=None):
#         """ call Dynamics.run and keep track of fmax"""
#         self.fmax = fmax
#         if steps:
#             self.max_steps = steps
#         return Dynamics.run(self)
#
#     def converged(self, forces=None):
#         """Did the optimization converge?"""
#         if forces is None:
#             forces = self.atoms.get_forces()
#         return (forces ** 2).sum(axis=1).max() < self.fmax ** 2
#
#
#     def irun(self):
#         """Run dynamics algorithm as generator. This allows, e.g.,
#         to easily run two optimizers or MD thermostats at the same time.
#
#         Examples:
#         >>> opt1 = BFGS(atoms)
#         >>> opt2 = BFGS(StrainFilter(atoms)).irun()
#         >>> for _ in opt2:
#         >>>     opt1.run()
#         """
#
#         # compute initial structure and log the first step
#         self.atoms.get_forces()
#
#         # yield the first time to inspect before logging
#         yield False
#
#         if self.nsteps == 0:
#             self.log()
#             self.call_observers()
#
#         # run the algorithm until converged or max_steps reached
#         while not self.converged() and self.nsteps < self.max_steps:
#
#             # compute the next step
#             self.step()
#             self.nsteps += 1
#
#             # let the user inspect the step and change things before logging
#             # and predicting the next step
#             yield False
#
#         # finally check if algorithm was converged
#         yield self.converged()
#
#
# class BFGS:
#     # default parameters
#     defaults = {'alpha': 70.0}
#
#     def __init__(self, structure:Structure, maxstep=None, alpha=None,cale=None):
#         if maxstep is None:
#             self.maxstep = self.defaults['maxstep']
#         else:
#             self.maxstep = maxstep
#
#         if self.maxstep > 1.0:
#             warnings.warn('You are using a *very* large value for '
#                           'the maximum step size: %.1f Å' % maxstep)
#
#         if alpha is None:
#             self.alpha = self.defaults['alpha']
#         else:
#             self.alpha = alpha
#
#         self.fmax = None
#
#         self.structure = structure
#         self.cale=cale
#
#
#     def initialize(self):
#         # initial hessian
#         self.H0 = np.eye(3 * len(self.structure)) * self.alpha
#
#         self.H = None
#         self.r0 = None
#         self.f0 = None
#
#     def get_force(self):
#         return self.cale.get_force()
#
#     def step(self, f=None):
#
#         if f is None:
#             f = self.get_force()
#
#         r = self.structure.cart_coords
#         f = f.reshape(-1)
#         self.update(r.flat, f, self.r0, self.f0)
#         omega, V = eigh(self.H)
#
#         dr = np.dot(V, np.dot(f, V) / np.fabs(omega)).reshape((-1, 3))
#         steplengths = (dr**2).sum(1)**0.5
#         dr = self.determine_step(dr, steplengths)
#         self.structure
#         atoms.set_positions(r + dr)
#         self.r0 = r.flat.copy()
#         self.f0 = f.copy()
#
#     def determine_step(self, dr, steplengths):
#         """Determine step to take according to maxstep
#
#         Normalize all steps as the largest step. This way
#         we still move along the eigendirection.
#         """
#         maxsteplength = np.max(steplengths)
#         if maxsteplength >= self.maxstep:
#             scale = self.maxstep / maxsteplength
#             dr *= scale
#
#         return dr
#
#     def update(self, r, f, r0, f0):
#         if self.H is None:
#             self.H = self.H0
#             return
#         dr = r - r0
#
#         if np.abs(dr).max() < 1e-7:
#             # Same configuration again (maybe a restart):
#             return
#
#         df = f - f0
#         a = np.dot(dr, df)
#         dg = np.dot(self.H, dr)
#         b = np.dot(dr, dg)
#         self.H -= np.outer(df, df) / a + np.outer(dg, dg) / b
