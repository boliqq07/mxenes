# -*- coding: utf-8 -*-

# @Time     : 2021/10/22 16:33
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from pymatgen.core import Element
from pymatgen.electronic_structure.core import Spin, OrbitalType
from scipy.ndimage.filters import gaussian_filter1d


class DOSPlotter:

    def __init__(
            self,
            vb_energy_range=4,
            cb_energy_range=4,
            fixed_cb_energy=False,
            egrid_interval=1,
            font="Arial",
            axis_fontsize=20,
            tick_fontsize=15,
            legend_fontsize=16,
            dos_legend="best",
            rgb_legend=True,
            fig_size=(11, 8.5),
            sigma=0.1,
    ):
        """
        Instantiate plotter settings.

        Args:
            vb_energy_range (float): abs energy in eV to show of valence bands
            cb_energy_range (float): abs energy in eV to show of conduction bands
            fixed_cb_energy (bool): If true, the cb_energy_range will be interpreted
                as constant (i.e., no gap correction for cb energy)
            egrid_interval (float): interval for grid marks
            font (str): font family
            axis_fontsize (float): font gen_num for axis
            tick_fontsize (float): font gen_num for axis tick labels
            legend_fontsize (float): font gen_num for legends
            dos_legend (str): matplotlib string location for legend or None
            rgb_legend (bool): (T/F) whether to draw RGB triangle/bar for element proj.
            fig_size(tuple): dimensions of figure gen_num (width, height)
        """
        self.vb_energy_range = vb_energy_range
        self.cb_energy_range = cb_energy_range
        self.fixed_cb_energy = fixed_cb_energy
        self.egrid_interval = egrid_interval
        self.font = font
        self.axis_fontsize = axis_fontsize
        self.tick_fontsize = tick_fontsize
        self.legend_fontsize = legend_fontsize
        self.dos_legend = dos_legend
        self.rgb_legend = rgb_legend
        self.fig_size = fig_size
        self.sigma = sigma
        plt.rcParams['font.sans-serif'] = "Arial"

    def smear(self, dens, sigma=None):
        """
        Returns the Dict representation of the densities, {Spin: densities},
        but with a Gaussian smearing of std dev sigma applied about the fermi
        level.

        Args:
            sigma: Std dev of Gaussian smearing function.

        Returns:
            Dict of Gaussian-smeared densities.
        """
        if sigma is None and self.sigma is None:
            return dens
        elif sigma is None and self.sigma is not None:
            return gaussian_filter1d(dens, self.sigma)
        else:
            return gaussian_filter1d(dens, sigma)

    def get_plot(self, dos=None):
        """
        Get a matplotlib plot object.
        Args:
            dos (Dos): the Dos to plot. Projection data must exist (i.e.,
                CompleteDos) for projected plots.

        Returns:
            matplotlib.pyplot object on which you can call commands like show()
            and savefig()
        """

        import matplotlib.pyplot as mplt

        # make sure the user-specified band structure projection is valid

        elements = [e.symbol for e in dos.structure.composition.elements]

        # specify energy range of plot
        emin = -self.vb_energy_range
        emax = self.cb_energy_range

        # initialize all the k-point labels and k-point x-distances for bs plot
        xlabels = []  # all symmetry point labels on x-axis
        xlabel_distances = []  # positions of symmetry point x-labels

        # set up bs and dos plot

        fig = mplt.figure(figsize=self.fig_size)
        fig.patch.set_facecolor("white")

        dos_ax = mplt.subplot(121)
        self.dos_projection = "elements"
        # set basic axes limits for the plot

        dos_ax.set_ylim(emin, emax)

        dos_ax.set_yticks(np.arange(emin, emax + 1e-5, self.egrid_interval))
        dos_ax.set_yticklabels([])
        dos_ax.grid(color=[0.5, 0.5, 0.5], linestyle="dotted", linewidth=1)

        # renormalize the DOS energies to Fermi level
        dos_energies = [e - dos.efermi for e in dos.energies]

        # Plot the DOS and projected DOS
        for spin in (Spin.up, Spin.down):
            if spin in dos.densities:
                # plot the total DOS
                dos_densities = dos.densities[spin] * int(spin)
                dos_densities = self.smear(dos_densities)
                label = "Total" if spin == Spin.up else None
                dos_ax.plot(dos_densities, dos_energies, color=(0.6, 0.6, 0.6), label=label)
                dos_ax.fill_betweenx(
                    dos_energies,
                    0,
                    dos_densities,
                    color=(0.7, 0.7, 0.7),
                    facecolor=(0.7, 0.7, 0.7),
                )

                if self.dos_projection is None:
                    pass

                elif self.dos_projection.lower() == "elements":
                    # plot the atom-projected DOS
                    colors = ["b", "r", "g", "m", "y", "c", "k", "w"]
                    el_dos = dos.get_element_dos()
                    for idx, el in enumerate(elements):
                        dos_densities = el_dos[Element(el)].densities[spin] * int(spin)
                        dos_densities = self.smear(dos_densities)
                        label = el if spin == Spin.up else None
                        dos_ax.plot(
                            dos_densities,
                            dos_energies,
                            color=colors[idx],
                            label=label,
                        )

                elif self.dos_projection.lower() == "orbitals":
                    # plot each of the atomic projected DOS
                    colors = ["b", "r", "g", "m"]
                    spd_dos = dos.get_spd_dos()
                    for idx, orb in enumerate([OrbitalType.s, OrbitalType.p, OrbitalType.d, OrbitalType.f]):
                        if orb in spd_dos:
                            dos_densities = spd_dos[orb].densities[spin] * int(spin)
                            dos_densities = self.smear(dos_densities)
                            label = orb if spin == Spin.up else None
                            dos_ax.plot(
                                dos_densities,
                                dos_energies,
                                color=colors[idx],
                                label=label,
                            )

            # get index of lowest and highest energy being plotted, used to help auto-scale DOS x-axis
            emin_idx = next(x[0] for x in enumerate(dos_energies) if x[1] >= emin)
            emax_idx = len(dos_energies) - next(x[0] for x in enumerate(reversed(dos_energies)) if x[1] <= emax)

            # determine DOS x-axis range
            dos_xmin = (
                0 if Spin.down not in dos.densities else -max(dos.densities[Spin.down][emin_idx: emax_idx + 1] * 1.05)
            )
            dos_xmax = max([max(dos.densities[Spin.up][emin_idx:emax_idx]) * 1.05, abs(dos_xmin)])

            # set up the DOS x-axis and add Fermi level line
            dos_ax.set_xlim(dos_xmin, dos_xmax)
            dos_ax.set_xticklabels([])
            dos_ax.hlines(y=0, xmin=dos_xmin, xmax=dos_xmax, color="k", lw=2)
            dos_ax.set_xlabel("PDOS for each atom", fontsize=self.axis_fontsize, family=self.font)

        # add legend for DOS
        if dos and self.dos_legend:
            dos_ax.legend(
                fancybox=True,
                prop={"family": self.font, "size": self.legend_fontsize},
                loc=self.dos_legend,
            )

        dos_ax = mplt.subplot(122)
        self.dos_projection = "orbitals"
        # set basic axes limits for the plot

        dos_ax.set_ylim(emin, emax)

        dos_ax.set_yticks(np.arange(emin, emax + 1e-5, self.egrid_interval))
        dos_ax.set_yticklabels([])
        dos_ax.grid(color=[0.5, 0.5, 0.5], linestyle="dotted", linewidth=1)

        # renormalize the DOS energies to Fermi level
        dos_energies = [e - dos.efermi for e in dos.energies]

        # Plot the DOS and projected DOS
        for spin in (Spin.up, Spin.down):
            if spin in dos.densities:
                # plot the total DOS
                dos_densities = dos.densities[spin] * int(spin)
                dos_densities = self.smear(dos_densities)
                label = "Total" if spin == Spin.up else None
                dos_ax.plot(dos_densities, dos_energies, color=(0.6, 0.6, 0.6), label=label)
                dos_ax.fill_betweenx(
                    dos_energies,
                    0,
                    dos_densities,
                    color=(0.7, 0.7, 0.7),
                    facecolor=(0.7, 0.7, 0.7),
                )

                if self.dos_projection is None:
                    pass

                elif self.dos_projection.lower() == "elements":
                    # plot the atom-projected DOS
                    colors = ["b", "r", "g", "m", "y", "c", "k", "w"]
                    el_dos = dos.get_element_dos()
                    for idx, el in enumerate(elements):
                        dos_densities = el_dos[Element(el)].densities[spin] * int(spin)
                        dos_densities = self.smear(dos_densities)
                        label = el if spin == Spin.up else None
                        dos_ax.plot(
                            dos_densities,
                            dos_energies,
                            color=colors[idx],
                            label=label,
                        )

                elif self.dos_projection.lower() == "orbitals":
                    # plot each of the atomic projected DOS
                    colors = ["b", "r", "g", "m"]
                    spd_dos = dos.get_spd_dos()
                    for idx, orb in enumerate([OrbitalType.s, OrbitalType.p, OrbitalType.d, OrbitalType.f]):
                        if orb in spd_dos:
                            dos_densities = spd_dos[orb].densities[spin] * int(spin)
                            dos_densities = self.smear(dos_densities)
                            label = orb if spin == Spin.up else None
                            dos_ax.plot(
                                dos_densities,
                                dos_energies,
                                color=colors[idx],
                                label=label,
                            )

            # get index of lowest and highest energy being plotted, used to help auto-scale DOS x-axis
            emin_idx = next(x[0] for x in enumerate(dos_energies) if x[1] >= emin)
            emax_idx = len(dos_energies) - next(x[0] for x in enumerate(reversed(dos_energies)) if x[1] <= emax)

            # determine DOS x-axis range
            dos_xmin = (
                0 if Spin.down not in dos.densities else -max(dos.densities[Spin.down][emin_idx: emax_idx + 1] * 1.05)
            )
            dos_xmax = max([max(dos.densities[Spin.up][emin_idx:emax_idx]) * 1.05, abs(dos_xmin)])

            # set up the DOS x-axis and add Fermi level line
            dos_ax.set_xlim(dos_xmin, dos_xmax)
            dos_ax.set_xticklabels([])
            dos_ax.set_xticklabels([])
            dos_ax.hlines(y=0, xmin=dos_xmin, xmax=dos_xmax, color="k", lw=2)
            dos_ax.set_xlabel("PDOS for each orbital", fontsize=self.axis_fontsize, family=self.font)

        # add legend for DOS
        if dos and self.dos_legend:
            dos_ax.legend(
                fancybox=True,
                prop={"family": self.font, "size": self.legend_fontsize},
                loc=self.dos_legend,
            )

        mplt.subplots_adjust(wspace=0.1)
        return mplt

    def get_plot_detail(self, dos=None, ymax=15, filtter=False, tolerance=0.5, mark_orbital=None, mark_element=None,
                        ax_num=None):
        import matplotlib.pyplot as mplt

        # make sure the user-specified band structure projection is valid
        elements = [e.symbol for e in dos.structure.composition.elements]
        if mark_element is None:
            mark_element = elements
        mark_element_num = [elements.index(i) for i in mark_element]
        # specify energy range of plot
        emin = -self.vb_energy_range
        emax = self.cb_energy_range

        # set up bs and dos plot

        fig = mplt.figure(figsize=self.fig_size)
        fig.patch.set_facecolor("white")

        dos_energies = [e - dos.efermi for e in dos.energies]

        start = np.argmin(np.abs(np.array(dos_energies) - emin))
        end = np.argmin(np.abs(np.array(dos_energies) - emax))

        colors = ['orangered', 'maroon', 'aquamarine', 'coral', 'lightgreen', 'magenta', 'orange', 'sienna',
                  'salmon', 'yellow', 'chocolate', 'yellowgreen', 'white', 'aqua', 'goldenrod', 'khaki',
                  'violet', 'lightblue', 'indigo', 'darkgreen', 'purple', 'black', 'wheat', 'azure', 'teal', 'tomato',
                  'green', 'silver', 'blue', 'ivory', 'chartreuse', 'grey', 'navy', 'lavender', 'olive', 'turquoise',
                  'orchid', 'tan', 'brown', 'lime', 'fuchsia', 'crimson', 'darkblue', 'cyan', 'red']
        if mark_orbital is None:
            mark_orbital = [OrbitalType.s, OrbitalType.p, OrbitalType.d, OrbitalType.f]
        else:
            kv = {"s": OrbitalType.s, "p": OrbitalType.p, "d": OrbitalType.d, "f": OrbitalType.f}
            mark_orbital = [kv[i] for i in mark_orbital]

        spd_dos = [dos.get_element_spd_dos(_) for _ in dos.structure.composition]

        spd_dos_res = {}

        for i in mark_element_num:
            spd_dosi = spd_dos[i]
            for ii in spd_dosi.keys():
                if ii in mark_orbital:
                    if filtter and np.max((spd_dosi[ii].densities[Spin.up]
                                           + spd_dosi[ii].densities[Spin.down])[start:end]) > tolerance:
                        spd_dos_res.update({f"{elements[i]}-{ii}-PDOS": spd_dosi[ii].densities[Spin.up] +
                                                                        spd_dosi[ii].densities[Spin.down]})
                    elif filtter is False:
                        spd_dos_res.update(
                            {f"{elements[i]}-{ii}-PDOS": spd_dosi[ii].densities[Spin.up] + spd_dosi[ii].densities[
                                Spin.down]})
                    else:
                        pass

        rows = ax_num + 1 if ax_num is not None and len(spd_dos_res) + 1 <= ax_num + 1 else len(spd_dos_res) + 1
        gs = gridspec.GridSpec(rows, 1)

        for axi, (labels, dos_densities) in enumerate(spd_dos_res.items()):
            dos_ax = mplt.subplot(gs[axi])
            dos_ax.tick_params(axis='both', which='both', direction='in')
            dos_densities = self.smear(dos_densities)
            dos_ax.set_xlim(emin, emax)
            dos_ax.set_ylim(0, ymax)
            dos_ax.set_xticks([])
            dos_ax.set_yticks([])

            dos_ax.vlines(x=0, ymin=0, ymax=ymax, color="k", linestyles="--", lw=0.5, alpha=0.5)
            dos_ax.plot(
                dos_energies,
                dos_densities,
                color=colors[axi],
                label=labels,
                lw=2
            )
            dos_ax.fill_between(
                dos_energies,
                dos_densities,
                color=colors[axi], alpha=0.5
            )
            dos_ax.legend(loc='upper left', fontsize="small", frameon=False)

        dos_ax = mplt.subplot(gs[rows - 1])
        dos_densities = dos.densities[Spin.up] + dos.densities[Spin.down]
        dos_densities = self.smear(dos_densities)
        label = "Total-DOS"
        dos_ax.set_xlim(emin, emax)
        dos_ax.set_ylim(0, 3 * ymax)
        dos_ax.set_xticks(np.linspace(emin, emax, 5))
        dos_ax.set_xticklabels(["-10", "-5", "0 ($E_{f}$)", "5", "10"])
        dos_ax.set_yticklabels([])
        # dos_ax.set_xticks([])
        dos_ax.vlines(x=0, ymin=0, ymax=3 * ymax, color="k", linestyles="--", lw=0.5, alpha=0.5)
        dos_ax.plot(
            dos_energies,
            dos_densities,
            color="k",
            label=label,
            lw=2
        )
        dos_ax.fill_between(
            dos_energies,
            dos_densities,
            color="k", alpha=0.5
        )

        dos_ax.tick_params(axis='both', which='both', direction='in')
        dos_ax.legend(loc='upper left', fontsize="small", frameon=False)

        fig.subplots_adjust(wspace=0.05, hspace=0)
        return mplt
