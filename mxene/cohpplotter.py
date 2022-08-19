#!/usr/bin python3
# -*- coding: utf-8 -*-

# Author        :  Administrator
# Project Name  :  pythonProject
# IDE Name      :  PyCharm
# Time          :  2022/2/19 22:28

"""This is one script for ... """
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.lobster import Cohpcar, Icohplist


class COHPPlotterBatch:

    def transform_path(self, paths, format_path=None):
        """paths: dir paths, each one contain ICOHPLIST.lobster and COHPCAR.lobster."""

        if format_path == "default":
            import re
            format_path = lambda x: re.split(r" |-|/|\\", x)[-2]

        elif format_path is None:
            format_path = lambda x: x

        mark = [format_path(i) for i in paths]

        data_all = {}

        for pathi, i in zip(paths, mark):

            if os.path.isfile(Path(pathi) / "ICOHPLIST.lobster"):
                icohplist = Icohplist(are_coops=False, are_cobis=False, filename=str(Path(pathi) / "ICOHPLIST.lobster"))
                data = self._get_icohp(icohplist=icohplist)
                data_all.update({f"{i}": data})
            else:
                print(f"no data for {i}.")
                data_all.update({f"{i}": None})

        da = pd.DataFrame.from_dict(data_all).T
        da.to_csv("-ichop_point.csv")

        data_all = {}
        for pathi, i in zip(paths, mark):

            if os.path.isfile((Path(pathi) / "COHPCAR.lobster")):
                cohpcar = Cohpcar(are_coops=False, are_cobis=False, filename=str(Path(pathi) / "COHPCAR.lobster"))
                data = self._get_cohp(cohpcar=cohpcar)
                data_all.update({f"{i}": data})
            else:
                print(f"no data for {i}.")
                data_all.update({f"{i}": None})

        da2 = pd.DataFrame.from_dict(data_all)
        da2.to_csv("chop_point.csv")
        return da, da2

    def _get_icohp(self, icohplist):
        data = np.array([
            icohplist.icohplist["1"]["length"],
            -(icohplist.icohplist["1"]["icohp"][Spin.up]),
            -(icohplist.icohplist["1"]["icohp"][Spin.down])
        ]).ravel()
        return data

    def _get_cohp(self, cohpcar):
        # import matplotlib.pyplot as plt
        # plt.plot(cohpcar.energies,cohpcar.cohp_data["average"]["COHP"][Spin.up])
        # plt.show()
        data = np.vstack((cohpcar.cohp_data["average"]["COHP"][Spin.up])).ravel()
        # data = np.vstack((cohpcar.energies, cohpcar.cohp_data["average"]["COHP"][Spin.up])).ravel()
        return data

    def transform_obj(self, icohplist_list: List[Icohplist], cohpcar_list: List[Cohpcar], marks: List[str]):
        """
        Get table.

        Parameters
        ----------
        icohplist_list: List[Icohplist]
        cohpcar_list: List[Cohpcar]
        marks: List[str]

        """
        assert len(icohplist_list) == len(cohpcar_list) == len(marks)

        data_all = {}

        for icohplist, i in zip(icohplist_list, marks):
            data = self._get_icohp(icohplist=icohplist)
            data_all.update({f"{i}": data})
        da = pd.DataFrame.from_dict(data_all).T
        da.to_csv("-ichop_point.csv")

        data_all = {}
        for cohpcar, i in zip(cohpcar_list, marks):
            data = self._get_cohp(cohpcar=cohpcar)
            data_all.update({f"{i}": data})

        da2 = pd.DataFrame.from_dict(data_all)
        da2.to_csv("chop_point.csv")

    def get_plot_from_path(self, paths, format_path=None):
        """paths: dir paths, each one contain ICOHPLIST.lobster and COHPCAR.lobster."""
        self.transform_path(paths=paths, format_path=format_path)
        return self.get_plot_from_table(cohp_file="chop_point.csv", icohp_file="-ichop_point.csv")

    def get_plot_from_obj(self, icohplist_list: List[Icohplist], cohpcar_list: List[Cohpcar], marks: List[str]):
        """
        plot from pymatgen list of Icohplist, and list of Cohpcar.

        Parameters
        ----------
        icohplist_list: List[Icohplist]
        cohpcar_list: List[Cohpcar]
        marks: List[str]
        """
        self.transform_obj(icohplist_list=icohplist_list, cohpcar_list=cohpcar_list, marks=marks)
        return self.get_plot_from_table(cohp_file="chop_point.csv", icohp_file="-ichop_point.csv")

    @staticmethod
    def get_plot_from_table(cohp_file="chop_point.csv", icohp_file="-ichop_point.csv"):
        """
        plotter cohp data.

        Parameters
        ----------
        cohp_file:str
             chop_point.csv
        icohp_file:str
            -ichop_point.csv
        """
        data = pd.read_csv(cohp_file)
        data2 = pd.read_csv(icohp_file)
        data2 = data2.to_dict()  # -ichop

        data2_icohp = {k: "%.2f" % v for k, v in zip(data2["Unnamed: 0"].values(), data2["1"].values())}

        overlap = ['orangered', 'maroon', 'aquamarine', 'coral', 'lightgreen', 'magenta', 'orange', 'sienna', 'gold',
                   'salmon', 'yellow', 'pink', 'plum', 'chocolate', 'yellowgreen', 'white', 'aqua', 'goldenrod',
                   'khaki', 'violet', 'lightblue', 'indigo', 'darkgreen', 'purple', 'black', 'wheat', 'azure', 'teal',
                   'tomato', 'green', 'silver', 'blue', 'ivory', 'chartreuse', 'grey', 'navy', 'lavender', 'olive',
                   'turquoise', 'orchid', 'tan', 'brown', 'lime', 'fuchsia', 'crimson', 'darkblue', 'cyan', 'red']

        plt.rcParams['font.sans-serif'] = "Arial"

        fig, axs = plt.subplots(10, 3)

        fig.set_size_inches(6.4, 9)
        # fig.set_dpi(400)

        fig.subplots_adjust(wspace=0.05, hspace=0.1)

        for n, i in enumerate(["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", ]):
            datai = data[i].values
            color = overlap[n]
            axs[n][0].tick_params(axis='both', which='both', direction='in')
            axs[n][0].plot(np.linspace(-10, 5, 301, ), datai, label=f"{i} (-ICOHP={data2_icohp[i]})", color=color)
            axs[n][0].fill_between(np.linspace(-10, 5, 301, ), 0, datai, datai, alpha=.7, linewidth=0, color=color)
            axs[n][0].set_yticks([-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.25])
            axs[n][0].set_yticklabels(["", "-0.2", "", "0", "", "0.2", ""])
            axs[n][0].legend(loc='lower right', fontsize="x-small", frameon=False, bbox_to_anchor=(1, -0.1))
            axs[n][0].vlines(0, ymin=-0.25, ymax=0.25, colors='k', linestyles='--', linewidths=0.5, alpha=.5)
            axs[n][0].set_ylim(ymin=-0.25, ymax=0.25)
            axs[n][0].set_xlim(xmin=-10, xmax=5)
            if n != 9:
                axs[n][0].tick_params(axis='both', which='both', direction='in')
                axs[n][0].set_xticklabels([])
                # axs[n][0].set_yticklabels([])
            else:
                axs[n][0].tick_params(axis='both', which='both', direction='in')

        for n, i in enumerate(["Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", ]):

            color = overlap[n]
            if i != "Tc":
                datai = data[i].values
                axs[n][1].plot(np.linspace(-10, 5, 301, ), datai, label=f"{i} (-ICOHP={data2_icohp[i]})", color=color)
                axs[n][1].fill_between(np.linspace(-10, 5, 301, ), 0, datai, datai, alpha=.7, linewidth=0, color=color)
                axs[n][1].legend(loc='lower right', fontsize="x-small", frameon=False, bbox_to_anchor=(1, -0.1))
                axs[n][1].vlines(0, ymin=-0.25, ymax=0.25, colors='k', linestyles='--', linewidths=0.5, alpha=.5)
            else:
                axs[n][1].plot(np.linspace(-10, 5, 301, ), np.linspace(-0.25, 0.25, 301), label=i, color='k', lw=0.5)
                axs[n][1].legend(loc='lower right', fontsize="x-small", frameon=False, bbox_to_anchor=(1, -0.1))

            axs[n][0].tick_params(axis='both', which='both', direction='in')
            axs[n][1].set_ylim(ymin=-0.25, ymax=0.25)
            axs[n][1].set_xlim(xmin=-10, xmax=5)
            if n != 9:
                axs[n][1].tick_params(axis='both', which='both', direction='in')
                axs[n][1].set_xticklabels([])
                axs[n][1].set_yticklabels([])
            else:
                axs[n][1].tick_params(axis='both', which='both', direction='in')
                axs[n][1].set_yticklabels([])
            if i == "Tc":
                axs[n][1].set_xticks([])
                axs[n][1].set_yticks([])

        for n, i in enumerate(["La-Gd", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"]):
            color = overlap[n]
            if i not in ["La-Gd", "Hg"]:
                datai = data[i].values

                axs[n][2].plot(np.linspace(-10, 5, 301), datai, label=f"{i} (-ICOHP={data2_icohp[i]})", color=color)
                axs[n][2].fill_between(np.linspace(-10, 5, 301, ), 0, datai, datai, alpha=.7, linewidth=0, color=color)
                axs[n][2].legend(loc='lower right', fontsize="x-small", frameon=False, bbox_to_anchor=(1, -0.1))
                axs[n][2].vlines(0, ymin=-0.25, ymax=0.25, colors='k', linestyles='--', linewidths=0.5, alpha=.5)
            else:
                axs[n][2].plot(np.linspace(-10, 5, 301), np.linspace(-0.25, 0.25, 301), label=i, color='k', lw=0.5)
                axs[n][2].legend(loc='lower right', fontsize="x-small", frameon=False, bbox_to_anchor=(1, -0.1))

            axs[n][2].set_ylim(ymin=-0.25, ymax=0.25)
            axs[n][2].set_xlim(xmin=-10, xmax=5)
            if n != 9:
                axs[n][2].tick_params(axis='both', which='both', direction='in')
                axs[n][2].set_xticklabels([])
                axs[n][2].set_yticklabels([])
            else:
                axs[n][2].tick_params(axis='both', which='both', direction='in')
                axs[n][2].set_yticklabels([])
            if i in ["La-Gd"]:
                axs[n][2].set_xticks([])
                axs[n][2].set_yticks([])

        # plt.savefig("Figure.3.jpg", dpi=300, bbox_inches="tight")
        # plt.show()
        return plt


if __name__ == '__main__':
    os.chdir(r"../MoCMo-O-4")

    ds_by_number = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb", "Mo",
                    "Ru", "Rh", "Pd", "Ag", "Cd", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", ]

    cb = COHPPlotterBatch()

    plt = cb.get_plot_from_path(paths=[f"./{i}/pure_static" for i in ds_by_number], format_path="default")

    plt.savefig("Figure.3.jpg", dpi=300, bbox_inches="tight")

    plt.show()
