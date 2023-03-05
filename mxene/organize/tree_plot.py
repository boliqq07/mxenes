
from typing import Union, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from mgetool.tree_plot import BuildTree


class MXBuildTree(BuildTree):
    """
    For mxenes composition importance show.
    1. Each layer of color are decided by the predefined 'color_maps',
        and each branch color is decided by 'color_maps' and 'com_maps'.
    2. If to sort, the tree are sort by 'com_maps' in each layer.
    3. The site of each node are decided by super root.
    4. After set site (``set_site_all``), don't change the tree (add/append/delete any branch).

    Notes:
        The tree must end with doping, unless you change the 'color_maps' and 'com_maps'.
        (the tree start from 'MX' is not enforced).
        due to the index to get message from 'color_maps' and 'com_maps' is inverted order.

    Examples:
    # Generate 1
    >>> com = {"l3": {"O": {"V": 0.1, "Mo": 0.2}, "S": {"Ti": 0.93, "W": 0.4}}}
    >>> tt0 = BuildTree.from_com_dict(com, sort=True)

    # Generate 2
    >>> tt1 = BuildTree.from_one_branch(branch=['l3', "S", "V"], w=1)

    # Cat 2 tree
    >>> tt2 = BuildTree(name="Cr", w=1.0, sub_tree=[tt0,tt1])

    #  Set point sites, color, and line automatriclly.
    >>> tt2.settle()
    or
    >>> tt2.set_site_all()
    >>> tt2.set_color_all()
    >>> res = tt2.get_line()

    >>> tt2.show(text=True, mark=True)
    >>> msg_dict = tt2.get_msg()
    >>> num_dict = tt2.get_num()
    >>> branch_dict = tt2.get_branch()

    #  Error code !!!
    >>> t1 = BuildTree.from_one_branch(branch=['l3', "S",], w=1) # !!! loss the last layer.
    """

    pre_def_color_maps = [to_rgba("k", alpha=1),
                          to_rgba("k", alpha=1),
                          to_rgba("k", alpha=1),
                          to_rgba("k", alpha=1),
                          plt.get_cmap("cool"),
                          plt.get_cmap("cool"),
                          plt.get_cmap("cool"),
                          ]

    pre_def_com_maps = [
        # 0
        {"MX": 0},
        # 1
        {"C": 0, 'N': 1},
        # 2
        {'L2': 0, 'L3': 1, 'L4': 2},
        # 3
        {'S': 0, 'B': 1,'D': 1,},
        # 4
        {"Hf": 0, 'Zr': 1, "Ta": 2, "Nb": 3, "W": 4,
         "Mo": 5, "Ti": 6, 'V': 7, "Cr": 8},
        # 5
        {"O": 0, 'F': 1, "S": 2, "Cl": 3},
        # 6
        {"Y": 0, "Hf": 1, 'Zr': 2, "Ta": 3, "Nb": 4, "W": 5, "Mo": 6,
         "Re": 7, "Sc": 8, "Os": 9, "Ir": 10, "Ru": 11, "Pt": 12, "Ti": 13,
         "Au": 14, "Rh": 15, 'V': 16, "Pd": 17, "Cr": 18, "Ag": 19, "Cd": 20,
         "Mn": 21, "Fe": 22, "Co": 23, "Ni": 24, "Cu": 25, "Zn": 26,
         "P": 27, "S": 28, "Cl": 29, "B": 30, "C": 31, "N": 32, "O": 33, "F": 34,
         }, ]

    def __init__(self, name: Union[str, Sequence[str]], w: float = 1.0,
                 sub_tree: Union["BuildTree", Sequence["BuildTree"]] = None,
                 array: np.ndarray = None, color_maps=None, com_maps=None,
                 ):
        """

        Args:
            name: str, name of node.
            w: float, weight of this node.
            sub_tree: list of BuildTree, sub-tree.
        """
        if com_maps is None:
            com_maps = self.pre_def_com_maps
        if color_maps is None:
            color_maps = self.pre_def_color_maps
        super(MXBuildTree, self).__init__(name=name, w=w, sub_tree=sub_tree, array=array,
                                          color_maps=color_maps, com_maps=com_maps, )


if __name__ == "__main__":
    np.random.seed(4)


    tt1 = MXBuildTree.from_one_branch(branch=["C", ("Ta", "Ti"), 'l3', "O", "V"], w=1, )
    tt2 = MXBuildTree.from_one_branch(branch=["C", "Ta", 'l3', "O", "Pd"], w=1, )
    tt3 = MXBuildTree.from_one_branch(branch=["C", "Ta", 'l3', "O", "Rh"], w=1, )
    tt4 = MXBuildTree.from_one_branch(branch=["C", "Ta", 'l3', "O", "Mo"], w=1, )
    tt5 = MXBuildTree.from_one_branch(branch=["C", "Ta", 'l3', "O", "Au"], w=1, )
    #
    tt6 = MXBuildTree.from_one_branch(branch=["C", "Ta", 'l5', "O", "Mo"], w=1, )
    tt7 = MXBuildTree.from_one_branch(branch=["C", "Ta", 'l5', "O", "Au"], w=1, )
    tt8 = MXBuildTree.from_one_branch(branch=["C", "Ta", 'l5', "O", "Ti"], w=1, )
    tt9 = MXBuildTree.from_one_branch(branch=["C", "Ta", 'l5', "O", "Zn"], w=1, )
    tt10 = MXBuildTree.from_one_branch(branch=["C", "Ta", 'l5', "S", "Zn"], w=0.2, )
    tt11 = MXBuildTree.from_one_branch(branch=["C", "Ta", 'l5', "S", "V"], w=0.2, )
    tt12 = MXBuildTree.from_one_branch(branch=["C", "Cr", 'l5', "S", "Fe"], w=0.2, )

    ttt = [tt12, tt11, tt9, tt8, tt7, tt1, tt10, tt2, tt3]
    ttts = MXBuildTree.sum_tree(ttt, n_jobs=1)
    ttts.settle()
    plt = ttts.get_plt(text=True, mark=True, mark_size=500)
    plt.show()
