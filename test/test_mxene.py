import unittest

import numpy as np
from pymatgen.core import Lattice

from mxene.mxene import MXene


class MyTestCase(unittest.TestCase):

    def test_mxene_init(self):
        lattice = Lattice([8.563571, 0.000000, 0.000000, -4.281786, 7.416274, 0.000000, 0.000000, 0.000000, 25.000000])
        frac = [-0.000171, 0.000171, 0.500238,
                0.000119, 0.333393, 0.500267,
                -0.000171, 0.666325, 0.500238,
                0.333675, 0.000171, 0.500238,
                0.335265, 0.337196, 0.500488,
                0.335265, 0.664735, 0.500488,
                0.666607, -0.000119, 0.500267,
                0.666607, 0.333393, 0.500267,
                0.662804, 0.664735, 0.500488,
                -0.000065, 0.000065, 0.603618,
                0.001574, 0.334121, 0.603884,
                -0.000065, 0.666536, 0.603618,
                0.333464, 0.000065, 0.603618,
                0.338273, 0.343213, 0.601497,
                0.338272, 0.661728, 0.601497,
                0.665879, -0.001574, 0.603884,
                0.665880, 0.334120, 0.603884,
                0.656788, 0.661727, 0.601497,
                0.000162, -0.000162, 0.396561,
                0.000113, 0.333390, 0.396467,
                0.000162, 0.666991, 0.396561,
                0.333009, -0.000162, 0.396561,
                0.333464, 0.333595, 0.396667,
                0.333464, 0.666536, 0.396667,
                0.666610, -0.000113, 0.396467,
                0.666610, 0.333390, 0.396467,
                0.666405, 0.666536, 0.396667,
                0.113557, 0.224480, 0.554758,
                0.113557, 0.555744, 0.554758,
                0.111111, 0.888889, 0.554921,
                0.444256, 0.224480, 0.554758,
                0.444256, 0.886443, 0.554758,
                0.777778, 0.222222, 0.555124,
                0.775520, 0.555744, 0.554758,
                0.775520, 0.886443, 0.554758,
                0.222437, 0.111540, 0.445558,
                0.222347, 0.444507, 0.445586,
                0.222437, 0.777563, 0.445558,
                0.555440, 0.111053, 0.445322,
                0.555493, 0.444507, 0.445586,
                0.555493, 0.777653, 0.445586,
                0.888947, 0.111053, 0.445322,
                0.888947, 0.444560, 0.445322,
                0.888460, 0.777563, 0.445558,
                0.444444, 0.555556, 0.552949, ]
        frac = np.array(frac).reshape(-1, 3)
        mx = MXene(lattice, ["C"] * 9 + ["O"] * 18 + ["Mo"] * 17 + ["Cr"] * 1, frac)
        print(mx.cart_coords)
        self.assertEqual(True, True)

    def test_from_poscar(self):
        mx = MXene.from_file("Y-POSCAR")
        print(mx)

    def test_split_layer(self):
        mx = MXene.from_file("Mo-POSCAR")
        label = mx.split_layer(ignore_index=None, n_cluster=None, tol=0.5, axis=2, method=None,
                               force_plane=False, reverse=False)
        mx.make_supercell((3, 3, 1))
        print(label)
        print(mx.cart_coords)

    def test_split_layer2(self):
        mx = MXene.from_file("Y-POSCAR")
        label2 = mx.split_layer(ignore_index=None, n_cluster=None, tol=0.5, axis=2, method=None,
                                force_plane=False, reverse=False)
        print(label2)

    def test_split_layer3(self):
        mx = MXene.from_file("Y-POSCAR")
        label3 = mx.split_layer(ignore_index=None, n_cluster=None, tol=0.5, axis=2, method=None,
                                force_plane=True, reverse=False)
        print(label3)

    def test_split_layer4(self):
        mx = MXene.from_file("Y-POSCAR")
        label3 = mx.split_layer(ignore_index=[-1], n_cluster=None, tol=0.5, axis=2, method=None,
                                force_plane=True, reverse=False)
        print(label3)

    def test_split_layer5(self):
        mx = MXene.from_file("Y-POSCAR")
        label3 = mx.split_layer(ignore_index=[-1], n_cluster=None, tol=0.5, axis=2, method=None,
                                force_plane=True, reverse=True)
        print(label3)

    def test_split_layer6(self):
        mx = MXene.from_file("Y-POSCAR")
        label3 = mx.split_layer(ignore_index=[-1], n_cluster=None, tol=0.5, axis=2, method=None,
                                force_plane=True, reverse=True)
        print(label3)

    def test_next_layery(self):
        mx = MXene.from_file("Y-POSCAR")
        coor = mx.get_next_layer_sites()
        print(coor)

    def test_from_stand(self):
        mx = MXene.from_standard(terminal_site="fcc", doping=None,
                                 terminal="O",
                                 base="Ti", carbide_nitride="C",
                                 n_base=2, add_noise=True,
                                 super_cell=(3, 3, 1), add_atoms=None, add_atoms_site=None)
        print(mx)

    def test_from_stand2(self):
        mx = MXene.from_standard(terminal_site="fcc", doping="Zr",
                                 terminal="O",
                                 base="Ti", carbide_nitride="C",
                                 n_base=2, add_noise=True,
                                 super_cell=(3, 3, 1), add_atoms=None, add_atoms_site=None)
        print(mx)

    def test_from_message(self):
        mx = MXene.from_standard(terminal_site="fcc", doping="Zr",
                                 terminal="O",
                                 base="Ti", carbide_nitride="C",
                                 n_base=2, add_noise=True,
                                 super_cell=(3, 3, 1), add_atoms=None, add_atoms_site=None)

        data = mx.get_structure_message()
        data2 = data

    def test_get_disk(self):
        mx = MXene.from_standard(terminal_site="fcc", doping="Zr",
                                 terminal="O",
                                 base="Ti", carbide_nitride="C",
                                 n_base=2, add_noise=True,
                                 super_cell=(3, 3, 1), add_atoms=None, add_atoms_site=None)
        mx.get_disk()
        print(mx.out_dir)

    def test_absorb(self):
        mx = MXene.from_standard(terminal_site="fcc", doping="Zr",
                                 terminal="O",
                                 base="Ti", carbide_nitride="C",
                                 n_base=2, add_noise=True,
                                 super_cell=(3, 3, 1), add_atoms=None, add_atoms_site=None)
        mx.add_absorb(absorb="H")
        mx.get_disk()
        print(mx.out_dir)

    def test_add_interp(self):
        import matplotlib.pyplot as plt
        mx = MXene.from_standard(terminal_site="fcc", doping="Zr",
                                 terminal="O",
                                 base="Ti", carbide_nitride="C",
                                 n_base=2, add_noise=True,
                                 super_cell=(3, 3, 1), add_atoms=None, add_atoms_site=None)
        f = mx.get_interp2d()
        x = np.arange(-12, 23, 0.2)
        y = np.arange(0, 21, 0.2)
        z = f(x, y, meshed=False)

        plt.imshow(z.T)
        plt.show()

    def test_add_random(self):
        import matplotlib.pyplot as plt
        mx = MXene.from_standard(terminal_site="fcc", doping="Zr",
                                 terminal="O",
                                 base="Ti", carbide_nitride="C",
                                 n_base=2, add_noise=True,
                                 super_cell=(3, 3, 1), add_atoms=None, add_atoms_site=None)
        st = mx.add_face_random(number=40, debug=False)
        st[39].show()

    def test_add_random_16(self):
        import matplotlib.pyplot as plt
        mx = MXene.from_standard(terminal_site="hcp", doping="Zr",
                                 terminal="O",
                                 base="Mo", carbide_nitride="C",
                                 n_base=2, add_noise=True,
                                 super_cell=(3, 3, 1), add_atoms=None, add_atoms_site=None)
        st = mx.add_face_random_z_16_site(debug=True, number_each=5)
        st.to(fmt="poscar", filename="Mo_POSCAR")
        # st.show()

    def test_non_equivalent_site(self):
        import matplotlib.pyplot as plt
        mx = MXene.from_standard(terminal_site="fcc", doping="Zr",
                                 terminal="O",
                                 base="Ti", carbide_nitride="C",
                                 n_base=2, add_noise=True,
                                 super_cell=(3, 3, 1), add_atoms=None, add_atoms_site=None)
        sites = mx.non_equivalent_site(center=44, ignore_index=None, base_m=None, terminal=None)


if __name__ == '__main__':
    unittest.main()
