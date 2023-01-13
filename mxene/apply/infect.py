# -*- coding: utf-8 -*-

# @Time  : 2022/10/2 13:20
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

from functools import wraps

import numba
import numpy as np
import scipy.interpolate as interp
from PIL import Image
from ase.visualize import view
from matplotlib import pyplot as plt
from pymatgen.core import Structure, SymmOp
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.interpolate import griddata

from mgetool.tool import tt

frames = []


def store_fig(func):
    """画图"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        plt.imshow(result)
        plt.show()
        frames.append(Image.fromarray(result.astype(np.uint8)))
        return result

    return wrapper


def from_points_to_matrix(x, y, z, new_shape=(int(9 * 100), int(9 * (1.732 / 2) * 100)),
                          method="cubic"):
    """产生网格。"""
    # f = interp.interp2d(x, y, z,bounds_error=False,fill_value=fill_value,kind=kind)
    x_min = min(x)
    y_min = min(y)
    x_max = max(x)
    y_max = max(y)
    fill_value = max(z)
    # newz = f(np.linspace(x_min,x_max,new_shape[0]), np.linspace(y_min,y_max,new_shape[1]))

    grid_x, grid_y = np.linspace(x_min, x_max, new_shape[0]), np.linspace(y_min, y_max, new_shape[1])
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    newz = griddata((x, y), z, (grid_x, grid_y), method=method, fill_value=fill_value,
                    rescale=False)

    return newz


def get_test_data():
    """产生数据"""
    x = range(0, 50, 10)
    y = range(0, 50, 10)
    np.random.seed(1)
    z = np.random.random((5, 5))

    xx, yy = np.meshgrid(x, y)
    f = interp.interp2d(xx, yy, z)

    newz = f(range(50), range(50))

    x, y = np.mgrid[-5:5:20j, -5:5:20j]
    sigma = 2
    zz = 1 / (2 * np.pi * (sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) * 20

    newz[20:40, 10:30] -= zz
    newz[0:20, 10:30] -= zz
    plt.imshow(newz)
    plt.show()
    return newz


@store_fig
def infect_with_plot(*args, **kwargs):
    return infect(*args, **kwargs)


@numba.njit("UniTuple(UniTuple(int32,2),8)(int32,int32)")
def getnebor(x, y):
    """获取邻域索引"""
    nebor = ((x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
             (x, y - 1), (x, y + 1),
             (x + 1, y - 1), (x + 1, y), (x + 1, y + 1),)
    return nebor


@numba.njit("int32[:,:](int32[:,:],int32,int32)")  # numba 加速
def infect(zb, site0x, site0y):
    """
    从一个起始值传播。

    Args:
        zb: (np.ndarray), numpy 整数数组，初始全部0或者1。
        site0x: (int32)， 起始x值。
        site0y: (int32), 起始y值。

    Returns:
        zb: (np.ndarray), numpy 整数数组，全部0或者1,或者-1。
    """
    temp = list(getnebor(site0x, site0y))  # 获取邻域索引
    size_x, size_y = zb.shape
    while len(temp) > 0:
        nebor_ix, nebor_iy = temp.pop()  # 取出一个位置(x,y)，并从temp删除
        if nebor_ix == -1 or nebor_iy == -1 or nebor_ix == size_x or nebor_iy == size_y:  # 防止边界溢出
            pass
        elif zb[nebor_ix, nebor_iy] == 0:
            temp.extend(getnebor(nebor_ix, nebor_iy))  # 获取，添加邻域索引
            zb[nebor_ix, nebor_iy] = -1  # 赋值为-1
        # else:  # 值为1,不作操作。
        #     pass
    return zb


def find_energy(newz, site0x, site0y, site1x, site1y, step=None, number=100, plot_or_not=False):
    """
    寻找势垒

    Args:
        newz: (np.ndarray), 势能面。
        site0x: (int),起始位置x
        site0y: (int),起始位置y
        site1x: (int),结束位置x
        site1y: (int),结束位置y
        plot_or_not: (bool), 画图
        step: (bool), zi划分个数
        number: (int), zi 划分个数，若步长step存在，优先使用步长,此参数忽略。

    Returns:
        zb:(np.ndarray) 势能面决策结果矩阵。
        zi:(np.ndarray) 能量值。

    """
    zs_start = newz[site0x, site0y]
    if isinstance(step, (float, int)):  # 定义z可变化的序列zs
        zs = np.arange(zs_start, np.max(newz), step)
    else:
        zs = np.linspace(zs_start, np.max(newz), number)

    infect0 = infect if not plot_or_not else infect_with_plot  # 确定使用函数，infect_with_plot比infect多一个画图功能。

    zb = (newz - np.min(newz)) >= 0  # 无用，占位
    zi = np.min(newz)  # 无用，占位

    for zi in zs:  # 核心部分
        zb = (newz - zi) >= 0  # 判断大小
        zb = zb.astype(np.int32)  # 复制 0, 1
        zb = infect0(zb, site0x, site0y)  # 传播0的领域0为-1
        if zb[site1x, site1y] == -1:  # 结束点为-1，终止
            break

    if plot_or_not:  # 存储gif
        try:
            frames[0]._save("find.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
        except BaseException:
            pass
    return zb, zi


def show(self):
    """
    展示画图。
    """
    aaa = AseAtomsAdaptor()
    atomss = aaa.get_atoms(self)
    view(atomss)


def get_final_atom_cart_cords(st: Structure):
    return st.cart_coords[-1]


def get_data_spilt(cart_all_all):
    index = np.lexsort(cart_all_all[:, ::-1].T, axis=-1)
    return index


def mx_reflection(structure, energys, center_site_cart):
    """反射。"""
    sui = structure

    so = SymmOp.reflection([1, 0, 0], origin=center_site_cart)  # 确保保证原子在面同侧!!!!!!!!!!!

    sui_ref = sui.copy()
    sui_ref.apply_operation_no_lattice(so)
    sui0 = sui + sui_ref

    energys = np.concatenate([energys] * 2)
    energys = energys.reshape((-1, 1))

    return sui0, energys


def mx_rotate3(structure, energys, center_site_cart):
    """旋转三次。"""
    sui0 = structure

    angle = sui0.lattice.angles[-1]  # 完美的为120度

    sui120 = sui0.copy()
    sui240 = sui0.copy()
    sui120.rotate_sites(theta=angle / 360 * 2 * np.pi, axis=[0, 0, 1], anchor=center_site_cart, to_unit_cell=False)
    sui240.rotate_sites(theta=-angle / 360 * 2 * np.pi, axis=[0, 0, 1], anchor=center_site_cart, to_unit_cell=False)

    sui_all = sui0 + sui120 + sui240

    energys = np.concatenate([energys] * 3)

    energys = energys.reshape((-1, 1))
    return sui_all, energys


def mx_supercell(structure, energys):
    """超胞3X3X1"""
    sui_all = structure

    sui_all.make_supercell((3, 3, 1), to_unit_cell=False)

    energys = np.repeat(energys, repeats=9, axis=1).ravel()
    energys = energys.reshape((-1, 1))
    return sui_all, energys


def mx_reflection_and_rotate_supercell(structure, energys, center_site_cart):
    """反射"""
    sui = structure

    angle = sui.lattice.angles[-1]  # 完美的为120度

    so = SymmOp.reflection([1, 0, 0], origin=center_site_cart)  # 确保保证原子在面同侧!!!!!!!!!!!

    sui_ref = sui.copy()
    sui_ref.apply_operation_no_lattice(so)
    sui0 = sui + sui_ref
    sui120 = sui0.copy()
    sui240 = sui0.copy()
    sui120.rotate_sites(theta=angle / 360 * 2 * np.pi, axis=[0, 0, 1], anchor=center_site_cart, to_unit_cell=False)
    sui240.rotate_sites(theta=-angle / 360 * 2 * np.pi, axis=[0, 0, 1], anchor=center_site_cart, to_unit_cell=False)

    sui_all = sui0 + sui120 + sui240

    sui_all.make_supercell((3, 3, 1), to_unit_cell=False)

    energys = np.concatenate([energys] * 6)
    energys = energys.reshape((-1, 1))
    energys = np.repeat(energys, repeats=9, axis=1).ravel()

    energys = energys.reshape((-1, 1))
    return sui_all, energys


@numba.njit("List(float64[:])(float64[:,:],int32)")
def _remove_dup(data_min, n=2):
    data_min2 = []
    k = data_min.shape[0] - 1
    min_indexs = []
    for i in range(k):
        if i not in min_indexs:
            same_index = np.where(np.sum((data_min[:, :n] - data_min[i, :n]) ** 2, axis=1) ** 0.5 < 0.2)[0]
            min_index = np.argmin(data_min[same_index, 3])
            data_min2.append(data_min[same_index[min_index]])
            min_indexs.extend(same_index)
    return data_min2


def remove_dup(structure_or_cart_coords, energys, num_z=5):
    """删除重复原子。"""
    if isinstance(structure_or_cart_coords, Structure):
        cart_coords = structure_or_cart_coords.cart_coords
    else:
        cart_coords = structure_or_cart_coords

    data = np.concatenate((cart_coords, energys), axis=1)
    index = get_data_spilt(data[:, (0, 1, 2)])

    data = data[index]

    label = np.repeat(np.arange(num_z).reshape((-1, 1)), repeats=int(data.shape[0] / num_z), axis=1).T.reshape((-1, 1))

    data_min = np.concatenate((data, label), axis=1)

    data_min = np.array(_remove_dup(data_min, n=3))
    data_min = np.array(_remove_dup(data_min, n=2))

    return data_min


def get_2_equ_min_site(face, angle, tol=0.01, super_cell=(3, 3, 1), axis=1):
    """获取相邻晶胞的两个等效位。"""
    min_value = np.min(face)
    index = np.where(face > (min_value + tol))
    index = np.concatenate((index[0].reshape(-1, 1), index[1].reshape(-1, 1)), axis=1)
    dis = index - np.array([face.shape[0] / 2, face.shape[1] / 2])
    dis = np.sum(dis ** 2, axis=1) ** 0.5
    sel = np.argmin(dis)
    point1 = index[sel]
    if np.cos(angle / 180 * np.pi) >= 0:
        w = 1
    else:
        w = 1 - np.cos(angle / 180 * np.pi)
    num = face.shape[axis]
    offset_num = int(num / w / super_cell[0])
    point2 = np.copy(point1)
    point2[axis] = point2[axis] + offset_num
    return point1, point2


######################
if __name__ == "__main__":
    tt.t1
    newz = get_test_data()
    site0x, site0y = 10, 20
    site1x, site1y = 28, 30

    ##################
    tt.t2  # 纯计算
    zb, zi = find_energy(newz, site0x, site0y, site1x, site1y, step=0.02, plot_or_not=True)
    tt.t3  # 计算+ 画图
    zb2, zi2 = find_energy(newz, site0x, site0y, site1x, site1y, step=0.02, plot_or_not=False)
    tt.t4
    tt.p
    print("Energy:", zi)
