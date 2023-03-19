# -*- coding: utf-8 -*-

# @Time  : 2023/3/17 15:56
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

incar_para = {

    "ISTART": [
        0,  # 新启动
        1,  # 读取 WAVECAR启动  （首选）
        2,  # 读取 WAVECAR启动
        3,  # （MD使用）
    ],

    "ICHARG": [
        0,  # 根据初始波函数生成
        1,  # 读取CHGCAR中的数据 （热启动）
        2,  # 原子电荷密度                （首选）
        11,  # 固定CHGCAR电荷密度，计算能带

    ],

    "ISPIN": [
        # 自旋
        1,  # 不考虑
        2  # 开
    ],

    "ENCUT": [
        # 截断能,关联PREC
        400,
        500,
        600,
    ],

    "PREC": [
        # 越往下精度越高, 设置此参数可以省略ENCUT
        "Low",  # ENCUT 为POTCAR里最大的 ENMIN
        "Medium",  # ENCUT 为POTCAR里最大的 ENMAX
        "Normal",  # ENCUT 为POTCAR里最大的 ENMAX   (首选)
        "Single",  # ENCUT 为POTCAR里最大的 ENMAX
        "Accurate",  # ENCUT 为POTCAR里最大的 ENMIN
        "High",  # ENCUT 为POTCAR里最大的 ENMIN*1.3
    ],

    "ISIF":[
        0,  # 不优化晶胞，有残余应力，不建议
        1,  # 不优化晶胞，有残余应力，不建议
        2,  # 不优化晶胞，                                                （适当考虑）
        3,  #结构优化时设3难以收敛，可以考虑在2/6/7之间切换优化最后再设成3跑一次。   #（首选）
        4,  # 不优化晶胞体积
        5,  # 只有优化晶胞形状
        6,  # 只有优化晶胞，不优化相对位置
        7,  # 只优化晶胞体积
    ],

    "EDIFF": [
        # 电子收敛差
        "1E-3",  # 粗略测试
        "5E-4",
        "1E-4",  # (默认，优化)
        "1E-5",  # （静态）
    ],

    "EDIFFG": [
        0.001,   # 正值，受力收敛差
        "-1E-3",  # 负值，受力收敛差    （精细优化）
        "-1E-2",  # 负值，受力收敛差   （优化）
        "1E-4",  # (默认，优化)
        "1E-5",  # （静态）
    ],

    "LWAVE": [  # 文件输出
        "T"
        "F"
    ],

    "LCHARG": [  # 文件输出
        "T"
        "F"
    ],

    "LORBIT": [  # 文件输出 结合RWIGS标签确定是否输出PROCAR或PROOUT文件
        0,  #
        11,  # 分波态密度
        "F"  # 默认

    ],

    "ISMEAR": [  # 轨道分布函数，金属不可大于0

        1,  # 用于金属，# 不建议用MXenens，兼容性差  SIGMA=0.2
        0,  # 高斯                             （小胞，少k点，SIGMA=0.05）
        -1,  # 费米
        -5,  # 带布洛赫修正的四面体方法           （ 总能量，态密度，半导体，绝缘体，SIGMA=0.05）

    ],

    "SIGMA": [  # 能量展宽

        0.2,  # 用于金属  ISMEAR=1
        0.05,  # （MXenes 首选）
    ],

    "NELM": [  # 电子自洽迭代数目
        60,  #                      （不建议更改）
    ],

    "NELMIN": [  # 最小次数
        2,  # 默认，建议
    ],

    "NELMDL": [  # 电子步可非自洽次数，负值代表每个离子步均进行
        -5,  # 默认，建议
        -12,  #
        0,  #  （全部自洽）
    ],

    "NEDOS": [  # 能带数目
        301,  # DOS时将画图能量划分成多少个格点         （越高越平滑）
    ],

    "EMIN": [  # DOS最小值 ， 画图， 默认不用
    ],

    "EMAX": [  # DOS最大值， 画图，默认不用
    ],

    "ALGO": [  # 轨道优化算法
        "N",  #  等于38
        "F",  # 等于38+48                                    （首选，LREAL=Auto）
        "Very_Fast",  # 等于48

    ],

    "IALGO": [  # 轨道优化算法 (默认不用)
        38,  # DOS时将画图能量划分成多少个格点， ALGO=N        （默认）
        -1,  # 测试时间
        48,
        "N",  #
    ],

    "ISYM":[ # 对称性
        -1, # 不考虑对称性 （COHP时需要设为-1）
        0, # 不考虑对称性，但会假定波矢相反时波函数为复共轭减少布里渊区中的取样    （首选）
        2, # 考虑对称性
    ],

    "SYMPREC": [  # 对称性
        "1E-5",  # 默认
        "1E-4",
        "1E-3",

    ],

    "NCORE": [  # NCORE(算一个能带需要的核)×NPAR=N(总核数)
        4,
        8,
        16,

    ],

    "NPAR": [  # 计算，所用总核数的平方根，或者单cpu核数  NPAR= SQRT(NBANDS)
        1, # 小体系
        4,
        8,
    ],

    "NSIM": [  #  针对 Very_Fast 48
        2,
        4,  # 默认
        6,
        8,   # 大体系

    ],

    "NSW": [  # 离子步骤
        80,
    ],

    "IBRION": [  # 原子核使用什么方法进行移动
        -1,                                      #（静态）
        0,   #（MD）
        1,   # 1时使用准牛顿方法进行弛豫            （精修）
        2,   # 距离较远时使用2更合适                （首选）
    ],

    "POTIM":[0.5], # MD

    "MAGMOM":[
        # 初始磁矩，个数等于原子数，如： 1 1 1 1 1 1 1 1

    ],

    "LSORBIT":[# 自选轨道耦合
        "T",
        "F" #（默认）
    ],

    "NBANDS":[
        # （默认不要）决定了计算时考虑的能带条数，计算COHP时需要设高一些，差不多是0.8*价电子数。
        300, # 优化计算不要默认不用设置，静态计算可设置
    ],

    "LREAL": [
        "Auto",  #                           （首选）
        "F",  # 默认,单胞
        "T",  # 默认不用设置
        # 决定投影在实空间还是倒空间进行，原子数大于30时建议在实空间投影，同时强烈建议只设置成A（Auto）
    ],

    "IVDW":[
        #是否考虑范德华修正。
        0,
        1,
        12, #                      (首选)
    ],

    "GGA": [
        "PE",
        "RE",
        "RP",
        "PS",
    ],

}

"""
输出文件相关：

LVTOT（总局域势LOCPOT，可获取真空能级）、LELF（电子局域函数ELFCAR）、LAECHG（全电荷密度AECCAR，进行bader电荷分析需要用到）

各种混合参数：

IMIX、AMIX、BMIX、AMIX_MAG、BMIX_MAG、WC、INIMIX、MIXPRE、MAXMIX

介电、压电相关：

LEPSILON、LPRA、LCALCEPS、IBRION

局部（k空间）电荷密度相关：

LPARD、IBAND、EINT、NBMOD、KPUSE、LSEPB、LSEPK

U值相关：

LDAU、LMAXMIX、LDAUTYPE、LDAUL、LDAUJ、LDAUU

磁矩约束相关：

I_CONSTRAINED_M、M_CONSTR、LAMBDA

贝里相位相关：

LBERRY、IGPAR、NPPSTR、DIPOL

杂化泛函相关：

LHFCALC、AEXX、AGGAX、AGGAC、ALDAC、HFSCREEN、LTHOMAS、PRECFOCK、HFLMAX

分子动力学相关：

IBRION、NSW、POTIM、SMASS、MDALGO、ISIF、TEBEG、TEEND、NBLOCK、KBLOCK、PSTRESS、PMASS"""
