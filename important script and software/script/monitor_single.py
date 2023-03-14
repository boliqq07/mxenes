# -*- coding: utf-8 -*-

# @Time  : 2023/3/14 15:45
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

from psutil import cpu_percent

import os
import time

msg = []

def cmd_sys(cmd, d=None):
    """Run linux cmd"""
    if d is None:
        os.system(cmd)
    else:
        old = os.getcwd()
        os.chdir(d)
        os.system(cmd)
        os.chdir(old)


def cmd_popen(cmd, d=None):
    """Run linux cmd and return result."""
    if d is None:
        res = os.popen(cmd).readlines()
    else:
        old = os.getcwd()
        os.chdir(d)
        res = os.popen(cmd).readlines()
        os.chdir(old)
    return res


def get_left_limit():
    pct = int(cpu_percent(interval=2))
    if pct > 75.0:
        return 0
    else:
        return 1


def sub_left_jobs(file="paths.temp", n=1):
    with open(file, "r") as f:
        words = f.readlines()

    words = [i.replace("\n", "") for i in words]

    to_word = words[:n]
    left_word = words[n:]

    succeed = []
    failed = []

    old = os.getcwd()

    if isinstance(to_word,str):
        to_word = [to_word]

    if isinstance(left_word,str):
        left_word = [left_word]

    for i in to_word:
        msg.append(f"Try to submit {i}")
        try:
            os.chdir(i)
            res = cmd_popen(xx_run)
            msg.append(str(res))
            succeed.append(i)
            os.chdir(old)
        except BaseException as e:
            n = get_left_limit()
            msg.append(f"Error check: {n} jobs can be submit.")
            os.chdir(old)
            if isinstance(str(e), (list, tuple)):
                [msg.append(str(i)) for i in str(e)]
            else:
                msg.append(str(e))
            failed.append(i)
            print(msg)

    os.chdir(old)
    if len(succeed) > 0:
        with open(f"Succeed_{file}", 'a+') as ff:
            ff.write("\n" + ("\n".join(succeed)))
    if len(failed) > 0:
        with open(f"Failed_{file}", 'a+') as ff:
            ff.write("\n" + ("\n".join(failed)))
    print(msg)
    with open(file, 'w') as ff:
        ff.write("\n".join(left_word))


def loop(file="paths.temp"):

    msg.append("Init Start:\n")
    with open(file, "r") as f:
        words = f.readlines()

    if os.path.isfile(f"log_{file}"):
        os.remove(f"log_{file}")

    lp = 1

    msg.append(f">>> Next Loop ({lp}): Left {len(words)} jobs >>>")

    while len(words) > 0:
        data = time.time()
        timearray = time.localtime(data)
        ti = time.strftime('%Y-%m-%d %H:%M:%S', timearray)
        msg.append(str(ti))

        n = get_left_limit()

        msg.append(f"{n} jobs can be submit.")
        if n > 0:
            sub_left_jobs(file=file, n=n)

        with open(file, "r") as f:
            words = f.readlines()

        msg.append(f"<<< Loop ({lp}) end. Left {len(words)} jobs <<<\n")

        with open(f"log_{file}", 'a+') as ff:
            ff.write("\n" + ("\n".join(msg)))
        msg.clear()

        time.sleep(120)
        lp += 1
        msg.append(f">>> Next Loop ({lp}): Left {len(words)} jobs >>>")

    msg.append(f"\nEnd")

    with open(f"log_{file}", 'a+') as ff:
        ff.write("\n" + ("\n".join(msg)))
    msg.clear()


xx_run = '''
mpirun -np 64 /home/xdjf/app/vasp_fix_neb/bin/vasp_std > log.txt &
'''

if __name__ == "__main__":
    print(__file__)
    loop(file="paths.temp")
