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


def get_node_run(run_file=None, node="normal3"):
    cp = {"normal2":"28","normal3":"28","normal4":"28","6248":"40","8360":"72"}

    n_cpu=cp[node]
    
    if run_file is None:
        run_file = xx_run
    run_file = run_file.replace("normal3", node)
    run_file = run_file.replace("28", n_cpu, 2)
    return run_file


def get_left_limit(up_limit=4):
    username=os.getlogin()
    res = os.popen(f"squeue -u {username}|grep PD|wc -l").readlines()[0]
    num = int(res)
    left_num = up_limit - num
    return max(left_num, 0)


# 提取待提交任务：
def sub_left_jobs(file="paths.temp", n=1, run_file=None, node="normal3"):
    with open(file, "r") as f:
        words = f.readlines()

    words = [i.replace("\n", "") for i in words]

    to_word = words[:n]
    left_word = words[n:]

    node_mark = len(to_word)*[node]

    msg.append(f"Use node: {str(node_mark)}")

    succeed = []
    failed = []

    old = os.getcwd()

    for i, node_marki in zip(to_word, node_mark):
        msg.append(f"Try to submit {i}")
        print(i, node_marki)
        try:
            os.chdir(i)
            if run_file is None:
                with open(f"dfy_{node_marki}.run", "w") as ff:
                    ff.write(get_node_run(run_file=None, node=node_marki))
                res = cmd_popen(f"sbatch $(find -name dfy_{node_marki}.run)")[0]
            elif "#!/bin" in run_file:
                with open(f"dfy_special.run", "w") as ff:
                    ff.write(get_node_run(run_file=run_file, node=node_marki))
                res = cmd_popen(f"sbatch $(find -name dfy_special.run)")[0]
            else:
                res = cmd_popen(f"sbatch $(find -name {run_file})")[0]

            if isinstance(res, (list, tuple)):
                [msg.append(str(i)) for i in res]

            else:
                msg.append(str(res))

            if "Submitted batch" in res:
                succeed.append(i)
            else:
                failed.append(i)
                n = get_left_limit()
                msg.append(f"Error check: {n} jobs can be submit.")
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


def loop(file="paths.temp", run_file=None, node="normal3"):
    msg.append("Init Start:\n")
    with open(file, "r") as f:
        words = f.readlines()

    if os.path.isfile(f"log_{file}"):
        os.remove(f"log_{file}")

    lp = 1

    msg.append(f">>> Next Loop ({lp}): Left {len(words)} jobs >>>")

    while len(words) > 0:
        n = get_left_limit()
        data = time.time()
        timearray = time.localtime(data)
        ti = time.strftime('%Y-%m-%d %H:%M:%S', timearray)
        msg.append(str(ti))
        msg.append(f"{n} jobs can be submit.")
        if n > 0:
            sub_left_jobs(file=file, n=n, run_file=run_file,node=node)

        with open(file, "r") as f:
            words = f.readlines()

        msg.append(f"<<< Loop ({lp}) end. Left {len(words)} jobs <<<\n")

        with open(f"log_{file}", 'a+') as ff:
            ff.write("\n" + ("\n".join(msg)))
        msg.clear()

        time.sleep(150)
        lp += 1
        msg.append(f">>> Next Loop ({lp}): Left {len(words)} jobs >>>")

    msg.append(f"\nEnd")

    with open(f"log_{file}", 'a+') as ff:
        ff.write("\n" + ("\n".join(msg)))
    msg.clear()


xx_run = '''#!/bin/sh 
#SBATCH -N 1  
#SBATCH -n 28  
#SBATCH --ntasks-per-node=28
#SBATCH --partition=normal3
#SBATCH --output=%j.out 
#SBATCH --error=%j.err

source /data/home/wangchangxin/intel/oneapi/mkl/2022.0.2/env/vars.sh intel64 --force
source /data/home/wangchangxin/intel/oneapi/mpi/2021.5.1/env/vars.sh intel64 --force
source /data/home/wangchangxin/intel/oneapi/compiler/2022.0.2/env/vars.sh intel64 --force

export PATH=/data/home/wangchangxin/app/vasp.5.4.4.fix/bin:$PATH

ulimit -s unlimited

mpirun -np $SLURM_NPROCS vasp_std
'''

if __name__ == "__main__":
    loop(file="paths.temp", run_file=None,node="8360")

    # 0. file为存放所有vasp路径名称的文件，每行为绝对路径.

    # 1. 若使用vasp文件夹下，已经存在的提交脚本，直接赋值提交脚本的文件名(无需路径，若需要匹配，使用*代替)
    # 如：loop(file="paths.temp", run_file="wang_***.run")

    # 2. 若统一使用提交脚本xx_run (通常不需要更改xx_run，自动提交)
    # 如：loop(file="paths.temp", run_file=None)

    # 3. 若统一使用提交脚本, 并确实需要自定义，可以复制xx_run为xx2_run, 并修改xx2_run
    # 如：xx2_run = ...
    #    loop(file="paths.temp", run_file=xx2_run)

    # 使用方式：
    # nohup python monitor.py &
    print(__file__)
