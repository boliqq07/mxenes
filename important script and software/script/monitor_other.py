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


def get_node_run(run_file=None, name="cpu"):
    if run_file is None:
        run_file = xx_run
    if name == "normal":
        run_file = run_file.replace("-q squeue", "-q normal")
        run_file = run_file.replace("-n nn", "-n 40")
        run_file = run_file.replace("ptile=nn", "ptile=40")
    elif name == "fat":
        run_file = run_file.replace("-q squeue", "-q fat")
        run_file = run_file.replace("-n nn", "-n 96")
        run_file = run_file.replace("ptile=nn", "ptile=96")
    else:
        run_file = run_file.replace("-q squeue", "-q cpu")
        run_file = run_file.replace("-n nn", "-n 64")
        run_file = run_file.replace("ptile=nn", "ptile=64")
    return run_file


def get_avail_num():
    res = cmd_popen("jhosts")
    res = [i.split(" ") for i in res]
    res = [[ii for ii in i if ii != ''] for i in res]

    normal_n = 0
    for resi in res[2:41]:
        if resi[4] == "0" and resi[5] == "0":
            normal_n += 1
    cpu_n = 0
    for resi in res[41:71]:
        if resi[4] == "0" and resi[5] == "0":
            cpu_n += 1
    return {"cpu": cpu_n, "normal": normal_n}


def dispatch(n):
    data = get_avail_num()
    msg.append(f"Available node: {str(data)}")
    base = n // 3
    left = n % 3

    cpu = base * 2 + left
    normal = base

    diff = data["cpu"] - data["normal"] - 2
    if diff > 0 or all([data["cpu"] == 0, data["normal"]>=5]):
        diff = max(diff, -3)
        cpu = diff + cpu
        normal = -diff + normal

    normal = int(max(normal, 0))
    cpu = int(max(cpu, 0))
    return ["normal"] * normal + ["cpu"] * cpu



def get_left_limit(up_limit=29):

    res = cmd_popen("jjobs|grep -wo 'PEND'|wc -l")[0]
    num = int(res)
    left_num = up_limit - num
    return max(left_num, 0)


def sub_left_jobs(file="paths.temp", n=1, run_file=None):
    with open(file, "r") as f:
        words = f.readlines()

    words = [i.replace("\n", "") for i in words]

    to_word = words[:n]
    left_word = words[n:]
    node_mark = dispatch(n)
    node_mark.reverse()
    node_mark = node_mark[:n]

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
                with open(f"gjj_{node_marki}.run", "w") as ff:
                    ff.write(get_node_run(run_file=None, name=node_marki))
                res = cmd_popen(f"jsub < $(find -name gjj_{node_marki}.run)")[0]
            elif "#!/bin" in run_file:
                with open(f"gjj_special.run", "w") as ff:
                    ff.write(get_node_run(run_file=run_file, name=node_marki))
                res = cmd_popen(f"jsub < $(find -name gjj_special.run)")[0]
            else:
                res = cmd_popen(f"jsub < $(find -name {run_file})")[0]

            if isinstance(res, (list, tuple)):
                [msg.append(str(i)) for i in res]

            else:
                msg.append(str(res))

            if "is submitted" in res:
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


def loop(file="paths.temp", run_file=None):

    import getpass

    if getpass.getuser() != "yangmei":
        pass
    else:
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
                sub_left_jobs(file=file, n=n, run_file=run_file)

            with open(file, "r") as f:
                words = f.readlines()

            msg.append(f"<<< Loop ({lp}) end. Left {len(words)} jobs <<<\n")

            with open(f"log_{file}", 'a+') as ff:
                ff.write("\n" + ("\n".join(msg)))
            msg.clear()

            time.sleep(600)
            lp += 1
            msg.append(f">>> Next Loop ({lp}): Left {len(words)} jobs >>>")

        msg.append(f"\nEnd")

        with open(f"log_{file}", 'a+') as ff:
            ff.write("\n" + ("\n".join(msg)))
        msg.clear()


xx_run = '''
#!/bin/sh
#JSUB -J ccy       
#JSUB -n nn
#JSUB -R span[ptile=nn]        
#JSUB -q squeue
#JSUB -o out.%J                  
#JSUB -e err.%J

source /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/bin/mklvars.sh intel64
source /opt/intel/compilers_and_libraries_2020.1.217/linux/mpi/intel64/bin/mpivars.sh intel64
source /opt/intel/compilers_and_libraries_2020/linux/bin/compilervars.sh intel64

ulimit -s 5120000

source /beegfs/jhinno/unischeduler/conf/unisched
export PATH=~/app/vasp.5.4.4.fix/:$PATH

########################################################
#   $JH_NCPU:         Number of CPU cores              #
#   $JH_HOSTFILE:     List of computer hostfiles       #
########################################################

mpirun -np $JH_NCPU -machinefile $JH_HOSTFILE vasp_std  > vasp.log

'''

if __name__ == "__main__":
    print(__file__)
    loop(file="paths.temp", run_file=None)

    # 0. file为存放所有vasp路径名称的文件.

    # 1. 若使用vasp文件夹下，已经存在的提交脚本，直接赋值提交脚本的文件�?无需路径，若需要匹配，使用*代替)
    # 如：loop(file="paths.temp", run_file="***.run")

    # 2. 若统一使用提交脚本xx_run (通常不需要更改xx_run，此代码自动判断normal, cpu节点空余，并自动提交)
    # 如：loop(file="paths.temp", run_file=None)

    # 3. 若统一使用提交脚本, 并确实需要自定义，可以复制xx_run为xx2_run, 并修改xx2_run
    # 如：xx2_run = ...
    #    loop(file="paths.temp", run_file=xx2_run)

    # 使用方式
    # nohup python monitor.py &
    # 运行过程中，请勿更改 "paths.temp"

    print(__file__)
