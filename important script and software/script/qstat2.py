# -*- coding: utf-8 -*-

# @Time  : 2022/8/8 21:36
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License
import os
import re

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def get_message(msg, names=('Job_Id', 'job_state', 'comment', 'init_work_dir',)):
    data_all = {}
    for name in names:
        s1 = re.findall(r"\<{name}\>.*\</{name}\>".format(name=name), msg)
        data = {}
        for n, si in enumerate(s1):

            si = si.replace("<{name}>".format(name=name), "")
            si = si.replace("</{name}>".format(name=name), "")

            if name == "comment":

                if "Not Running" in si:
                    si = "N"
                else:
                    try:
                        si = '-'.join(si.split(" ")[-4:])
                    except:
                        si = "N"

            if name == "init_work_dir":
                home = os.path.expanduser('~')
                si = si.replace(home, "~")

            data.update({str(n): si})
        name = "State" if name == "job_state" else name
        name = "Dir" if name == "init_work_dir" else name
        name = "Start Time" if name == "comment" else name

        data_all.update({name: data})
    return data_all


msg = os.popen("qstat -x").readlines()
msg = "".join(msg)

# msg = """
# <?xml version="1.0"?>
# <Data><Job><Job_Id>72.c0</Job_Id><Job_Name>vasp</Job_Name><Job_Owner>wcx@localhost</Job_Owner><resources_used><cput>02:07:06</cput><energy_used>0</energy_used><mem>8942880kb</mem><vmem>285471748kb</vmem><walltime>00:02:57</walltime></resources_used><job_state>R</job_state><queue>master</queue><server>c0</server><Checkpoint>u</Checkpoint><ctime>1659969957</ctime><Error_Path>c0:/home/wcx/data/W2CO2_add/W2CO2/Cd/pure_static/vasp.e72</Error_Path><exec_host>c0/0-43</exec_host><Hold_Types>n</Hold_Types><Join_Path>n</Join_Path><Keep_Files>n</Keep_Files><Mail_Points>a</Mail_Points><mtime>1659969957</mtime><Output_Path>c0:/home/wcx/data/W2CO2_add/W2CO2/Cd/pure_static/vasp.o72</Output_Path><Priority>0</Priority><qtime>1659969957</qtime><Rerunable>True</Rerunable><Resource_List><nodes>1:ppn=44</nodes><walltime>120:00:00</walltime><nodect>1</nodect></Resource_List><session_id>51549</session_id><Shell_Path_List>/bin/bash</Shell_Path_List><Variable_List>PBS_O_QUEUE=master,PBS_O_HOME=/home/wcx,PBS_O_LOGNAME=wcx,PBS_O_PATH=/home/wcx/app/vaspkit.1.2.1/bin:/opt/intel/oneapi/mpi/2021.5.1//libfabric/bin:/opt/intel/oneapi/mpi/2021.5.1//bin:/opt/intel/oneapi/compiler/2022.0.2/linux/lib/oclfpga/bin:/opt/intel/oneapi/compiler/2022.0.2/linux/bin/intel64:/opt/intel/oneapi/compiler/2022.0.2/linux/bin:/opt/intel/oneapi/mkl/2022.0.2/bin/intel64:/home/wcx/anaconda3/bin:/home/wcx/anaconda3/condabin:/usr/local/torque/bin:/usr/local/torque/sbin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/usr/local/torque/bin:/usr/local/torque/sbin:/opt/vasp.5.4.4.fix/bin:/home/wcx/bin:/home/wcx/.local/bin:/home/wcx/bin,PBS_O_MAIL=/var/spool/mail/wcx,PBS_O_SHELL=/bin/bash,PBS_O_LANG=zh_CN.UTF-8,PBS_O_WORKDIR=/home/wcx/data/W2CO2_add/W2CO2/Cd/pure_static,PBS_O_HOST=c0,PBS_O_SERVER=localhost.localdomain</Variable_List><euser>wcx</euser><egroup>wcx</egroup><queue_type>E</queue_type><comment>Job started on Mon Aug 08 at 22:45</comment><etime>1659969957</etime><submit_args>pbs.run</submit_args><start_time>1659969957</start_time><Walltime><Remaining>431771</Remaining></Walltime><start_count>1</start_count><fault_tolerant>False</fault_tolerant><job_radix>0</job_radix><submit_host>c0</submit_host><init_work_dir>/home/wcx/data/W2CO2_add/W2CO2/Cd/pure_static</init_work_dir><request_version>1</request_version></Job><Job><Job_Id>73.c0</Job_Id><Job_Name>vasp</Job_Name><Job_Owner>wcx@localhost</Job_Owner><job_state>Q</job_state><queue>master</queue><server>c0</server><Checkpoint>u</Checkpoint><ctime>1659969970</ctime><Error_Path>c0:/home/wcx/data/W2CO2_add/W2CO2-H/Co/H/S1/ini_static/vasp.e73</Error_Path><Hold_Types>n</Hold_Types><Join_Path>n</Join_Path><Keep_Files>n</Keep_Files><Mail_Points>a</Mail_Points><mtime>1659969970</mtime><Output_Path>c0:/home/wcx/data/W2CO2_add/W2CO2-H/Co/H/S1/ini_static/vasp.o73</Output_Path><Priority>0</Priority><qtime>1659969970</qtime><Rerunable>True</Rerunable><Resource_List><nodes>1:ppn=44</nodes><walltime>120:00:00</walltime><nodect>1</nodect></Resource_List><Shell_Path_List>/bin/bash</Shell_Path_List><Variable_List>PBS_O_QUEUE=master,PBS_O_HOME=/home/wcx,PBS_O_LOGNAME=wcx,PBS_O_PATH=/home/wcx/app/vaspkit.1.2.1/bin:/opt/intel/oneapi/mpi/2021.5.1//libfabric/bin:/opt/intel/oneapi/mpi/2021.5.1//bin:/opt/intel/oneapi/compiler/2022.0.2/linux/lib/oclfpga/bin:/opt/intel/oneapi/compiler/2022.0.2/linux/bin/intel64:/opt/intel/oneapi/compiler/2022.0.2/linux/bin:/opt/intel/oneapi/mkl/2022.0.2/bin/intel64:/home/wcx/anaconda3/bin:/home/wcx/anaconda3/condabin:/usr/local/torque/bin:/usr/local/torque/sbin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/usr/local/torque/bin:/usr/local/torque/sbin:/opt/vasp.5.4.4.fix/bin:/home/wcx/bin:/home/wcx/.local/bin:/home/wcx/bin,PBS_O_MAIL=/var/spool/mail/wcx,PBS_O_SHELL=/bin/bash,PBS_O_LANG=zh_CN.UTF-8,PBS_O_WORKDIR=/home/wcx/data/W2CO2_add/W2CO2-H/Co/H/S1/ini_static,PBS_O_HOST=c0,PBS_O_SERVER=localhost.localdomain</Variable_List><euser>wcx</euser><egroup>wcx</egroup><queue_type>E</queue_type><comment>Not Running: Not enough of the right type of nodes are available</comment><etime>1659969970</etime><submit_args>pbs.run</submit_args><fault_tolerant>False</fault_tolerant><job_radix>0</job_radix><submit_host>c0</submit_host><init_work_dir>/home/wcx/data/W2CO2_add/W2CO2-H/Co/H/S1/ini_static</init_work_dir><request_version>1</request_version></Job></Data>
# <Data><Job><Job_Id>73.c0</Job_Id><Job_Name>vasp</Job_Name><Job_Owner>wcx@localhost</Job_Owner><resources_used><cput>01:36:09</cput><energy_used>0</energy_used><mem>9727188kb</mem><vmem>285784940kb</vmem><walltime>00:02:14</walltime></resources_used><job_state>C</job_state><queue>master</queue><server>c0</server><Checkpoint>u</Checkpoint><ctime>1659969970</ctime><Error_Path>c0:/home/wcx/data/W2CO2_add/W2CO2-H/Co/H/S1/ini_static/vasp.e73</Error_Path><exec_host>c0/0-43</exec_host><Hold_Types>n</Hold_Types><Join_Path>n</Join_Path><Keep_Files>n</Keep_Files><Mail_Points>a</Mail_Points><mtime>1659970747</mtime><Output_Path>c0:/home/wcx/data/W2CO2_add/W2CO2-H/Co/H/S1/ini_static/vasp.o73</Output_Path><Priority>0</Priority><qtime>1659969970</qtime><Rerunable>True</Rerunable><Resource_List><nodes>1:ppn=44</nodes><walltime>120:00:00</walltime><nodect>1</nodect></Resource_List><session_id>52013</session_id><Shell_Path_List>/bin/bash</Shell_Path_List><Variable_List>PBS_O_QUEUE=master,PBS_O_HOME=/home/wcx,PBS_O_LOGNAME=wcx,PBS_O_PATH=/home/wcx/app/vaspkit.1.2.1/bin:/opt/intel/oneapi/mpi/2021.5.1//libfabric/bin:/opt/intel/oneapi/mpi/2021.5.1//bin:/opt/intel/oneapi/compiler/2022.0.2/linux/lib/oclfpga/bin:/opt/intel/oneapi/compiler/2022.0.2/linux/bin/intel64:/opt/intel/oneapi/compiler/2022.0.2/linux/bin:/opt/intel/oneapi/mkl/2022.0.2/bin/intel64:/home/wcx/anaconda3/bin:/home/wcx/anaconda3/condabin:/usr/local/torque/bin:/usr/local/torque/sbin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/usr/local/torque/bin:/usr/local/torque/sbin:/opt/vasp.5.4.4.fix/bin:/home/wcx/bin:/home/wcx/.local/bin:/home/wcx/bin,PBS_O_MAIL=/var/spool/mail/wcx,PBS_O_SHELL=/bin/bash,PBS_O_LANG=zh_CN.UTF-8,PBS_O_WORKDIR=/home/wcx/data/W2CO2_add/W2CO2-H/Co/H/S1/ini_static,PBS_O_HOST=c0,PBS_O_SERVER=localhost.localdomain</Variable_List><euser>wcx</euser><egroup>wcx</egroup><queue_type>E</queue_type><sched_hint>Unable to copy files back - please see the mother superior&apos;s log for exact details.</sched_hint><comment>Job started on Mon Aug 08 at 22:56</comment><etime>1659969970</etime><exit_status>0</exit_status><submit_args>pbs.run</submit_args><start_time>1659970605</start_time><start_count>1</start_count><fault_tolerant>False</fault_tolerant><comp_time>1659970747</comp_time><job_radix>0</job_radix><total_runtime>141.720458</total_runtime><submit_host>c0</submit_host><init_work_dir>/home/wcx/data/W2CO2_add/W2CO2-H/Co/H/S1/ini_static</init_work_dir><request_version>1</request_version></Job></Data>
# """

if msg is None or len(msg) < 100:
    print("No Jobs Found.")
else:
    msg = msg.replace("><", ">\n<")
    res = get_message(msg)
    res = pd.DataFrame.from_dict(res)
    print(res)
