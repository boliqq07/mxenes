# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/wcx/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/wcx/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/wcx/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/wcx/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<



# 放在.intel_bashrc,需要的时候再运行
# >>> intel 2019 environment variable
#source /opt/intel/oneapi/setvars.sh intel64
# <<< intel intel 2019 environment variable

# vasp
export PATH=/opt/software/vasp.5.4.4/bin:$PATH

#vasp
ulimit -s unlimited

export PATH=/home/wcx/vaspkit.1.3.0/bin:${PATH}

alias lob="~/bin/lobster/lobster > look &"



export PATH=/opt/intel/oneapi/mpi/2021.4.0/bin:$PATH
MPI=/opt/intel/oneapi/mpi/2021.4.0
export C_INCLUDE_PATH=$MPI/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$MPI/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$MPI/lib:$LD_LIBRARY_PATH


MPIRE=/opt/intel/oneapi/mpi/2021.4.0/lib/release
export LIBRARY_PATH=$MPIRE:$LIBRARY_PATH
export LD_LIBRARY_PATH=$MPIRE:$LD_LIBRARY_PATH


export PATH=/opt/intel/oneapi/mpi/2021.4.0/libfabric/bin:$PATH
MPI=/opt/intel/oneapi/mpi/2021.4.0/libfabric
export LIBRARY_PATH=$MPI/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$MPI/lib:$LD_LIBRARY_PATH


export PATH=/opt/intel/oneapi/mkl/2021.4.0/bin/intel64:$PATH
MKL=/opt/intel/oneapi/mkl/2021.4.0
export C_INCLUDE_PATH=$MKL/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$MKL/lib/intel64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$MKL/lib/intel64:$LD_LIBRARY_PATH


export PATH=~/libxc-5.2.0/bin:$PATH
XC=~/libxc-5.2.0
export C_INCLUDE_PATH=$XC/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$XC/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$XC/lib:$LD_LIBRARY_PATH

