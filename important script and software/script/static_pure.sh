#!/bin/bash

old_path=$(cd "$(dirname "$0")"; pwd)

for i in $(cat paths.temp)

do
cd $i
echo $(cd "$(dirname "$0")"; pwd)

cd .. 
rm -rf pure_static 
cp -r pure_opt pure_static 
cp pure_opt/CONTCAR pure_static/POSCAR 
cp $old_path/static_INCAR pure_static/INCAR

cd $old_path
done
        