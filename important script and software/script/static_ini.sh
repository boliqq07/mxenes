#!/bin/bash

old_path=$(cd "$(dirname "$0")"; pwd)

for i in $(cat paths.temp)

do
cd $i
echo $(cd "$(dirname "$0")"; pwd)

cd .. 
rm -rf ini_static 
cp -r ini_opt ini_static
cp ini_opt/CONTCAR ini_static/POSCAR
cp $old_path/static_INCAR ini_static/INCAR

cd $old_path
done
        