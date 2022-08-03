#!/bin/bash

old_path=$(cd "$(dirname "$0")"; pwd)

for i in $(cat paths.temp)

do
cd $i
echo $(cd "$(dirname "$0")"; pwd)

cp /beegfs/home/yangmei/wcx/W2CO2/gjj_ym.run ./

cd $old_path
done
        