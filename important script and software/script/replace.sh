#!/bin/bash

old_path=$(cd "$(dirname "$0")"; pwd)

for i in $(cat paths.temp)

do
cd $i
echo $(cd "$(dirname "$0")"; pwd)

sed -i 's/^"NCORE = 8"/"NCORE = 4"/g' INCAR

cd $old_path
done
        