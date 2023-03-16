#!/bin/bash

# 替换命令必须用单引号
old_cmd='NCORE = *'
new_cmd='NCORE = 16'

old_path=$(cd "$(dirname "$0")"; pwd)

for i in $(cat paths.temp)
do
cd $i
echo $(cd "$(dirname "$0")"; pwd)

res=$(grep "$old_cmd" INCAR)

if [[ "$res" != "" ]]
then
  cmd="s/^"$res"/"$new_cmd"/g"
  sed -i "$cmd" INCAR
else
  sed -i '$a '"$new_cmd"  INCAR
fi

cd $old_path
done
        