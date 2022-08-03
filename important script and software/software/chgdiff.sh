#!/bin/bash

old_path=$(cd "$(dirname "$0")"; pwd)

paste  paths1.temp paths2.temp| while read chg1 chg2;
do

echo "Try to" $chg2 "-" $chg1 ">>>"

n1=${chg1//\//_}
n2=${chg2//\//_}

~/bin/chgdiff.pl $chg1/CHGCAR $chg2/CHGCAR

mv CHGCAR_diff $n1$n2-"CHGCAR_diff"

echo $n1$n2-"CHGCAR_diff" "store in" $old_path


done