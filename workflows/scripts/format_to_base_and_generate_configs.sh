#!/usr/bin/env bash
best_concept=$(basename -s ".yaml" $1)
root=$(dirname $(dirname $1))
scripts/format_to_base_coinfig.py $1
scripts/gen_configs.py $1 $root/$best_concept 
#echo $1
#echo $1 $root/$best_concept
echo $(ls -1 $root/$best_concept | wc -l) lines in $root/$best_concept
echo Marking $1 as proccessed...
chmod -x $1
