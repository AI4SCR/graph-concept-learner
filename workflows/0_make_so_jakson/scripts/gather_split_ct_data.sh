#!/bin/bash

# Unpack input
z_pg=$1
b_pg=$2
z_mtc=$3
b_mtc=$4
z_scl=$5
b_scl=$6
map=$7
out=$8

# Mkdir outdir and cd into it
mkdir -p $out
cd $out

# Make two lists, one for each cohort, and sample specifi files
z=($z_pg,$z_mtc,$z_scl)
b=($b_pg,$b_mtc,$b_scl)

# Change header of b_pg
sed -i".bak" 's/PhenoGraphBasel/PhenoGraph/g' $b_pg

# Path to q
q=/u/scastro/q-3.1.6/bin/q.py

# Iterte over lists
for l in $z $b; do
    # Unpack paths
    pg=$(echo $l | cut -d "," -f 1)
    mtc=$(echo $l | cut -d "," -f 2)
    scl=$(echo $l | cut -d "," -f 3)

    # Preprocces map then join cell id with cell type data, and the with cell location data
    sed 's/;/,/g' $map |
    sed -E 's/[[:blank:]]+/_/g' |
    $q -O -H -d',' "SELECT mtc.id,mtc.cluster,map.Cell_Type,map.Class FROM $mtc mtc JOIN - map ON (mtc.cluster = map.Metacluster_)" |
    $q -O -H -d',' "SELECT pg.id,mtc.cluster,mtc.Cell_Type,mtc.Class,pg.PhenoGraph FROM $pg pg LEFT JOIN - mtc ON (pg.id = mtc.id)" |
    $q -O -H -d',' "SELECT scl.core,scl.ObjectNumber_renamed,scl.id,mc.cluster,mc.Cell_Type,mc.Class,mc.PhenoGraph,scl.Location_Center_X,scl.Location_Center_Y FROM $scl scl LEFT JOIN - mc ON (mc.id = scl.id) ORDER BY scl.core" |
    awk -F ',' '
        BEGIN {
            file_name="core"
        }
        {
            if(file_name==$1) {
                print > $1
            }
            else {
                close(file_name)
                file_name=$1
                print > $1
            }
        }
    '
done

# Rename with csv extension
for f in $(ls); do
	mv $f $f.csv
done
