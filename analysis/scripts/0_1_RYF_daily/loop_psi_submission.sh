#!/bin/bash

## loop over count, submit job to gadi with count that gets communicated to python

for i in {2170..2179}
do
   echo "creating job for year $i"
   qsub -v year=$i save_psi.sh
done
