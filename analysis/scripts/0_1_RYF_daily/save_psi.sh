#!/bin/bash
#PBS -P jk72
#PBS -q normalbw
#PBS -l mem=250gb
#PBS -l ncpus=28
#PBS -l walltime=4:00:00
#PBS -l storage=gdata/ik11+gdata/jk72+gdata/v45+gdata/qv56+gdata/hh5+gdata/cj50
#PBS -l jobfs=400gb
#PBS -N save_psi_year
#PBS -j oe
#PBS -v year

# submit with:
# qsub -v year=2170 save_psi.sh

module use /g/data3/hh5/public/modules
module load conda/analysis3

cd /g/data/jk72/ed7737/SO-channel_embayment/analysis/scripts/0_1_RYF_daily

# call python
python3 calc_uh_vh_h_binned.py $year &>> output_save_psi_10_$year.txt

exit
