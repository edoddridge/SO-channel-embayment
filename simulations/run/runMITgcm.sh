#!/bin/bash

#PBS -l ncpus=144
#PBS -l mem=190GB
#PBS -l jobfs=200GB
#PBS -q normal
#PBS -P jk72
#PBS -l walltime=28:00:00
#PBS -l storage=gdata/jk72+scratch/jk72
#PBS -l wd
#PBS -N SO-channel_embayment

module load intel-compiler
module load intel-mpi

ln -s ../input/* .

mpirun -n 144 ./mitgcmuv
