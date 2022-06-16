module load intel-compiler
module load intel-mpi

/home/552/ed7737/MITgcm/tools/genmake2 -mpi -rootdir=/home/552/ed7737/MITgcm/ -mods=../code -optfile=linux_amd64_ifort_gadi

make depend

make -j 16

cp mitgcmuv ../run
