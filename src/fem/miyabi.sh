#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1:mpiprocs=1:ompthreads=72
#PBS -l walltime=00:30:00
#PBS -W group_list=gb25
#PBS -j oe
#PBS -o ./cpp_log/slurm.out
#PBS -e ./cpp_log/slurm.err

#------ Environment module ------#
module purge
module load nvidia/24.9
module load nv-hpcx/24.9

export OMP_NUM_THREADS=72
export PGI_ACC_TIME=1
# export PGI_ACC_NOTIFY=31
# export PGI_ACC_DEBUG=1
# export PGI_ACC_SYNCHRONOUS=1

cd $PBS_O_WORKDIR

# コンパイル
echo "Compiling..."
make clean
make CXX=nvc++

# 実行
EXECUTABLE=./fem_simulation
mpirun -np 1 $EXECUTABLE

mv ./cpp_log/slurm.out ./cpp_log/slurm.${PBS_JOBID}.out
mv ./cpp_log/slurm.err ./cpp_log/slurm.${PBS_JOBID}.err