#!/bin/bash -l

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=40g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH -o ./cpp_log/slurm.%j.out
#SBATCH -e ./cpp_log/slurm.%j.err

. /etc/profile.d/modules.sh
module load nvhpc-openmpi3/24.7

export OMP_NUM_THREADS=12
export PGI_ACC_TIME=1
# export PGI_ACC_NOTIFY=31
# export PGI_ACC_DEBUG=1
# export PGI_ACC_SYNCHRONOUS=1

# Makefileでコンパイル
echo "Compiling with Makefile..."
make clean
make CXX=nvc++

# 実行ファイル名（Makefileの出力ファイルに合わせて変更）
EXECUTABLE=./fem_simulation

# 総MPIプロセス数の計算
processes=$(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES))

echo "Running with $processes MPI processes..."

# MPI実行
mpirun -np $processes $EXECUTABLE