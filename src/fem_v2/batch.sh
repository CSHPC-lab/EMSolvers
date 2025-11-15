#!/bin/bash -l

#SBATCH --array=1-2%1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --partition=all
#SBATCH --time=24:00:00
#SBATCH -o ./cpp_log/slurm.%j.out
#SBATCH -e ./cpp_log/slurm.%j.err

module load nvhpc

n=$(( ( SLURM_ARRAY_TASK_ID + 1 ) / 2 ))
use_ofem=$(( ( SLURM_ARRAY_TASK_ID - 1 ) % 2 ))
domain_size="3.3"
base_num=330
duration="1.0e-8"
source_position="1.64"
observation_position="1.66"
target="simulation_${SLURM_ARRAY_TASK_ID}.o"
dim_x=$SLURM_NTASKS_PER_NODE
dim_y=$(( SLURM_JOB_NUM_NODES < 4 ? SLURM_JOB_NUM_NODES : 4 ))
dim_z=$(( SLURM_JOB_NUM_NODES / dim_y ))

export OMP_NUM_THREADS=12
export PGI_ACC_TIME=1
# export PGI_ACC_NOTIFY=31
# export PGI_ACC_DEBUG=1
# export PGI_ACC_SYNCHRONOUS=1

make TARGET=$target
echo "Build completed: $target"

# 実行ファイル名（Makefileの出力ファイルに合わせて変更）
EXECUTABLE=./$target

# 総MPIプロセス数の計算
processes=$(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES))

# MPI実行(n, stabilization_factor, use_ofem)
mpirun -np $processes \
    $EXECUTABLE $n $use_ofem $domain_size $domain_size $domain_size $base_num $duration $source_position $source_position $source_position $observation_position $observation_position $observation_position $dim_x $dim_y $dim_z

echo "Execution completed."

make clean TARGET=$target