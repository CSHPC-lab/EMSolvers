#!/bin/bash -l

#SBATCH --array=2-2%1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --partition=80g
#SBATCH --time=24:00:00
#SBATCH -o ./cpp_log/slurm.%j.out
#SBATCH -e ./cpp_log/slurm.%j.err

module load nvhpc

use_ofem="1"
domain_size="1.0"
duration="3.0e-9"
source_position="0.48"
observation_position="0.50"
target="simulation_${SLURM_ARRAY_TASK_ID}_${use_ofem}_${domain_size}_${duration}_${source_position}_${observation_position}.o"
dim_x=2
dim_y=1
dim_z=1

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
    $EXECUTABLE ${SLURM_ARRAY_TASK_ID} $use_ofem $domain_size $domain_size $domain_size $duration $source_position $source_position $source_position $observation_position $observation_position $observation_position $dim_x $dim_y $dim_z

echo "Execution completed."

make clean TARGET=$target