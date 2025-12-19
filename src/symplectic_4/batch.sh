#!/bin/bash -l

#SBATCH --array=1-12%1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=8
#SBATCH --partition=all
#SBATCH --time=1:00:00
#SBATCH -o ./cpp_log/slurm.%j.out
#SBATCH -e ./cpp_log/slurm.%j.err

module load nvhpc

processes=$(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES))
n=$(( ( SLURM_ARRAY_TASK_ID + 1 ) / 2 ))
use_ofem=$(( ( SLURM_ARRAY_TASK_ID - 1 ) % 2 ))
domain_size="3.0"
base_num=300
duration="9.0e-9"
source_position_x="1.10"
observation_position_x="1.90"
source_position_y="1.10"
observation_position_y="1.90"
source_position_z="1.10"
observation_position_z="1.90"
distance_x=$(echo "$observation_position_x - $source_position_x" | bc -l)
distance_x=$(printf "%.3f" "$distance_x")
distance_y=$(echo "$observation_position_y - $source_position_y" | bc -l)
distance_y=$(printf "%.3f" "$distance_y")
distance_z=$(echo "$observation_position_z - $source_position_z" | bc -l)
distance_z=$(printf "%.3f" "$distance_z")
distance="${distance_x}_${distance_y}_${distance_z}"
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
EXECUTABLE=./$target

mpirun -np $processes \
    $EXECUTABLE $n $use_ofem $domain_size $domain_size $domain_size $base_num $duration $source_position_x $source_position_y $source_position_z $observation_position_x $observation_position_y $observation_position_z $dim_x $dim_y $dim_z

make clean TARGET=$target
DIR="results/$distance"
mkdir -p $DIR
mv *.csv $DIR/
echo "Execution completed."