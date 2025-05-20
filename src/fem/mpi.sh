#!/bin/bash -l

#SBATCH --gres=gpu:4
#SBATCH --nodes=8
#SBATCH --partition=all
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH -o ./fortran_log/slurm.%j.out
#SBATCH -e ./fortran_log/slurm.%j.err


. /etc/profile.d/modules.sh
module load nvhpc/21.3


# export PGI_ACC_TIME=1
# export PGI_ACC_NOTIFY=31
# export PGI_ACC_DEBUG=1
# export PGI_ACC_SYNCHRONOUS=1
export OMP_NUM_THREADS=12

rm -f ./results/*
# rm -f ./results_v/*
rm -f ./fourier/*
rm -f test_openacc

FILENAME="scalar_wave_fem_3d_manufacture.f90"
CONSTANTS="constants.f90"
num_replacements=0
processes=$(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES))
# processes=50
INFOFILE=./fortran_log/slurm.${SLURM_JOB_ID}.info
NCUOUTPUT=ncu_${SLURM_JOB_ID}
echo "Compiling $FILENAME"

mpif90 -c -acc $CONSTANTS -o constants.o

mpif90 -O3 -lm -llapack -mp=multicore -Mcuda=ptxinfo -Minfo=accel -acc -gpu=cc80,cuda11.2,fastmath,pinned $FILENAME pds8.f90 pds27_.f90 PDSvox64_ave_.f pds125.f90 pds216.f90 pds343.f90 pds512.f90 constants.o -o test_openacc 2> $INFOFILE


if [ $? -ne 0 ]; then
    echo "Compilation failed. See infofile for details."
    exit 1
fi
echo 'run with openacc'
mpirun -np $processes ./test_openacc
# nsys profile mpirun -np $processes ./test_openacc
# ncu -o $NCUOUTPUT -f --set full --target-processes all mpirun -np $processes ./test_openacc

if [ $? -ne 0 ]; then
    echo "Execution failed."
fi

for i in $(seq 1 $num_replacements); do
    rm ./*.DAT
    sed "s/target.txt/target${i}.txt/; s/answer.txt'/answer${i}.txt'/" "misfit/HF_TF-MISFIT_GOF_MASTER" > "HF_TF-MISFIT_GOF"
    ./misfit/tf_misfits_gof
    mkdir -p ./misfit/results/${SLURM_JOB_ID}
    mv ./MISFIT-GOF.DAT ./misfit/results/${SLURM_JOB_ID}/${i}.DAT
    rm ./*.DAT
done


# rm ./*.DAT
# sed "s/target.txt/target1.txt/; s/answer.txt'/answer1.txt'/" "misfit/HF_TF-MISFIT_GOF_MASTER" > "HF_TF-MISFIT_GOF"
# ./misfit/tf_misfits_gof
# mkdir -p ./misfit/results/${SLURM_JOB_ID}
# mv ./MISFIT-GOF.DAT ./misfit/results/${SLURM_JOB_ID}/1.DAT
# rm ./*.DAT
# sed "s/target.txt/target4.txt/; s/answer.txt'/answer4.txt'/" "misfit/HF_TF-MISFIT_GOF_MASTER" > "HF_TF-MISFIT_GOF"
# ./misfit/tf_misfits_gof
# mkdir -p ./misfit/results/${SLURM_JOB_ID}
# mv ./MISFIT-GOF.DAT ./misfit/results/${SLURM_JOB_ID}/4.DAT
# rm ./*.DAT
# sed "s/target.txt/target8.txt/; s/answer.txt'/answer8.txt'/" "misfit/HF_TF-MISFIT_GOF_MASTER" > "HF_TF-MISFIT_GOF"
# ./misfit/tf_misfits_gof
# mkdir -p ./misfit/results/${SLURM_JOB_ID}
# mv ./MISFIT-GOF.DAT ./misfit/results/${SLURM_JOB_ID}/8.DAT
# rm ./*.DAT