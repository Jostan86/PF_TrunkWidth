#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -o /nfs/hpc/share/browjost/pf_eval/logdirs/output_pf_test_bin_angle_2_%a.out
#SBATCH -e /nfs/hpc/share/browjost/pf_eval/logdirs/errors_pf_test_bin_angle_2_%a.err
#SBATCH -J pf_test_bin_angle_2						  # name of job
#SBATCH -p share
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1


module load python3/3.8
source /nfs/hpc/share/browjost/pf_eval/venv/bin/activate

#run my job (e.g. matlab, python)
srun --ntasks=1 --export ALL python3 headless_eval.py --benchmark \
                                           -dir_out "/nfs/hpc/share/browjost/pf_eval/results/" \
                                           -dir_data "/nfs/hpc/share/browjost/pf_eval/data/" \
                                           -nt 20 \
                                           -bin_angle 8 \
                                           --verbose \
                                           -name "bin_angle-8" &
srun --ntasks=1 --export ALL python3 headless_eval.py --benchmark \
                                           -dir_out "/nfs/hpc/share/browjost/pf_eval/results/" \
                                           -dir_data "/nfs/hpc/share/browjost/pf_eval/data/" \
                                           -nt 20 \
                                           --verbose \
                                           -bin_angle 2 \
                                           -name "bin_angle-2" &
wait
