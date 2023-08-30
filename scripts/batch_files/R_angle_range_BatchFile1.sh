#!/bin/bash
#SBATCH -t 0-12:00:00
#SBATCH -J pf_test_sd_w1						  # name of job
#SBATCH -p share								  # name of partition or queue
#SBATCH -o /nfs/hpc/share/browjost/pf_eval/logdirs/output_%a.out			  # name of output file for this submission script
#SBATCH -e /nfs/hpc/share/browjost/pf_eval/logdirs/errors_%a.err				  # name of error file for this submission script1
#SBATCH -N 2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1


module load python3/3.8
source /nfs/hpc/share/browjost/pf_eval/venv/bin/activate

# run my job (e.g. matlab, python)
#srun --ntasks=1 --export ALL python3 headless_eval.py --benchmark \
#                                           -dir_out "/nfs/hpc/share/browjost/pf_eval/results/" \
#                                           -dir_data "/nfs/hpc/share/browjost/pf_eval/data/" \
#                                           -nt 20 \
#                                           -r_angle 5 \
#                                           -name "r_dist-5" &
srun --ntasks=1 --export ALL python3 headless_eval.py --benchmark \
                                           -dir_out "/nfs/hpc/share/browjost/pf_eval/results/" \
                                           -dir_data "/nfs/hpc/share/browjost/pf_eval/data/" \
                                           -nt 20 \
                                           -r_angle 10 \
                                           -name "r_angle-10" &
srun --ntasks=1 --export ALL python3 headless_eval.py --benchmark \
                                           -dir_out "/nfs/hpc/share/browjost/pf_eval/results/" \
                                           -dir_data "/nfs/hpc/share/browjost/pf_eval/data/" \
                                           -nt 20 \
                                           -r_angle 15 \
                                           -name "r_angle-15" &
wait
