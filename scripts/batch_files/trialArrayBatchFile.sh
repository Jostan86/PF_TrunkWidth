#!/bin/bash
#SBATCH --array=1-3
#SBATCH -J pf_test_1						  # name of job
#SBATCH -p share								  # name of partition or queue
#SBATCH -o /nfs/hpc/share/browjost/pf_eval/logdirs/output_%a.out			  # name of output file for this submission script
#SBATCH -e /nfs/hpc/share/browjost/pf_eval/logdirs/errors_%a.err				  # name of error file for this submission script1

case $SLURM_ARRAY_TASK_ID in
    1)
        VALUE=0.1
        ;;
    2)
        VALUE=0.2
        ;;
    3)
        VALUE=0.3
        ;;
    *)
        echo "Invalid array index"
        exit 1
        ;;
esac


module load python3/3.8
source /nfs/hpc/share/browjost/pf_eval/venv/bin/activate

# run my job (e.g. matlab, python)
srun --export ALL python3 headless_eval.py --benchmark \
                                           -dir_out "/nfs/hpc/share/browjost/pf_eval/results/" \
                                           -dir_data "/nfs/hpc/share/browjost/pf_eval/data/" \
                                           -nt 3 \
                                           -sd_d $VALUE \
                                           -name "sd_d_test-$VALUE" &

