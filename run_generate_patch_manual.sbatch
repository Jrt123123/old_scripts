#!/bin/bash
#SBATCH -J UnicycleLatentTrain
#SBATCH --account=gts-gchou3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1        # 1 main + 8 workers + spare
#SBATCH --cpus-per-task=2        # use 4 CPU cores for dataloading/threads
# #SBATCH --mem-per-cpu=0G            # ~64GB total
#SBATCH --mem=8G                  # Request total memory directly
#SBATCH -t 8:00:00                 # 24 hours (adjust as needed)
#SBATCH -q embers
#SBATCH --gres=gpu:1                # use 1 GPU
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rjiang77@gatech.edu
#SBATCH --array=100-105



cd $SLURM_SUBMIT_DIR
module purge
module load anaconda3/2023.03



conda activate dino_wm




# srun python run_script_3.py
python run_unicycle_patch_generate_manual_s.py

echo "Job started on $(hostname)"
echo "Array ID = $SLURM_ARRAY_TASK_ID"

# srun python concat_all_data.py