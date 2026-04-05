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
#SBATCH --array=200-219

export TORCH_HOME=/storage/scratch1/6/rjiang77/torch_cache
mkdir -p $TORCH_HOME


cd $SLURM_SUBMIT_DIR
module load anaconda3/2023.03



conda activate  /storage/scratch1/6/rjiang77/conda_envs/torch_env

# srun python -m experiments.unicycle.train.run_script_3
srun python -m experiments.unicycle.jobs.run_unicycle_generate_straight
# srun python -m experiments.unicycle.data.concat_all_data