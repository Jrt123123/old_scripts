#!/bin/bash
#SBATCH -J UnicycleLatentTrain
#SBATCH --account=gts-gchou3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1        # 1 main + 8 workers + spare
#SBATCH --cpus-per-task=6        # use 4 CPU cores for dataloading/threads

#SBATCH --mem=48G                  # Request total memory directly
#SBATCH -t 8:00:00                 # 24 hours (adjust as needed)
#SBATCH --qos=embers
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rjiang77@gatech.edu
# #SBATCH --array=20-39




cd $SLURM_SUBMIT_DIR
module load anaconda3/2023.03



#conda activate  /storage/scratch1/6/rjiang77/conda_envs/torch_env
# conda activate dino_wm
conda activate /storage/scratch1/6/rjiang77/conda_envs_scratch/dino_env_scratch


#srun python run_script_11.py


export WANDB_DIR=/storage/scratch1/6/rjiang77/wandb
export WANDB_MODE=online
export HYDRA_FULL_ERROR=1
export DATASET_DIR= /storage/scratch1/6/rjiang77/DINO_WM_git/datasets/data
cd /storage/scratch1/6/rjiang77/straighten_DINOWM/temporal-straightening

echo "===== GPU INFO ====="
nvidia-smi
echo "===================="

export TQDM_MININTERVAL=60

python train.py --config-name train.yaml env=pusht_noise
#srun python plan.py --config-name plan_point_maze.yaml model_name=point_maze



# srun python train_DINO_patch_decoder.py
# srun python learn_unicycle_dino_wm_decoder_direct.py
# srun python eval_wm.py
# srun python RSE.py
# srun python DINO_vit.py

# srun python Franca_CLS_to_CLS_prediction.py

# srun python Franca_CLS_to_CLS_prediction_residual.py

# srun python run_unicycle_generate_pace.py
# srun python concat_all_data.py