#!/bin/bash
#SBATCH -J UnicycleLatentTrain
#SBATCH --account=gts-gchou3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1        # 1 main + 8 workers + spare
#SBATCH --cpus-per-task=4        # use 4 CPU cores for dataloading/threads

#SBATCH --mem=32G                  # Request total memory directly
#SBATCH -t 8:00:00                 # 24 hours (adjust as needed)
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rjiang77@gatech.edu
# #SBATCH --array=20-39




cd $SLURM_SUBMIT_DIR
module load anaconda3/2023.03



#conda activate  /storage/scratch1/6/rjiang77/conda_envs/torch_env
conda activate dino_wm

#srun python run_script_11.py


export WANDB_DIR=/storage/scratch1/6/rjiang77/wandb
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
cd /storage/scratch1/6/rjiang77/DINO_WM_git

echo "===== GPU INFO ====="
nvidia-smi
echo "===================="

export TQDM_MININTERVAL=60

srun python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3 hydra.run.dir=/storage/scratch1/6/rjiang77/DINO_WM_git/checkpoints/outputs/2026-03-04/21-17-03
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