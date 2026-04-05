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

#srun python -m experiments.unicycle.train.run_script_11


export WANDB_DIR=/storage/scratch1/6/rjiang77/wandb
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
cd /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle

echo "===== GPU INFO ====="
nvidia-smi
echo "===================="

export TQDM_MININTERVAL=60

python -m experiments.dino_wm.eval.plot_linear_error
#python -m experiments.dino_wm.eval.check_distance
#srun python plan.py --config-name plan_point_maze.yaml model_name=point_maze



# srun python -m experiments.unicycle.train.train_DINO_patch_decoder
# srun python -m experiments.unicycle.train.learn_unicycle_dino_wm_decoder_direct
# srun python -m experiments.unicycle.eval.eval_wm
# srun python -m experiments.shared.models.RSE
# srun python -m experiments.dino_wm.models.DINO_vit

# srun python -m experiments.franca.models.Franca_CLS_to_CLS_prediction

# srun python -m experiments.franca.models.Franca_CLS_to_CLS_prediction_residual

# srun python run_unicycle_generate_pace.py
# srun python -m experiments.unicycle.data.concat_all_data