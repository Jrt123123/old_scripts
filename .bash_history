squeue -u rjiang77
scancel 2143082
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
scancel 2143142
sbatch run_run_run.sbatch
scancel 2143152
sbatch run_run_run.sbatch
scancel 2143160
sbatch run_run_run.sbatch
scancel 2143204
sbatch run_run_run.sbatch
pace-quota
scancel 2143380
sbatch run_run_run.sbatch
pace-quota
sacctmgr show associations user=$USER
sacct -j 2143386 --format=JobID,Account,User,Partition,Elapsed,AllocTRES%40,ReqTRES%30,State
pace-usage -j 2143386
seff 2143386
pace-quota
sbatch run_run_run.sbatch
conda info --envs
conda activate torch_env
module load anaconda3/2023.03
conda info --envs
conda activate torch_env
conda activate  /storage/scratch1/6/rjiang77/conda_envs/torch_env
conda deactivate
sbatch run_run_run.sbatch
module load conda
module load anaconda3/2023.03
conda activate  /storage/scratch1/6/rjiang77/conda_envs/torch_env
conda install matplotlib
sbatch run_run_run.sbatch
conda install scikit-image
sbatch run_run_run.sbatch
scancel 2163994
sbatch run_run_run.sbatch
pace-quota
du -h --max-depth=1 /storage/coda1/p-gchou3/0 | sort -h
pace-quota
du -h --max-depth=1 /storage/coda1/p-gchou3/0 | sort -h
conda install casadi
pip install casadi
sbatch run_run_run.sbatch
sbatch run_run+run.sbatch
sbatch run_run_run.sbatch
pace-quota
sbatch run_run_run.sbatch
pace-quota
sbatch run_run_run.sbatch
scancel 2172433
sbatch run_run_run.sbatch
squeue
squeue -u rjiang77
sbatch run_2.sbatch
squeue -u rjiang77
sacct -j 2200178 -o JobID,State,Elapsed,MaxRSS,ReqMem,NodeList,ExitCode
sbatch run_2.sbatch
pace-quota
load module anaconda
module load anaconda
module load anaconda3/2023.03
conda activate  /storage/scratch1/6/rjiang77/conda_envs/torch_env
pip install itertools
sbatch run_2.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
scancel 2201324
squeue -u rjiang77
sbatch run_2.sbatch
squeue -u rjiang77
sbatch run_run_run_2.sbatch
squeue -u rjiang77
sbatch run_2.sbatch
squeue -u rjiang77
pace-quota
sbatch run_2.sbatch
sacct -j 2143386
pace-quota
squeue -u rjiang77
pace quota
pace-quota
sbatch run_run_run.sbatch
squeue -u rjiang77
pace-quota
sbatch run_run_run.sbatch
scancel 2262018
sbatch run_run_run.sbatch
sbatch run_run_run_2.sbatch
pace-quota
scontrol show job 2997817
sbatch run_run_run.sbatch
scontrol show job 3005980
sbatch run_run_run.sbatch
sbatch run_generate_bounded.sbatch
sbatch run_generate_straight.sbatch
sbatch run_generate_manual.sbatch
sbatch run_run_run.sbatch
scontrol show job 3034702
sbatch run_run_run_3.sbatch
sbatch run_run_run.sbatch
scontrol show job 3347402
squeue -u rjiang77
sinfo
squeue
squeue -u rjiang77
sbatch run_run_run.sbatch
conda activate /storage/scratch1/6/rjiang77/conda_envs/torch_env
which python
source ~/miniconda3/etc/profile.d/conda.sh
ls ~/anaconda*/bin/conda
ls /usr/local/*conda*/bin/conda
[rjiang77@login-phoenix-gnr-2 ~]$ source ~/miniconda3/etc/profile.d/conda.sh
-bash: /storage/home/hcoda1/6/rjiang77/miniconda3/etc/profile.d/conda.sh: No such file or directory
[rjiang77@login-phoenix-gnr-2 ~]$ ls ~/anaconda*/bin/conda
ls: cannot access '/storage/home/hcoda1/6/rjiang77/anaconda*/bin/conda': No such file or directory
[rjiang77@login-phoenix-gnr-2 ~]$ ls /usr/local/*conda*/bin/conda
ls: cannot access '/usr/local/*conda*/bin/conda': No such file or directory
[rjiang77@login-phoenix-gnr-2 ~]$module load anaconda3/2023.03
module load anaconda3/2023.03
conda activate  /storage/scratch1/6/rjiang77/conda_envs/torch_env
which python
conda install -y python=3.13 --force-reinstall
ls -ld /storage/scratch1/6/rjiang77/conda_envs/torch_env/conda-meta
ls -l /storage/scratch1/6/rjiang77/conda_envs/torch_env/pyvenv.cfg
ls -l /storage/scratch1/6/rjiang77/conda_envs/torch_env/conda-meta/history
ls -lh /storage/scratch1/6/rjiang77/conda_envs/torch_env/conda-meta | head
touch /storage/scratch1/6/rjiang77/conda_envs/torch_env/conda-meta/history
conda install -y -p /storage/scratch1/6/rjiang77/conda_envs/torch_env python=3.10 --force-reinstall
sbatch run_run_run.sbatch
conda deactivate
sbatch run_run_run.sbatch
quota -s
sbatch run_run_run.sbatch
nvidia-smi
sbatch run_run_run.sbatch
module load anaconda3/2023.03
conda activate /storage/scratch1/6/rjiang77/conda_envs/torch_env
conda install -y typing_extensions
conda install -y pytorch
df -h /storage/home/hcoda1/6/rjiang77
df -h /storage/scratch1/6/rjiang77
quota -u rjiang77
conda install -y pytorch
conda install -y torchvision
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8
torchvision torchaudio pytorch-cuda=11.8
quota -s
which python
conda clean --all
y
quota -s
module load anaconda3/2023.03
conda activate  /storage/scratch1/6/rjiang77/conda_envs/torch_env
quota -s
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8
rm -f /storage/home/hcoda1/6/rjiang77/.conda/pkgs/pytorch-2.5.1-py3.10_cuda11.8_cudnn9.1.0_0.tar.bz2
rm -f /storage/home/hcoda1/6/rjiang77/.conda/pkgs/pytorch-2.5.1-py3.10_cuda11.8_cudnn9.1.0_0.tar.bz2.*
conda clean -i -y
export CONDA_PKGS_DIRS=/storage/scratch1/6/rjiang77/.conda_pkgs
mkdir -p $CONDA_PKGS_DIRS
python - <<'EOF'

import os

print("CONDA_PKGS_DIRS =", os.environ.get("CONDA_PKGS_DIRS"))

EOF

echo 'export CONDA_PKGS_DIRS=/storage/scratch1/6/rjiang77/.conda_pkgs' >> ~/.bashrc
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8
python -c "import torch; print(torch.__version__, torch.version.cuda)"
pip install --force-reinstall torch torchvision torchaudio
module load anaconda3/2023.03
conda --version
conda create -y \
  -p /storage/scratch1/6/rjiang77/conda_envs/torch_env \
  python=3.10
conda create -y  -p /storage/scratch1/6/rjiang77/conda_envs/torch_env python=3.10
conda activate /storage/scratch1/6/rjiang77/conda_envs/torch_env
conda install -y pytorch
conda install -y torchvision
sbatch run_run_run.sbatch
module load anaconda3/2023.03
conda activate  /storage/scratch1/6/rjiang77/conda_envs/torch_env
pip install h5py
pip install tqdm
scancel 3672044
squeue
sbatch run_run_run.sbatch
module load anaconda3/2023.03
conda activate  /storage/scratch1/6/rjiang77/conda_envs/torch_env
pip install matplotlib
pip install scikit-image
cd /storage/scratch1/6/rjiang77
git clone https://github.com/valeoai/Franca.git
cd Franca
pip install -e ".[franca]"
cd..
cd ..
pip install casadi
sacct -j 3672069 --format=JobID,State,ExitCode,Elapsed,Timelimit
scontrol show job 3672069 | sed -n '1,200p'
sbatch run_generate_bounded.sbatch
quota -s
squeue -u
squeue
squeue -u rjiang77
squeue -u
quota -s
du -sh /storage/coda1/p-gchou3/0/rjiang77/torch_cache
sbatch run_generate_bounded.sbatch
ls -l slurm-3682943_119.out
sbatch run_generate_bounded.sbatch
scancel 3683135
sbatch run_generate_bounded.sbatch
sbatch run_generate_manual.sbatch
sbatch run_generate_straight.sbatch
sbatch run_generate_manual.sbatch
sbatch run_generate_straight.sbatch
sbatch run_generate_bounded.sbatch
sbatch run_generate_manual.sbatch
sbatch run_generate_straight.sbatch
sbatch run_run_run.sbatch
scancel 3953256
scancel 3953257
sbatch run_run_run.sbatch
scancel 3953261
sbatch run_run_run.sbatch
sbatch run_generate_bounded.sbatch
sbatch run_generate_manual.sbatch
sbatch run_run_run.sbatch
squeue -u rjiang77
squota
quota -s
sbatch run_run_run.sbatch
sbatch run_generate_bounded.sbatch
scancel 3973393
sbatch run_generate_bounded.sbatch
scancel 3973465
sbatch run_generate_bounded.sbatch
quota -s rjiang77
sbatch run_generate_bounded.sbatch
sbatch run_generate_manual.sbatch
sbatch run_run_run.sbatch
squeue -u
squeue 
squeue -u rjiang77
sbatch run_generate_bounded.sbatch
sbatch run_generate_manual.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
sbatch run_run_run_2.sbatch
dos2unix run_run_run_2.sbatch
sbatch run_run_run_2.sbatch
dos2unix run_run_run_3.sbatch
sbatch run_run_run_3.sbatch
sbatch run_generate_bounded.sbatch
squeue -u rjiang77
sbatch run_generate_bounded.sbatch
squeue -u rjiang77
sbatch run_generate_bounded.sbatch
module load anaconda3/2023.03
rm -rf /storage/scratch1/6/rjiang77/conda_envs/torch_env
conda env create -p /storage/scratch1/6/rjiang77/conda_envs/torch_env -f torch_env.y
conda install tqdm
sbatch run_generate_bounded.sbatch
conda deactivate
sbatch run_generate_bounded.sbatch
ls -l /storage/scratch1/6/rjiang77/conda_envs/torch_env/lib/python3.10/encodings/__init__.py
sbatch run_generate_bounded.sbatch
ls -ld /storage/scratch1/6/rjiang77/conda_envs/torch_env/lib/python3.10
conda create --clone /storage/scratch1/6/rjiang77/conda_envs/torch_env              -p /storage/scratch1/6/rjiang77/conda_envs/torch_env_clone
module load anaconda3/2023.03
conda create --clone /storage/scratch1/6/rjiang77/conda_envs/torch_env              -p /storage/scratch1/6/rjiang77/conda_envs/torch_env_clone
conda activate /storage/scratch1/6/rjiang77/conda_envs/torch_env
conda env export --no-builds > torch_env.yml
pwd
conda clean -a -y
conda deactivate
rm -rf /storage/scratch1/6/rjiang77/conda_envs/torch_env
conda env create -p /storage/scratch1/6/rjiang77/conda_envs/torch_env -f torch_env.yml
module load anaconda3/2023.03
rm -rf /storage/scratch1/6/rjiang77/.conda_pkgs
mkdir -p /storage/scratch1/6/rjiang77/.conda_pkgs
conda clean -a -y
conda env create -p /storage/scratch1/6/rjiang77/conda_envs/torch_env -f torch_env.yml
rm -rf /storage/scratch1/6/rjiang77/conda_envs/torch_env
conda env create -p /storage/scratch1/6/rjiang77/conda_envs/torch_env -f torch_env.yml
grep -n "1.3.3" /storage/home/hcoda1/6/rjiang77/condaenv.geian9ko.requirements.txt
conda activate /storage/scratch1/6/rjiang77/conda_envs/torch_env
python -c "import torch; print(torch.__version__)"
python -c "import numpy; print(numpy.__version__)"
python -c "import h5py; print('h5py ok')"
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8
conda install -y -c conda-forge h5py
conda install tqdm
conda install matplotlib
conda install casadi -y
pip install "casadi==3.7.2"
python -c "import torch, sys; print(torch.__version__); print(torch.__file__)"
conda list | egrep 'torch|pytorch|mkl|intel-openmp|iomp'
conda install -y -c conda-forge "blas=*=openblas" openblas
conda remove -y mkl mkl-service mkl_fft mkl_random intel-openmp
python -c "import torch, sys; print(torch.__version__); print(torch.__file__)"
pip install scikit-image
conda deactivate
conda activate /storage/scratch1/6/rjiang77/conda_envs/torch_env
python - <<'PY'
import torch, torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("torchvision file", torchvision.__file__)
try:
    import torchvision.ops as ops
    print("ops imported OK")
    print("has nms:", hasattr(ops, "nms"))
except Exception as e:
    print("ops import failed:", repr(e))
PY

conda remove -y torchvision
conda install -y -c pytorch torchvision=0.20.1
python - <<'PY'
import torch, torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("torchvision file", torchvision.__file__)
try:
    import torchvision.ops as ops
    print("ops imported OK")
    print("has nms:", hasattr(ops, "nms"))
except Exception as e:
    print("ops import failed:", repr(e))
PY

pip show torch torchvision torchaudio 2>/dev/null | sed -n '1,120p'
pip uninstall -y torch torchvision torchaudio
pip uninstall -y triton torchtriton
conda remove -y pytorch torchvision torchaudio pytorch-cuda pytorch-mutex
conda install -y -c pytorch -c nvidia   pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=11.8
python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, torch.__file__)
print("torchvision", torchvision.__version__, torchvision.__file__)
from torchvision.ops import nms
print("nms ok", nms)
PY

pip show torch torchvision torchaudio | sed -n '1,80p'
conda list | egrep '^(pytorch|torchvision|torchaudio|pytorch-cuda|intel-openmp|mkl)'
pip uninstall -y torch torchvision torchaudio triton torchtriton nvidia-*
pip uninstall -y torch torchvision torchaudio triton torchtriton
conda list | egrep '^(pytorch|torchvision|torchaudio|pytorch-cuda|intel-openmp|mkl)'
pip show torch torchvision torchaudio | sed -n '1,80p'
python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, torch.__file__)
print("torchvision", torchvision.__version__, torchvision.__file__)
from torchvision.ops import nms
print("nms ok", nms)
PY

conda remove -y pytorch torchvision torchaudio pytorch-cuda pytorch-mutex   mkl mkl-service mkl_fft mkl_random intel-openmp
conda install -y -c conda-forge "blas=*=openblas" openblas
conda install -y -c pytorch -c nvidia pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=11.8
python -c "import torch; print('torch ok', torch.__version__, 'cuda', torch.cuda.is_available())"
conda install mkl==2024.0 -c conda-forge -y
conda deactivate
sbatch run_generate_bounded.sbatch
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_generate_manual.sbatch
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run_2.sbatch
squeue -u rjiang77
sbatch run_generate_manual.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run_2.sbatch
squeue -u rjiang77
sbatch run_generate_manual.sbatch
squeue -u rjiang77
sbatch run_run_run_2.sbatch
sbatch run_run_run.sbatch
squeue -u rjiang77
sacct -j 4061251 --format=JobID,State,Elapsed,Timelimit
sshare -A gts-gchou3
sacctmgr show assoc user=rjiang77 format=Account,QOS,GrpTRES,GrpTRESMins,MaxWall
sacct -j 4061251 --format=JobID,State,Elapsed,Timelimit,MaxRSS
sbatch run_run_run.sbatch
sbatch run_run_run_2.sbatch
sbatch run_run_run_3.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_generate_bounded.sbatch
sbatch run_generate_manual.sbatch
quota -u rjiang77
sbatch run_generate_patch_bounded.sbatch
dos2unix dos2unix run_generate_patch_bounded.sbatch
sbatch run_generate_patch_bounded.sbatch
squeue -u rjiang77
scontrol show job 410551
dos2unix dos2unix run_generate_patch_bounded.sbatch
scontrol show job 4105519 | egrep -i 'array|jobstate|reason'
scancel 4105519
rm -rf ~/.cache/torch/hub/main.zip ~/.cache/torch/hub/dinov2 ~/.cache/torch/hub/facebookresearch_dinov2_main
dos2unix run_generate_patch_bounded.sbatch
sbatch run_generate_patch_bounded.sbatch
squeue -u rjiang77
scancel 4105583
sbatch run_generate_patch_bounded.sbatch
squeue -u rjiang77
scancel 4105595
sbatch run_generate_patch_bounded.sbatch
scancel 4105603
sbatch run_generate_patch_bounded.sbatch
scancel 4105611
sbatch run_generate_patch_bounded.sbatch
squeue -u rjiang77
scancel 4105612
sbatch run_generate_patch_bounded.sbatch
scancel 4105638
sbatch run_generate_patch_bounded.sbatch
scancel 4105657
sbatch run_generate_patch_bounded.sbatch
quota -u rjiang77
sbatch run_generate_patch_manual.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
module load anaconda3/2023.03
conda activate  /storage/scratch1/6/rjiang77/conda_envs/torch_env
pip install einops
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run_2.sbatch
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
which unzip
cd /storage/scratch1/6/rjiang77/DINO_WM_Wall
unzip wall_single.zip
squeue -u rjiang77
scancel 4428776
sbatch run_run_run.sbatch
scancel 4434494
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
sbatch run_run_run.sbatch
git clone https://github.com/gaoyuezhou/dino_wm.git /storage/scratch1/6/rjiang77/DINO_WM_git
cd /storage/scratch1/6/rjiang77/DINO_WM_git
cd dino_wm
conda env create -f environment.yaml
module load anaconda3/2023.03
conda env create -f environment.yaml
rm -rf ~/.cache/pip
du -sh ~/.cache
ls -a ~
nano ~/.bashrc
source ~/.bashrc
echo $PIP_CACHE_DIR
nano ~/.bashrc
source ~/.bashrc
echo $PIP_CACHE_DIR
conda env create -f environment.yaml
conda env remove -n dino_wm
conda env create -f environment.yaml
mkdir -p ~/.mujoco
pwd
mkdir -p ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/.mujoco/
cd ~/.mujoco
tar -xzvf mujoco210-linux-x86_64.tar.gz
nano ~/.bashrc
source ~/.bashrc
ls /usr/lib/nvidia
ls -d /usr/lib/nvidia*
echo $LD_LIBRARY_PATH
nano ~/.bashrc
source ~/.bashrc
ldd $HOME/.mujoco/mujoco210/bin/simulate | head
echo $LD_LIBRARY_PATH
conda deactivate
cd /storage/scratch1/6/rjiang77/DINO_WM_git
module load anaconda3/2023.03
nano ~/.bashrc
source ~/.bashrc
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
mkdir -p /storage/scratch1/6/rjiang77/wandb
sbatch run_run_run.sbatch
squeue -u rjiang77
sshare -u rjiang77 | grep rjiang77
sshare -u rjiang77 -o Account,User,RawUsage,NormUsage,FairShare
source ~/.bashrc
nano ~/.bashrc
source ~/.bashrc
sbatch run_run_run.sbatch
squeue -u rjiang77
unzip /storage/scratch1/6/rjiang77/DINO_WM_git/datasets/point_maze.zip /storage/scratch1/6/rjiang77/DINO_WM_git/datasets/point_maze_dataset
unzip /storage/scratch1/6/rjiang77/DINO_WM_git/datasets/point_maze.zip -d /storage/scratch1/6/rjiang77/DINO_WM_git/datasets/point_maze_dataset
squeue -u rjiang77
nano ~/.bashrc
sbatch run_run_run.sbatch
squeue -u rjiang77
sacctmgr show assoc user=rjiang77
sshare -u rjiang77
module load anaconda3/2023.03
conda activate dino_wm
export WANDB_DIR=/storage/scratch1/6/rjiang77/wandb
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
cd /storage/scratch1/6/rjiang77/DINO_WM_git
export HYDRA_FULL_ERROR=1
srun python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3
python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3
conda deactivate
exit
module load anaconda3/2023.03
conda activate dino_wm
export WANDB_DIR=/storage/scratch1/6/rjiang77/wandb
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
cd /storage/scratch1/6/rjiang77/DINO_WM_git
python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3
exit
module load anaconda3/2023.03
conda activate dino_wm
conda install python=3.10
conda deactivate
conda env remove -n dino_wm
cd /storage/scratch1/6/rjiang77/DINO_WM_git
conda env create -f environment.yaml
srun --pty --account=gts-gchou3 --partition=embers --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=2:00:00 bash
sinfo -a
srun --pty --account=gts-gchou3 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=2:00:00 bash
srun --pty --partition=gpu-v100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=2:00:00 bash
srun --pty --account=gts-gchou3 --partition=gpu-v100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=30:00 bash
squeue -u rjiang77
module load anaconda3/2023.03
conda activate dino_wm
python -V
conda activate dino_wm
sbatch run_run_run.sbatch
squeue -u rjiang77
grep -R "submitit" -n .
sinfo -p gpu-h100,gpu-v100 -o "%P %a %D %t %G %l"
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
pace-quota
squeue -u rjiang77
pace-quota
squeue -u rjiang77
scancel 4629277
ls /storage/scratch1/6/rjiang77/DINOv2_patch_224/2_21_dataset | head
sbatch run_run_run_2.sbatch
dos2unix run_run_run_2.sbatch
sbatch run_run_run_2.sbatch
squeue -u rjiang77
scancel 4629272
srun --pty --account=gts-gchou3 --partition=gpu-v100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=2:00:00 bash
sbatch run_run_run_2.sbatch
module load anaconda3/2023.03
conda activate dino_wm
export WANDB_DIR=/storage/scratch1/6/rjiang77/wandb
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
cd /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle
export TQDM_MININTERVAL=60
srun python train.py
python train.py
exit
sbatch run_run_run.sbatch
squeue -u rjiang77
scancel 4637094
sbatch run_run_run.sbatch
sbatch run_run_run_2.sbatch
squeue -u rjiang77
sbatch run_run_run_2.sbatch
pace-quota
sbatch run_run_run_2.sbatch
srun --pty --account=gts-gchou3 --partition=gpu-v100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=2:00:00 bash
sbatch run_run_run_2.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
squeue -u rjiang77
scancel 4641608
smi-nvidia
sbatch run_run_run.sbatch
sbatch run_run_run_2.sbatch
squeue -u rjiang77
pace-quota
sbatch run_run_run_2.sbatch
sbatch run_run_run.sbatch
exit
squeue -u rjiang77
srun --pty --account=gts-gchou3 --partition=gpu-v100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=30:00 bash
srun --pty --account=gts-gchou3 --partition=gpu-v100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=2:00:00 bash
cd /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle
module load anaconda3/2023.03
conda activate dino_wm
python test_CEM.py
exit
cd /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle
smi-nvidia
nvidia-smi
module load anaconda3/2023.03
conda activate dino_wm
export WANDB_DIR=/storage/scratch1/6/rjiang77/wandb
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
python test_CEM.py
exit
squeue -u rjiang77
cp /storage/scratch1/6/rjiang77/DINO_WM_git/checkpoints/outputs/2026-03-04/21-17-03/checkpoints/model_last.pth /storage/scratch1/6/rjiang77/DINO_WM_git/checkpoints/outputs/point_maze 
cp /storage/scratch1/6/rjiang77/DINO_WM_git/checkpoints/outputs/2026-03-04/21-17-03/checkpoints/model_latest.pth /storage/scratch1/6/rjiang77/DINO_WM_git/checkpoints/outputs/point_maze 
squeue -u rjiang77
srun --pty --account=gts-gchou3 --partition=gpu-v100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=2:00:00 bash
sbatch run_generate_patch_bounded.sbatch
sbatch run_generate_patch_manual.sbatch
sbatch run_generate_patch_straight.sbatch
sbatch run_run_run.sbatch
squeue -u rjiang77
sbatch run_generate_patch_manual.sbatch
sbatch run_generate_patch_straight.sbatch
sbatch run_generate_patch_manual.sbatch
squeue -u rjiang77
sbatch --begin=now+4hours run_run_run_2.sbatch
module load anaconda3/2023.03
conda activate dino_wm
cd /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle
python train_768.py
exit
sbatch run_run_run_2.sbatch
squeue -u rjiang77
srun --pty --account=gts-gchou3 --partition=gpu-v100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=1:00:00 bash
sbatch run_run_run_3.sbatch
squeue -u rjiang77
sbatch run_run_run.sbatch
module load anaconda3/2023.03
conda activate dino_wm
export WANDB_DIR=/storage/scratch1/6/rjiang77/wandb
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
cd /storage/home/hcoda1/6/rjiang77/Code_for_invariant_bisimilarity
python train_768.py
python train_384.py
exit
squeue -u rjiang77
sbatch run_run_run_4.sbatch
sbatch run_run_run_5.sbatch
squeue -u rjiang77
sbatch run_run_run_3.sbatch
sbatch run_run_run_2.sbatch
sbatch run_run_run.sbatch
cp -r /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle/__pycache__ /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle/cem_out /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle/ControlEncoderMLP.py /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle/ControlEncoderSingle.py /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle/dino_wm_dataloader.py /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle/Predictor.py /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle/test_CEM.py /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle/train_384.py /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle/train_768.py /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle/traj_dset.py /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle/unicycle_dataloader.py /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle/visual_world_model.py /storage/home/hcoda1/6/rjiang77/Code_for_invariant_bisimilarity/
srun --pty --account=gts-gchou3 --partition=gpu-v100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=1:00:00 bash
squeue -u rjiang77
sbatch run_run_run.sbatch
sbatch run_run_run_2.sbatch
sbatch run_run_run_3.sbatch
sbatch run_run_run_4.sbatch
sbatch run_run_run_5.sbatch
sbatch run_run_run_4.sbatch
sbatch run_run_run_5.sbatch
sbatch run_run_run_4.sbatch
sbatch run_run_run_5.sbatch
sbatch run_run_run_4.sbatch
cd /storage/scratch1/6/rjiang77/DINO_WM_git
python plan.py model_name=point_maze n_evals=5 planner=cem goal_H=5 goal_source='random_state' planner.opt_steps=30
module load anaconda3/2023.03
conda activate dino_wm
python plan.py model_name=point_maze n_evals=5 planner=cem goal_H=5 goal_source='random_state' planner.opt_steps=30
export WANDB_DIR=/storage/scratch1/6/rjiang77/wandb
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
python plan.py model_name=point_maze n_evals=5 planner=cem goal_H=5 goal_source='random_state' planner.opt_steps=30
nvidia-smi
exit
nvidia-smi
module load anaconda3/2023.03
conda activate dino_wm
export WANDB_DIR=/storage/scratch1/6/rjiang77/wandb
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
python plan.py model_name=point_maze n_evals=5 planner=cem goal_H=5 goal_source='random_state' planner.opt_steps=30
cd /storage/scratch1/6/rjiang77/DINO_WM_git
python plan.py model_name=point_maze n_evals=5 planner=cem goal_H=5 goal_source='random_state' planner.opt_steps=30
cd /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle
python test_CEM.py
python check_dataset.py
python test_CEM.py
python check_dataset.py
python test_CEM.py
exit
squeue -u rjiang77
1/6/rjiang77/DINO_WM_git/checkpoints/outputs/2026-03-04/21-17-03/checkpoints/model_latest.pth /storage/scratch1/6/rjiang77/DINO_WM_git/checkpoints/outputs/point_maze
squeue -u rjiang77
sbatch run_run_run.sbatch
sbatch run_run_run_2.sbatch
sbatch run_run_run_3.sbatch
sbatch run_run_run_4.sbatch
sbatch run_run_run_5.sbatch
squeue -u rjiang77
srun --pty --account=gts-gchou3 --partition=gpu-v100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=2:00:00 bash
srun --pty --account=gts-gchou3 --partition=gpu-a100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=2:00:00 bash
module load anaconda3/2023.03
conda activate dino_wm
export WANDB_DIR=/storage/scratch1/6/rjiang77/wandb
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
cd /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle
python test_CEM.py
python test_GD.py
cd /storage/home/hcoda1/6/rjiang77/Code_for_invariant_bisimilarity
python test_CEM.py
python check_bisim_latent.py
python check_latent_bisim.py
exit
module load anaconda3/2023.03
conda activate dino_wm
export WANDB_DIR=/storage/scratch1/6/rjiang77/wandb
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
cd /storage/home/hcoda1/6/rjiang77/Code_for_invariant_bisimilarity
python check_latent_bisim.py
cd /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle
python test_GD.py
exit
squeue -u rjiang77
srun --pty --account=gts-gchou3 --partition=gpu-a100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=3:00:00 bash
srun --pty --account=gts-gchou3 --partition=gpu-a100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=1:00:00 bash
squeue -u rjiang77
for i in {6..9}; do cp run_run_run_4.sbatch run_run_run_${i}.sbatch; done
sbatch run_run_run_6.sbatch
srun --pty --account=gts-gchou3 --partition=gpu-a100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=1:00:00 bash
sbatch run_run_run_6.sbatch
sbatch run_run_run_7.sbatch
sbatch run_run_run_8.sbatch
sbatch run_run_run_9.sbatch
squeue -u rjiang77
sbatch run_run_run_6.sbatch
sbatch run_run_run_7.sbatch
sbatch run_run_run_8.sbatch
sbatch run_run_run_9.sbatch
squeue -u rjiang77
sbatch run_run_run_6.sbatch
sbatch run_run_run_7.sbatch
sbatch run_run_run_8.sbatch
sbatch run_run_run_9.sbatch
srun --pty --account=gts-gchou3 --partition=gpu-a100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=1:00:00 bash
sbatch run_run_run_6.sbatch
module load anaconda3/2023.03
conda activate dino_wm
cd /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle
python check_linear_error.py
exit
srun --pty --account=gts-gchou3 --partition=gpu-a100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=2:00:00 bash
sbatch run_run_run_2.sbatch
sbatch run_run_run_3.sbatch
sbatch run_run_run_4.sbatch
unix2dos run_run_run_4.sbatch
sbatch run_run_run_4.sbatch
dos2unix run_run_run_4.batch
dos2unix run_run_run_4.sbatch
sbatch run_run_run_4.sbatch
sbatch run_run_run_5.sbatch
dos2unix run_run_run_5.sbatch
sbatch run_run_run_5.sbatch
squeue -u rjiang77
module load anaconda3/2023.03
conda activate dino_wm
cd /storage/home/hcoda1/6/rjiang77/Code_for_invariant_bisimilarity
python test_CEM.py
exit
srun --pty --account=gts-gchou3 --partition=gpu-a100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=1:00:00 bash
sbatch run_run_run_2.sbatch
sbatch run_run_run_3.sbatch
sbatch run_run_run_4.sbatch
sbatch run_run_run_5.sbatch
module load anaconda3/2023.03
conda activate dino_wm
cd /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle
python plot_linear_error.py
python test_CEM.py
exit
sbatch run_run_run_7.sbatch
sbatch run_run_run_8.sbatch
sbatch run_run_run_9.sbatch
squeue -u rjiang77
srun --pty --account=gts-gchou3 --partition=gpu-a100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=2:00:00 bash
srun --pty --account=gts-gchou3 --partition=gpu-a100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=1:00:00 bash
squeue -u rjiang77
sbatch run_run_run_7.sbatch
sbatch run_run_run_8.sbatch
sbatch run_run_run_9.sbatch
sbatch run_generate_patch_bounded.sbatch
sbatch run_generate_patch_manual.sbatch
sbatch run_generate_patch_straight.sbatch
squeue -u rjiang77
module load anaconda3/2023.03
conda activate dino_wm
which python
which pip
pwd
pace-quota
du -sh ~/.conda/envs/dino_wm
pip install casadi
deactivate conda
conda deactivate
sbatch run_generate_patch_manual.sbatch
sbatch run_generate_patch_bounded.sbatch
squeue -u rjiang77
scancel 5030688
squeue -u rjiang77
sbatch run_generate_patch_bounded.sbatch
squeue -u rjiang77
sbatch run_generate_patch_manual.sbatch
sbatch run_generate_patch_straight.sbatch
squeue -u rjiang77
sbatch run_run_run_2.sbatch
squeue -u rjiang77
sbatch run_run_run_2.sbatch
squeue -u rjiang77
sbatch run_run_run_2.sbatch
squeue -u rjiang77
sbatch run_run_run_2.sbatch
module load anaconda3/2023.03
conda activate dino_wm
cd /storage/home/hcoda1/6/rjiang77/Codes_for_dinowm_on_Unicycle
python test_CEM.py
exit
srun --pty --account=gts-gchou3 --partition=gpu-a100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=1:00:00 bash
srun --pty --account=gts-gchou3 --partition=gpu-a100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=3:00:00 bash
srun --pty --account=gts-gchou3 --partition=gpu-v100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=3:00:00 bash
sbatch run_run_run_2.sbatch
scancel 5407055
sbatch run_run_run_2.sbatch
squeue -u rjiang77
scancel 5408865
scancel 5399648
squeue -u rjiang77
sbatch run_run_run_2.sbatch
scancel 5409311
sbatch run_run_run_2.sbatch
pace-quota
sacctmgr show assoc user=rjiang77 format=account,partition,qos
sbatch run_run_run_3.sbatch
squeue -u rjiang77
sbatch run_run_run_3.sbatch
squeue -u rjiang77
sbatch run_run_run_3.sbatch
squeue -u rjiang77
sbatch run_run_run_3.sbatch
squeue -u rjiang77
sbatch run_run_run_3.sbatch
squeue -u rjiang77
sbatch run_run_run_3.sbatch
squeue -u rjiang77
sbatch run_run_run_2.sbatch
squeue -u rjiang77
sbatch run_run_run_3.sbatch
cd /storage/scratch1/6/rjiang77/straighten_DINOWM
cd /storage/scratch1/6/rjiang77
cd straighten_DINOWM
cd /storage/scratch1/6/rjiang77/straighten_DINOWM
git clone git@github.com:agentic-learning-ai-lab/temporal-straightening.git
git clone https://github.com/agentic-learning-ai-lab/temporal-straightening.git
cd temporal-straightening
conda env create   -p /storage/scratch1/6/rjiang77/envs/ts   -f environment.yaml
module load anaconda3/2023.03
conda env create   -p /storage/scratch1/6/rjiang77/envs/ts   -f environment.yaml
module load anaconda3/2023.03
conda env list
cd /storage/scratch1/6/rjiang77
git clone https://github.com/lucas-maes/le-wm.git LeWM_code
cd LeWM_code
ls
export HF_HOME=/storage/scratch1/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/storage/scratch1/$USER/.cache/huggingface
source ~/.bashrc
nano ~/.bashrc
source ~/.bashrc
echo $HF_HOME
echo $TORCH_HOME
huggingface-cli download quentinll/lewm-pusht   --repo-type dataset   --local-dir /storage/scratch1/6/rjiang77/LeWM_data/pusht
conda activate /storage/scratch1/6/rjiang77/conda_envs_scratch/dino_env_scratch
pip install -U huggingface_hub
huggingface-cli download quentinll/lewm-pusht \
  --repo-type dataset \
  --local-dir /storage/scratch1/6/rjiang77/LeWM_data/pusht
huggingface-cli download quentinll/lewm-pusht   --repo-type dataset   --local-dir /storage/scratch1/6/rjiang77/LeWM_data/pusht
qqaqaqa
python -m huggingface_hub.cli download quentinll/lewm-pusht   --repo-type dataset   --local-dir /storage/scratch1/6/rjiang77/LeWM_data/pusht
hf --help
hf download quentinll/lewm-pusht   --repo-type dataset   --local-dir /storage/scratch1/6/rjiang77/LeWM_data/pusht
cd /storage/scratch1/6/rjiang77/LeWM_data/pusht
tar --zstd -xvf archive.tar.zst
cd ..
cd /storage/scratch1/6/rjiang77/LeWM_data/pusht
unzstd pusht_expert_train.h5.zst
sbatch run_run_run_5.sbatch
module load anaconda3/2023.03
conda activate activate /storage/scratch1/6/rjiang77/conda_envs_scratch/dino_env_scratch
pip install hydr
pip install hydra
pip uninstall hydra -y
pip install hydra-core
sbatch run_run_run_5.sbatch
pip install lightening
pip install lightning
cd /storage/scratch1/6/rjiang77/LeWM_code
pip install -e .
pip install stable-pretraining
pip install stable_worldmodel 
module load anaconda3/2023.03
conda activate activate /storage/scratch1/6/rjiang77/conda_envs_scratch/dino_env_scratch
wandb login
wandb status
sbatch run_run_run_5.sbatch
squeue -u rjiang77
sbatch run_run_run_5.sbatch
squeue -u rjiang77
cp -r /storage/scratch1/6/rjiang77/LeWM_code       /storage/scratch1/6/rjiang77/Tests_LeWM_code
squeue -u rjiang77
wandb login
sbatch run_run_run_5.sbatch
sbatch run_run_run_6.sbatch
squeue -u rjiang77
sbatch run_run_run_6.sbatch
nano ~/.bashrc
source ~/.bashrc
nano ~/.bashrc
pace-quota
squeue -u rjiang77
nano ./bashrc
nano ~/.bashrc
sbatch run_run_run_5.sbatch
squeue -u rjiang77
sbatch run_run_run_6.sbatch
sbatch run_run_run_5.sbatch
cd /storage/scratch1/6/rjiang77/DINO_WM_git/datasets/data
wget https://osf.io/k2d8w/download
file download
unzip download
sbatch run_run_run_7.sbatch
cd /storage/home/hcoda1/6/rjiang77
sbatch run_run_run_7.sbatch
module load anaconda3/2023.03
conda activate /storage/scratch1/6/rjiang77/conda_envs_scratch/dino_env_scratch
pip install accelerate
sbatch run_run_run_5.sbatch
sbatch run_run_run_6.sbatch
squeue -u rjiang77
