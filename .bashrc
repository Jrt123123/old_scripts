# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

# User specific aliases and functions
export CONDA_PKGS_DIRS=/storage/scratch1/6/rjiang77/.conda_pkgs
export PIP_CACHE_DIR=/storage/scratch1/6/rjiang77/pip_cache

export TORCH_HOME=/storage/scratch1/6/$USER/.cache/torch
export HF_HOME=/storage/scratch1/6/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/storage/scratch1/6/$USER/.cache/huggingface


# Mujoco
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH
export DATASET_DIR=/storage/scratch1/6/rjiang77/DINO_WM_git/datasets/data
export STABLEWM_HOME=/storage/scratch1/6/rjiang77/LeWM_data/pusht


mkdir -p $CONDA_PKGS_DIRS
mkdir -p $PIP_CACHE_DIR
mkdir -p $TORCH_HOME
mkdir -p $HF_HOME
