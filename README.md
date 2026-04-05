# Codebase Experiments

This repository contains research scripts and experiments, organized by experiment family and pipeline stage.

## Directory Structure

The codebase is organized under the `experiments/` directory into the following families:

- `experiments/unicycle/`: Scripts for Unicycle dataset generation, model training, and evaluation.
- `experiments/dino_wm/`: Scripts for DINO-based Visual World Models (DINO-WM).
- `experiments/franca/`: Scripts for Franca models and experiments.
- `experiments/shared/`: Shared utilities and models used across multiple experiment families.

Inside each experiment family, the code is further divided by stage:
- `data/`: Dataset generation, dataloaders, and preprocessing scripts.
- `models/`: Neural network architectures and model definitions.
- `train/`: Training scripts and entrypoints.
- `eval/`: Evaluation scripts, testing, and plotting.
- `jobs/`: Slurm `.sbatch` files and Python job launchers.
- `utils/`: Miscellaneous helper scripts.

## Running Scripts

Because the codebase has been reorganized into Python packages, you should run scripts as modules from the repository root.

For example, to run a data generation script for Unicycle:
```bash
python -m experiments.unicycle.data.generate_unicycle_dataset
```

To run a training script:
```bash
python -m experiments.unicycle.train.run_script_11
```

To submit a Slurm job:
```bash
sbatch experiments/unicycle/jobs/run_2.sbatch
```

### Important Note on Imports
All local imports have been updated to use absolute package paths (e.g., `from experiments.dino_wm.models.DINO_vit import DINO_vit`). Ensure that you always execute commands from the root directory of the repository so that the `experiments` package is correctly resolved by Python.
