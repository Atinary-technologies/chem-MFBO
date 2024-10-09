#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=32G
source venv/bin/activate # Activate your virtual environment
python src/mf_kmc/benchmark/benchmark.py hydra/launcher=submitit_slurm --config-name=synthetic_sweep_cluster.yaml
