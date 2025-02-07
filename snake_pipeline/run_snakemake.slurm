#!/bin/bash
#SBATCH --job-name widevariant.main
#SBATCH -p sched_mit_hill,sched_mit_tami,sched_mit_chisholm
#SBATCH -n 1
#SBATCH --time=12:00:00
#SBATCH --mem=8GB
#SBATCH -o mainout.txt
#SBATCH -e mainerr.txt
#SBATCH --mail-user=YOUR_EMAIL_HERE
#SBATCH --mail-type=ALL
#SBATCH --exclude=node327

# Activate conda environment (may need to change name of env)
source activate snakemake

snakemake --profile .

# Print "Done!!!" at end of main log file
echo Done!!!
