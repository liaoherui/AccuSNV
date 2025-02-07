#!/bin/bash
#SBATCH --job-name widevariant.main
#SBATCH -n 1
#SBATCH --time=10:00:00
#SBATCH --mem=100GB
#SBATCH -o mainout.txt
#SBATCH -e mainerr.txt
#SBATCH --mail-user=YOUR_EMAIL_HERE
#SBATCH --mail-type=ALL

# Activate conda environment (may need to change name of env)
#source activate snakemake

snakemake  --profile .

# Print "Done!!!" at end of main log file
echo Done!!!
