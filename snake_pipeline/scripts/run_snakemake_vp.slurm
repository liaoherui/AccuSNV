#!/bin/bash
#SBATCH --job-name widevariant.main.vp
#SBATCH -n 1
#SBATCH -p sched_mit_tami,mit_normal,newnodes,sched_mit_chisholm,sched_mit_hill
#SBATCH --time=1-00:00:00
#SBATCH --mem=100GB
#SBATCH -o mainout_vp.txt
#SBATCH -e mainerr_vp.txt
#SBATCH --mail-user=YOUR_EMAIL_HERE
#SBATCH --mail-type=ALL

# Activate conda environment (may need to change name of env)
#source activate snakemake
snakemake --profile VP_strain100_more_real_msprime_new_fix/res/sim_reads_ecoli_mix_ls_l150/conf
#snakemake --profile VP_strain100_more_real_msprime_new_fix/res/sim_reads_cae_mix_ls_l150/conf
#exit
snakemake --profile VP_strain100_more_real_msprime_new_fix/res/sim_reads_kcp_mix_ls_l150/conf
snakemake --profile VP_strain100_more_real_msprime_new_fix/res/sim_reads_cdi_mix_ls_l150/conf
snakemake --profile VP_strain100_more_real_msprime_new_fix/res/sim_reads_spn_mix_ls_l150/conf
snakemake --profile VP_strain100_more_real_msprime_new_fix/res/sim_reads_sau_mix_ls_l150/conf
#exit
# Print "Done!!!" at end of main log file
echo Done!!!
