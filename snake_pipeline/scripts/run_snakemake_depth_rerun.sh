#!/bin/bash
#SBATCH --job-name widevariant.main.dr
#SBATCH -n 1
#SBATCH -p sched_mit_tami,mit_normal,newnodes,sched_mit_chisholm,sched_mit_hill
#SBATCH --time=1-00:00:00
#SBATCH --mem=100GB
#SBATCH -o mainout_depth_rerun.txt
#SBATCH -e mainerr_depth_rerun.txt
#SBATCH --mail-user=YOUR_EMAIL_HERE
#SBATCH --mail-type=ALL

snakemake --profile new_fix_rerun_depth/10x/res/sim_reads_ecoli_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/10x/res/sim_reads_cae_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/10x/res/sim_reads_kcp_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/10x/res/sim_reads_cdi_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/10x/res/sim_reads_spn_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/10x/res/sim_reads_sau_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/20x/res/sim_reads_ecoli_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/20x/res/sim_reads_cae_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/20x/res/sim_reads_kcp_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/20x/res/sim_reads_cdi_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/20x/res/sim_reads_spn_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/20x/res/sim_reads_sau_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/30x/res/sim_reads_ecoli_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/30x/res/sim_reads_cae_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/30x/res/sim_reads_kcp_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/30x/res/sim_reads_cdi_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/30x/res/sim_reads_spn_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/30x/res/sim_reads_sau_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/40x/res/sim_reads_ecoli_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/40x/res/sim_reads_cae_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/40x/res/sim_reads_kcp_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/40x/res/sim_reads_cdi_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/40x/res/sim_reads_spn_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/40x/res/sim_reads_sau_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/50x/res/sim_reads_ecoli_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/50x/res/sim_reads_cae_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/50x/res/sim_reads_kcp_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/50x/res/sim_reads_cdi_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/50x/res/sim_reads_spn_mix_ls_l150/conf
snakemake --profile new_fix_rerun_depth/50x/res/sim_reads_sau_mix_ls_l150/conf
