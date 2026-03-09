# Default setting for the pair-end dataset
python accusnv_snakemake.py -i test_data_csv/samples_cae_test_pe.csv -r reference_genomes -o cae_pe_test_new

# Use samclip for the pair-end dataset
#python accusnv_snakemake.py -p 1 -i test_data_csv/samples_cae_test_pe.csv -r reference_genomes -o cae_pe_test_samclip

# Use bowtie2 as aligner for the pair-end dataset
#python accusnv_snakemake.py -a bowtie2 -i test_data_csv/samples_cae_test_pe.csv -r reference_genomes -o cae_pe_test_bt2

# Default setting for the single-end dataset
#python accusnv_snakemake.py -i samples_cae_test_se.csv  -r reference_genomes -o cae_se_test

# Local run without Slurm system (e.g. Laptop with MacOS) - test with the pair-end dataset
#python accusnv_snakemake.py -f 1 -i test_data_csv/samples_cae_test_pe.csv -r reference_genomes -o cae_pe_test_local
