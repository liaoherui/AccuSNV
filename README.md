[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](https://anaconda.org/bioconda/accusnv)

# <img src="https://github.com/liaoherui/AccuSNV/blob/main/readme_files/logo.png" width = "100" height = "100" >  High-accuracy SNV calling for bacterial isolates using AccuSNV 

AccuSNV is a computational tool designed to identify single nucleotide variants (SNVs) in short-read whole genome sequencing (WGS) data from bacterial isolates. By leveraging deep learning, it classifies SNVs as either true or false, improving the accuracy of variant detection. The tool takes WGS data and a reference genome as input, and outputs a high-quality mutation table along with text and HTML reports. Additionally, it facilitates detailed downstream analysis, including phylogenetic tree construction and evolutionary analysis, among other features.

The workflow of AccuSNV:

# <img src="https://github.com/liaoherui/AccuSNV/blob/main/readme_files/method_fix.jpg" width = "800" height = "500" >  

-------------------------------------------------

### Version: V1.0.0.6 (Last update on 2025-03-05)
-------------------------------------------------


Note: This tool is powered by Lieberman and Key Lab SNV calling pipeline - [WideVariant](https://github.com/liebermanlab/WideVariant).

## Install

Git clone:<BR/>
`git clone https://github.com/liaoherui/AccuSNV.git`<BR/>


Change the permission of the file:<BR/>
`cd AccuSNV`<BR/>
`chmod 777 slurm_status_script.py`<BR/>

Install via bioconda:<BR/>
`mamba create -n accusnv -c conda-forge -c bioconda accusnv` or <BR/>
`conda create -n accusnv -c conda-forge -c bioconda accusnv`<BR/>

then, `conda activate accusnv`

If this installation method doesn’t work for your case, there are several other options you can try. See here.

## Quick Test (with the provided test data)

1. Test the tool on your Laptop (Support Linux or Ubuntu systems only):

```
# Step-1: Snakemake pipeline
python accusnv_snakemake.py -f 1 -i test_data_csv/samples_cae_test_pe.csv -r reference_genomes -o cae_pe_test_snakemake

# Snakemake dry-run step: simulates the execution of a workflow without actually running any jobs or creating output files
sh scripts/dru-run.sh

# Run the pipeline locally
sh scripts/run_snakemake_local.sh

# Step-2: Downstream analysis
python accusnv_downstream.py -i cae_pe_test_snakemake/3-AccuSNV/group_pe_test/candidate_mutation_table_final.npz -r reference_genomes/Cae_ref -o cae_accusnv_pe_downstream

----------------------------------------------------------------------------------------------------------------------------------
Note: Running the tool locally is convenient, but it may not fully utilize the capabilities of the Snakemake framework,
which can execute many jobs in parallel by submitting them to different nodes or partitions on an HPC cluster.
To improve efficiency (especially for large-scale datasets),
it is recommended to run the Snakemake pipeline on an HPC system with Slurm (see the example below).
```



2. Test the tool on the Linux HPC system with Slurm system:

```
# Step-1: Snakemake pipeline
python accusnv_snakemake.py -i test_data_csv/samples_cae_test_pe.csv -r reference_genomes -o cae_pe_test_snakemake

# Snakemake dry-run step
sh scripts/dru-run.sh

# Run the pipeline on HPC compute nodes; the jobs will be automatically submitted through the Slurm system.
sh scripts/run_snakemake.slurm
# (This Slurm script starts the entire pipeline. You can modify it as needed (e.g., change the partition for job submission).)

# Step-2: Downstream analysis (as this step requires minimal computational resources, it can still be run directly on a laptop.)
python accusnv_downstream.py -i cae_pe_test_snakemake/3-AccuSNV/group_pe_test/candidate_mutation_table_final.npz -r reference_genomes/Cae_ref -o cae_accusnv_pe_downstream

----------------------------------------------------------------------------------------------------------------------------------
Note: If you got error like "ValueError: The binary mode of fromstring is removed, use frombuffer instead",
This is because: On some clusters, activating your conda environment on compute nodes may require additional steps.
To solve it, in your scripts/run_snakemake.slurm file, you may need something like the following:

conda activate accusnv
```

To adjust the Slurm configuration (e.g., the partitions to submit to, CPU and memory requirements for specific tasks, or the maximum number of submitted jobs), you can modify the config.yaml file in the output folder generated in Step 1 (in this example: `cae_pe_test_snakemake/conf/config.yaml`). Some notes on how to modify this file can be found here.


3. For either Test (1) or Test (2), if the jobs finish successfully, the output folders should look like [this](readme_files/readme_test_output.md).
4. For more example command lines for the Step 1 Snakemake pipeline (e.g., using `samclip` or replacing bwa with `bowtie2`), please check the file `test_run.sh`.


## Usage

Key point: Ensure that all of your input files follow the same format as the tested files used in the **Quick Test** above.




