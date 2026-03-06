[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](https://anaconda.org/bioconda/accusnv)

# <img src="https://github.com/liaoherui/AccuSNV/blob/main/readme_files/logo.png" width = "100" height = "100" >  High-accuracy SNV calling for bacterial isolates using AccuSNV 

AccuSNV is a computational tool designed to identify single nucleotide variants (SNVs) in short-read whole genome sequencing (WGS) data from bacterial isolates. By leveraging deep learning, it classifies SNVs as either true or false, improving the accuracy of variant detection. The tool takes WGS data and a reference genome as input, and outputs a high-quality mutation table along with text and HTML reports. Additionally, it facilitates detailed downstream analysis, including phylogenetic tree construction and evolutionary analysis, among other features.

The workflow of AccuSNV:

# <img src="https://github.com/liaoherui/AccuSNV/blob/main/readme_files/method_fix.jpg" width = "800" height = "500" >  

-------------------------------------------------

### Version: V1.0.0.6 (Last update on 2026-Mar)
-------------------------------------------------


Note: This tool is powered by Lieberman and Key Lab SNV calling pipeline - [WideVariant](https://github.com/liebermanlab/WideVariant).

## Contents

- [Install](#install)
- [Quick Test](#quick-test)
- [Usage](#usage)
- [Output](#output)
- [Full command-line options](#full-command-line-options)
- [Contact](#-contact-)
- [Cite](#references)


-------------------------------------------------

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

## Quick Test 

Quick test with the provided test data.

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

# (or: python accusnv_downstream.py -i  Test_data_local/candidate_mutation_table_final.npz -r reference_genomes/Cae_ref -o cae_accusnv_ds_pe_downstream)

----------------------------------------------------------------------------------------------------------------------------------
# Note: Running the tool locally is convenient, but it may not fully utilize the capabilities of the Snakemake framework,
# which can execute many jobs in parallel by submitting them to different nodes or partitions on an HPC cluster.
# To improve efficiency (especially for large-scale datasets),
# it is recommended to run the Snakemake pipeline on an HPC system with Slurm (see the example below).
----------------------------------------------------------------------------------------------------------------------------------
```



2. Test the tool on the Linux HPC system with [Slurm](https://slurm.schedmd.com/overview.html) system:

```
# Step-1: Snakemake pipeline
python accusnv_snakemake.py -i test_data_csv/samples_cae_test_pe.csv -r reference_genomes -o cae_pe_test_snakemake

# Snakemake dry-run step
sh scripts/dru-run.sh

# Run the pipeline on HPC compute nodes; the jobs will be automatically submitted through the Slurm system.
sbatch scripts/run_snakemake.slurm
# (This Slurm script starts the entire pipeline. You can modify it as needed (e.g., change the partition for job submission).)

# Step-2: Downstream analysis (as this step requires minimal computational resources, it can still be run locally.You can also test this step directly using the provided test data, see the “or” option below.)
python accusnv_downstream.py -i cae_pe_test_snakemake/3-AccuSNV/group_pe_test/candidate_mutation_table_final.npz -r reference_genomes/Cae_ref -o cae_accusnv_pe_downstream

# (or: python accusnv_downstream.py -i  Test_data_local/candidate_mutation_table_final.npz -r reference_genomes/Cae_ref -o cae_accusnv_ds_pe_downstream)

----------------------------------------------------------------------------------------------------------------------------------
# Note: If you got error like "ValueError: The binary mode of fromstring is removed, use frombuffer instead",
# This is likely because: On some clusters, activating your conda environment on compute nodes may require additional steps.
# To solve it, in your scripts/run_snakemake.slurm file, you may need something like the following:

# conda activate accusnv
----------------------------------------------------------------------------------------------------------------------------------
```

To adjust the Slurm configuration (e.g., the partitions to submit to, CPU and memory requirements for specific tasks, or the maximum number of submitted jobs), you can modify the config.yaml file in the output folder generated in Step 1 (in this example: `cae_pe_test_snakemake/conf/config.yaml`). Some notes on how to modify this file can be found here.


3. For either Test (1) or Test (2), if the jobs finish successfully, the output folders should look like [this](readme_files/readme_test_output.md).
4. For more example command lines for the Step 1 Snakemake pipeline (e.g., using `samclip` or replacing `bwa` with `bowtie2`), please check the file `test_run.sh`.


## Usage

Key point: Ensure that all of your input files follow the same format as the tested files used in the **Quick Test** above.

To run the tool, you will need to:

### (1) Prepare inputs

- A sample sheet CSV (same format as the **Quick Test** CSV files). Details about this file can be found [here](readme_files/readme_input_csv.md). Examples can be found in the folder [test_data_csv](test_data_csv/).

- A reference genome directory (each reference should have `genome.fasta`; annotations such as `genome.gff` (generated by [Prokka](https://github.com/tseemann/prokka) or [Bakta](https://github.com/oschwengers/bakta)) are recommended for richer outputs). Examples can be found in the folder [reference_genomes](reference_genomes/).

An example of the input directory structure and corresponding input files can be found [here](readme_files/readme_input_csv.md).

### (2)  Run the Snakemake pipeline

```
# from AccuSNV root
python accusnv_snakemake.py -i <samples.csv> -r <reference_genomes_dir> -o <output_dir>
```

<!---
# or Bioconda command (equivalent to the above)
# accusnv_snakemake -i <samples.csv> -r <reference_genomes_dir> -o <output_dir>
-->

Then check workflow plan with Snakemake dry run:

`sh scripts/dry_run.sh`

Submit with Slurm:

`sbatch scripts/run_snakemake.slurm`

If you do not use Slurm (single node/local-style run), set:

`python accusnv_snakemake.py -f 1 -i <samples.csv> -r <reference_genomes_dir> -o <output_dir>`

(`-f 1` disables automatic Slurm submission mode).

### (3) Run downstream analysis (optional but recommended for dN/dS and re-analysis)

Use the final NPZ from:
`<output_dir>/3-AccuSNV/group_<group_id>/candidate_mutation_table_final.npz`

```
python accusnv_downstream.py \
  -i <output_dir>/3-AccuSNV/group_<group_id>/candidate_mutation_table_final.npz \
  -r <reference_genomes_dir>/<reference_name> \
  -o <downstream_output_dir>
```
Note: NPZ files under `2-Case/candidate_mutation_table` are for the local AccuSNV script (`new_snv_script.py`), not for `accusnv_downstream`

<!---
# Bioconda command (equivalent to the above):
# accusnv_downstream -i <final_cmt_npz> -r <reference_dir> -o <downstream_output_dir>
-->

### (4) Main files to look at

(Core output are in the `<output_dir>/3-AccuSNV/<folder_name>` from (2). E.g., for **Quick Test**, this is `cae_pe_test_snakemake/3-AccuSNV/group_pe_test/` folder.)

Final SNV table: `snv_table_merge_all_mut_annotations_final.tsv`

Interactive report: `snv_table_with_charts_final.html` (keep `bar_charts/` beside it).

Downstream input/output anchor: `candidate_mutation_table_final.npz`.

You can find examples to these four output files in the [demo_output](demo_output) folder.


## Output

The main output files  are in the `<output_folder>/3-AccuSNV/<folder_name>` (e.g., for **Quick Test**, this is `cae_pe_test_snakemake/3-AccuSNV/group_pe_test/`) folder. 
### Core files:

| File or Folder |  Description |
| ---  | --- | 
| `snv_table_merge_all_mut_annotations_final.tsv`  | Final merged SNV report table (recommended primary text result for interpretation). More details, including explanations of the columns in this file, can be found here.
| `snv_table_cnn_plus_filter.txt` | Per-position prediction/filter summary table (CNN output + rule-based filters (from WideVariant)). Note that this file does not include annotation information for each SNV.
| `snv_table_with_charts_final.html`  | Interactive final HTML report for the final merged table (recommended to view). Keep `bar_charts/` in the same output folder so image links work.
| `candidate_mutation_table_final.npz`  | Final machine-readable SNV matrix for downstream analysis. Contains arrays such as sample names, genomic positions, counts, quality values, prediction labels/probabilities, and recombination flags. This is the main input for `accusnv_downstream`.

You can find examples to these four core output files in the [demo_output](demo_output) folder.

For final SNV calling results, please use:

`snv_table_merge_all_mut_annotations_final.tsv` as the primary human-readable SNV result table (final filtered/merged report).

`candidate_mutation_table_final.npz` as the machine-readable final result for any downstream analysis or re-analysis.

For full documentation of all output files, please see [here](readme_files/readme_test_output.md).

A demo output HTML report of AccuSNV can be found at https://heruiliao.github.io/

## Full command-line options

Snakemake pipeline - accusnv_snakemake.py 
```
AccuSNV - SNV calling tool for bacterial isolates using deep learning.

options:
  -h, --help            show this help message and exit
  -i INPUT_SP, --input_sample_info INPUT_SP
                        The dir of input sample info file --- Required
  -a ALIGNER, --aligner ALIGNER
                        The aligner used for read mapping, can be either BWA or Bowtie2. E.g. You can set "-a bowtie2" to use bowtie2. (Default: -a bwa)
  -p SAMCLIP, --samclip SAMCLIP
                        If set to 1, samclip will be used when the aligner is BWA. Note that this parameter is not applicable when Bowtie2 is used as the
                        aligner. (Default: 0)
  -t ALIGNER_THREADS, --aligner_threads ALIGNER_THREADS
                        The threads for the aligner - bwa or bowtie2 (Default: 4)
  -f TF_SLURM, --turn_off_slurm TF_SLURM
                        If set to 1, the SLURM system will not be used for automatic job submission. Instead, all jobs will run locally or on a single node.
                        (Default: 0)
  -c CP_ENV, --conda_prebuilt_env CP_ENV
                        The absolute dir of your pre-built conda env. e.g.
                        /path/snake_pipeline/accusnv_sub
  -r REF_DIR, --ref_dir REF_DIR
                        The dir of your reference genomes
  -s MIN_COV_SAMP, --min_cov_for_filter_sample MIN_COV_SAMP
                        Before running the CNN model, low-quality samples with more than
                        45% of positions having zero aligned reads will be filtered out.
                        (default "-s 45") You can adjust this threshold with this
                        parameter; to include all samples, set "-s 100".
  -v MIN_COV, --min_cov_for_filter_pos MIN_COV
                        For the filter module: on individual samples, calls must have at
                        least this many reads on the fwd/rev strands individually. If
                        many samples have low coverage (e.g. <5), then you can set this
                        parameter to smaller value. (e.g. -v 2). Default is 5.
  -e EXCLUDE_SAMP, --exclude_samples EXCLUDE_SAMP
                        The names of the samples you want to exclude (e.g. -e S1,S2,S3).
                        If you specify a number, such as "-e 1000", any sample with more
                        than 1,000 SNVs will be automatically excluded.
  -g GENERATE_REP, --generate_report GENERATE_REP
                        If not generate html report and other related files, set to 0.
                        (default: 1)
  -o OUT_DIR, --output_dir OUT_DIR
                        Output dir (default: current dir/wd_out_(uid), uid is generated
                        randomly)

```

Local downstream analysis - accusnv_downstream.py

```
SNV calling tool for bacterial isolates using deep learning.

options:
  -h, --help            show this help message and exit
  -i INPUT_MAT, --input_mat INPUT_MAT
                        The input mutation table in npz file
  -r REF_DIR, --ref_dir REF_DIR
                        The dir of your reference genomes (must contain 
        		file called "genome.fasta" as well as a GFF file "*.gff" or "*.gff3")
  -c MIN_COV, --min_cov_for_call MIN_COV
                        For the fill-N module: on individual samples, calls must have at
                        least this many fwd+rev reads. Default is 1.
  -q MIN_QUAL, --min_qual_for_call MIN_QUAL
                        For the fill-N module: on individual samples, calls must have at
                        least this minimum quality score. Default is 30.
  -b EXCLUDE_RECOMB, --exclude_recomb EXCLUDE_RECOMB
                        Whether included SNVs from potential recombinations. Default
                        included. Set "-b 1" to exclude these positions in downstream
                        analysis modules.
  -f MIN_FREQ, --min_freq_for_call MIN_FREQ
                        For the fill-N module: on individual samples, a call's major
                        allele must have at least this freq. Default is 0.75.
  -S EXCLUDE_SAMPLE_IDS, --exclude_sample_ids EXCLUDE_SAMPLE_IDS
                        Comma-separated sample IDs to exclude from analysis
  -P EXCLUDE_POSITION_IDS, --exclude_position_ids EXCLUDE_POSITION_IDS
                        Comma-separated genomic positions to exclude from analysis
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        The output dir
```

## -Contact-
  
 If you have any questions, please post an issue on GitHub or email us: herui728@mit.edu

## References:

how to cite this tool:
```
Liao, H., Conwill, A., & Light-Maka, Ian., et al. High-accuracy SNV calling for bacterial isolates using deep learning with AccuSNV, BioRxiv, 2025; https://www.biorxiv.org/content/10.1101/2025.09.26.678787v2
```




