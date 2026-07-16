[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](https://anaconda.org/bioconda/accusnv)

# <img src="https://github.com/liaoherui/AccuSNV/blob/main/readme_files/logo.png" width = "100" height = "100" >  High-accuracy SNV calling for bacterial isolates using AccuSNV 

### Version: V1.0.1.0 (Last update on 2026-July)

AccuSNV is a computational tool designed to identify single nucleotide variants (SNVs) in short-read whole genome sequencing (WGS) data from bacterial isolates. By leveraging deep learning, it classifies SNVs as either true or false, improving the accuracy of variant detection. AccuSNV takes WGS data and a reference genome as input, and outputs a high-quality mutation table along with text and HTML reports. Additionally, it facilitates detailed downstream analysis, including phylogenetic tree construction and evolutionary analysis, among other features.



The architecture of the AccuSNV convolutional neural network:

# <img src="https://github.com/liaoherui/AccuSNV/blob/main/readme_files/method_fix.jpg" width = "800" height = "500" >  


## Overview

This pipeline and toolkit is used to identify and analyze single nucleotide differences between bacterial isolates from short read WGS data. 

* Key features
	* Excludes false SNVs through deep learning classification, and enables visualization of raw data behind each SNV call.
	* Enables downstream evolutionary analyses, including phylogenetic tree construction, nonsynonmous vs synonymous mutation counting, and detection of parallel evolution.

* Inputs
	* Short-read sequencing FASTQ data from multiple (>=3) bacterial isolates.
  * A sample sheet describing the reference genome, grouping, and outgroup to use for each sample.
	* Reference genome FASTA file(s), with optional annotations (gff). 
  * More details can be found under [Usage](#full-usage).
  
* Outputs 
	* Table of high-quality SNVs differentiating the isolates in the dataset.
  * Mutation types and their associated gene annotations.
  * dN/dS and dMRCA calculations.
  * Parsimony tree of isolates.
  * More details can be found under [Output](#output).


Note: This tool is based on the Lieberman and Key Lab SNV calling pipeline - [WideVariant](https://github.com/liebermanlab/WideVariant). We are maintaining an updated version of that pipeline in this repository for preprocessing data for input to the AccuSNV CNN.


## Contents

- [Install](#install)
- [Quick Test](#quick-test-local)
- [Usage](#full-usage)
- [Output](#output)
- [Full command-line options](#full-command-line-options)
- [Contact](#contact)
- [Cite](#citation)


-------------------------------------------------

## Install

To install AccuSNV, you must both clone the GitHub repository and also install the required dependencies by installing the conda/mamba package.

Git clone:<BR/>
`git clone https://github.com/liaoherui/AccuSNV.git`<BR/>


Change the permission of the SLURM status script file:<BR/>
`cd AccuSNV`<BR/>
`chmod +x slurm_status_script.py`<BR/>

Install dependencies via bioconda:<BR/>
`conda create -n accusnv -c conda-forge -c bioconda accusnv` or <BR/>
`mamba create -n accusnv -c conda-forge -c bioconda accusnv` <BR/>

then, `conda activate accusnv`

## Quick Test (local)

 Run in a local compute environment (e.g., laptop or on a single node) by passing the `-f local` parameter:

### 1. Prepare output directory and config files for the pipeline:
```
./accusnv prepare -f local -i Test_data/samples_cae_test_pe.csv -r Test_data/reference_genomes -o cae_accusnv_output
```

Here, `samples_cae_test_pe.csv` is an example input sample table, and `Test_data/reference_genomes` is an example directory of reference genomes for these samples.

### 2. Run the pipeline locally:
```
./accusnv run -o cae_accusnv_output
```

## Quick Test (HPC cluster) 

Run on a **Linux HPC system (cluster)** with a [Slurm](https://slurm.schedmd.com/overview.html) scheduler. Typically, we recommend that you run AccuSNV on an HPC cluster, and we provide out of the box Snakemake support for Slurm clusters, and this is currently the default run mode:

### 1. Prepare output directory and config files for the pipeline (Slurm):
```
./accusnv prepare -f slurm -i Test_data/samples_cae_test_pe.csv -r Test_data/reference_genomes -o cae_pe_test_snakemake
```

### 2. Submit and run the pipeline on an HPC Slurm cluster:

```
./accusnv run -o cae_pe_test_snakemake
```

This runs a Slurm script that starts the entire pipeline. **You must first modify it to fit your HPC specifications** by modifying the `conf/config.yaml` file in the output folder generated in Step 1 (in this example: `cae_pe_test_snakemake/conf/config.yaml`). You will likely want to modify the partitions to submit to, CPU and memory requirements for specific tasks, or the maximum number of submitted jobs.

>For a description of the resulting output files, see [Output files of AccuSNV](readme_files/readme_test_output.md), or read on below.

>*Note*: If you receive an error like "ValueError: The binary mode of fromstring is removed, use frombuffer instead", This is likely because on some clusters, you must first activate your conda environment on compute nodes. To solve it, in your `<output_dir>/run_snakemake.slurm` file, you may need something like the following: `conda activate accusnv`.

## Full Usage

First, you must ensure that all of your input files follow the same format as the example files used in the **Quick Test** above.

### 1. Prepare necessary input files

AccuSNV requires three types of inputs: 

- **A sample sheet CSV** 
  - Sample sheets 
  - Same format as the **Quick Test** CSV files. Examples can be found in the folder [Test_data](Test_data/) (e.g. `Test_data/samples_cae_test_pe.csv`).
  - In this sample sheet, individual samples can be assigned to sample *groups*. Samples within the same Group are analyzed together and separately from other samples.
  - Detailed description for this file can be found [here](readme_files/readme_input_csv.md).

- **A directory of FASTQ files.** 
  - AccuSNV does not take FASTQ paths directly. For each sample it locates the reads by combining the Path (the folder to search) and the FileName (the read-file prefix) columns from your sample sheet, and then searching the Path folder for files that match.
  - Accepts gzipped or plain FASTQ files with extensions: .fastq.gz, .fq.gz, .fastq, or .fq.
  - For paired end reads, R1/R2 files must be distinguished by a `1`/`2` joined by `_`, `.`, or `-`. 
- **A reference genome directory** 
  - Each reference should be in its own subfolder within the reference genome directory, and have a file named exactly `genome.fasta`.
  - Annotations such as `genome.gff` (generated by [Prokka](https://github.com/tseemann/prokka) or [Bakta](https://github.com/oschwengers/bakta)) are recommended for richer outputs. 
  - Avoid reference folder names that start with `ref_` or contain `_ref_` because AccuSNV uses `_ref_` as an internal filename delimiter. 
  - AccuSNV can generate BWA/samtools indexes during the Snakemake run (`genome.fasta.bwt`, `.amb`, `.ann`, `.pac`, `.sa`, and `.fai`); if the reference directory is not writable, pre-create them with `bwa index genome.fasta` and `samtools faidx genome.fasta`. 
  - Examples can be found in the folder [Test_data/reference_genomes](Test_data/reference_genomes/).



A detailed description of the input directory structure and files can be found here: [Input files of AccuSNV](readme_files/readme_input_csv.md).

### 2.  Run the Snakemake pipeline

For running on an HPC with Slurm, first prepare the config and input files:

```
./accusnv prepare -f slurm -i <samples.csv> -r <reference_genomes_dir> -o <output_dir>
```

You can first test the workflow with a Snakemake dry run: `./accusnv run --dryrun -o <output_dir>`.

To then submit with Slurm:

```
./accusnv run -o <output_dir>
```

If you do not use Slurm (single node/local run), instead run:

```
./accusnv prepare -f local -i <samples.csv> -r <reference_genomes_dir> -o <output_dir>
```

(`-f local` disables automatic Slurm submission mode).

and then:

```
./accusnv run -o <output_dir>
```

### 3. Re-run downstream evolutionary analyses separately

By default, the full AccuSNV snakemake pipeline will calculate distance to Most Recent Common Ancestor (dMRCA), dN/dS, and identify parallel mutations. 

However, users may often wish to re-run these analyses without re-running the entire SNV calling pipeline. To do so, you can use the final NPZ from:
`<output_dir>/3-AccuSNV/group_<group_id>/candidate_mutation_table_final.npz`

And then run:
```
python accusnv_downstream.py \
  -i <output_dir>/3-AccuSNV/group_<group_id>/candidate_mutation_table_final.npz \
  -r <reference_genomes_dir>/<reference_name> \
  -o <downstream_output_dir>
```
Note: NPZ files under `2-Case/candidate_mutation_table` are for the local AccuSNV script (`new_snv_script.py`), not for `accusnv_downstream`

In the future, we plan to facilitate more interactive versions of these evolutionary analyses that allow the user to visualize, inspect, and interact with their data. 

## Output

The main output files for each sample group are found in the `<output_folder>/3-AccuSNV/group_[group_name]` directory (e.g., for **Quick Test**, this is the `cae_pe_test_snakemake/3-AccuSNV/group_pe_test/`). 

### Core output files:

| File or Folder |  Description |
| ---  | --- | 
| `snv_table_merge_all_mut_annotations_final.tsv`  | Final merged SNV report table (recommended primary text result for interpretation). More details, including explanations of the columns in this file, can be found [here](readme_files/readme_annotation_table.md).
| `snv_table_cnn_plus_filter.txt` | Per-position prediction/filter summary table (CNN output + rule-based filters). Note that this file does not include annotation information for each SNV.
| `snv_table_with_charts_final.html`  | Interactive final HTML report for the final merged table (recommended to view). Keep `bar_charts/` in the same output folder so image links work.
| `candidate_mutation_table_final.npz`  | Final machine-readable full SNV matrix for downstream analysis. Contains arrays such as sample names, genomic positions, counts, quality values, prediction labels/probabilities, and recombination flags. This is the main input for `accusnv_downstream`.

You can find examples to these four core output files in the [demo_output](demo_output) folder.

For final SNV calling results, please use:

`snv_table_merge_all_mut_annotations_final.tsv` as the primary human-readable SNV result table (final filtered/merged report).

`candidate_mutation_table_final.npz` as the machine-readable final result for any downstream analysis or re-analysis.

For full documentation of all output files, please see [here](readme_files/readme_test_output.md).

A demo output HTML report of AccuSNV can be found at https://heruiliao.github.io/

### Downstream evolutionary analysis output files:
| File or Folder |  Description |
| ---  | --- | 
| `./3-AccuSNV/group_<group>/data_dNdS.npz`  | dN/dS values for each gene with mutations. Contains dNdS, a confidence interval, and the N/S mutation counts.
| `./3-AccuSNV/group_<group>/snv_tree_genome_latest.nwk.tree` | Newick parsimony tree built by **dnapars**.
| `./3-AccuSNV/group_<group>/snv_table_tree_distances.csv`  | Per-sample SNP distances to ancestor.
| `./3-AccuSNV/group_<group>/snp_trees/*`  | per-SNV files named p_<position>_<N>.tree where N ≥ 2 flags a homoplasic (parallel) mutation.



## Full command-line options

### `accusnv prepare` 
```
usage: AccuSNV [-h] -i INPUT_SP [-a ALIGNER] [-p SAMCLIP] [-t ALIGNER_THREADS] [-f {local,slurm}] [-c CP_ENV] [-r REF_DIR] [-s MIN_COV_SAMP] [-v MIN_COV] [-e EXCLUDE_SAMP] [-g GENERATE_REP] [-o OUT_DIR]

AccuSNV - SNV calling tool for bacterial isolates using deep learning.

options:
  -h, --help            show this help message and exit
  -i, --input_sample_info INPUT_SP
                        The dir of input sample info file --- Required
  -a, --aligner ALIGNER
                        The aligner used for read mapping, can be either BWA or Bowtie2. E.g. You can set "-a bowtie2" to use bowtie2. (Default: -a bwa)
  -p, --samclip SAMCLIP
                        If set to 1, samclip will be used when the aligner is BWA. Note that this parameter is not applicable when Bowtie2 is used as the aligner. (Default: 0)
  -t, --aligner_threads ALIGNER_THREADS
                        The threads for the aligner - bwa or bowtie2 (Default: 4)
  -f, --mode {local,slurm}
                        Execution mode: 'local' runs all jobs on a single node; 'slurm' submits jobs via the SLURM system. (Default: slurm)
  -c, --config_prebuilt_env CP_ENV
                        The absolute dir of your pre-built env. e.g. /path/snake_pipeline/accusnv_sub
  -r, --ref_dir REF_DIR
                        The dir of your reference genomes
  -s, --min_cov_for_filter_sample MIN_COV_SAMP
                        Before running the CNN model, low-quality samples with more than 45% of positions having zero aligned reads will be filtered out. (default "-s 45") You can adjust this threshold with this parameter; to
                        include all samples, set "-s 100".
  -v, --min_cov_for_filter_pos MIN_COV
                        For the filter module: on individual samples, calls must have at least this many reads on the fwd/rev strands individually. If many samples have low coverage (e.g. <5), then you can set this parameter to
                        smaller value. (e.g. -v 2). Default is 5.
  -e, --exclude_samples EXCLUDE_SAMP
                        The names of the samples you want to exclude (e.g. -e S1,S2,S3). If you specify a number, such as "-e 1000", any sample with more than 1,000 SNVs will be automatically excluded.
  -g, --generate_report GENERATE_REP
                        If not generate html report and other related files, set to 0. (default: 1)
  -o, --output_dir OUT_DIR
                        Output dir (default: current dir/wd_out_(uid), uid is generated randomly)
```

### `accusnv run`

```
usage: accusnv run [-h] -o OUTPUT_DIR [--dryrun]

options:
  -h, --help            show this help message and exit
  -o, --output_dir OUTPUT_DIR
                        Output dir created by 'accusnv prepare'.
  --dryrun              Simulate the run without executing any jobs.
```

### `python accusnv_downstream.py`

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

## Contact
  
 If you have any questions, please post an issue on GitHub or email us: 
 
 Herui Liao (creator) herui728@mit.edu

 Alex Crits-Christoph (maintainer) crits@mit.edu

## Citation

How to cite this software:

>Liao, Herui, Arolyn Conwill, Ian Light-Maka, Martin Fenk, Alyssa H. Mitchell, Evan B. Qu, Paul Torrillo, Jacob S. Baker, Lilly R. Bartsch, Felix M. Key, and Tami D. Lieberman. "[High-accuracy SNV calling for bacterial isolates using deep learning with AccuSNV](https://genome.cshlp.org/content/early/2026/07/02/gr281341125)." *Genome Research*. June 2026, Vol. 36, No. 6. [https://doi.org/10.1101/gr.281341.125](https://doi.org/10.1101/gr.281341.125)




