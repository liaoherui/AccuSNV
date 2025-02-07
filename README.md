# High-accuracy SNV calling for bacterial isolates using AccuSNV (Local test version by Herui)
-------------------------------------------------

### Version: Vxxxx (update on 2024-09-19)
-------------------------------------------------

## Install

Git clone:<BR/>
`git clone https://github.com/liaoherui/WideVariant-DL.git`<BR/>

### For Snakemake pipeline

`cd WideVariant-DL/snake_pipeline`<BR/>

Build the conda environment:<BR/>
`conda env create -n widevariant --file widevariant.yaml` or <BR/>`mamba env create -n widevariant --file widevariant.yaml` <BR/>
<!--- If the command above doesn't work, you may consider:<BR/>
`conda create -n widevariant snakemake biopython -y` or <BR/>
`mamba create -n widevariant snakemake biopython -y` <BR/> --> 

Activate the conda environment:<BR/>
`conda activate widevariant`<BR/>

Build other conda environments required by snakemake:<BR/>
`sh script/install_subenv.sh`<BR/>

Change the permission of the file:<BR/>
`chmod 777 slurm_status_script.py`<BR/>

### For Local analysis module



## Overview

This pipeline and toolkit is used to detect and analyze single nucleotide differences between closely related bacterial isolates. 

* Noteable features
	* Avoids false-negative mutations due to low coverage; if a mutation is found in at least one isolate in a set, the evidence at that position will be investigated to make a best-guess call.
	* Avoids false-positives mutations by facilitating visualization of raw data, across samples (whereas pileup formats must be investigated on a sample-by-sample basis) and changing of threshold to best fit your use case.
	* Enables easy evolutionary analysis, including phylogenetic construction, nonsynonmous vs synonymous mutation counting, and parallel evolution


* Inputs (to Snakemake cluster step): 
	* short-read sequencing data of closely related bacterial isolates
	* an annotated reference genome
* Outputs (of local analysis step): 
	* table of high-quality SNVs that differentiate isolates from each other
	* parsimony tree of how the isolates are related to each other 

The pipeline is split into two main components, as described below. A complete tutorial can be found at the bottom of this page.


### 1. Snakemake pipeline

The first portion of WideVariant aligns raw sequencing data from bacterial isolates to a reference genome, identifies candidate SNV positions, and creates useful data structure for supervised local data filtering. This step is implemented in a workflow management system called [Snakemake](http://snakemake.readthedocs.io) and is executed on a [SLURM cluster](https://slurm.schedmd.com/documentation.html). More information is available [here](readme_files/readme_snake_main.md).

#### 1.1 Update - 2024-09-22: A user-friendly Python script is now available to help users run the pipeline more easily. Instructions are provided below:

Make sure to configure your `config.yaml` file and `script/run_snakemake.slurm` before starting the steps below..

Step-1: run the python script: <BR/>

`python widevariant.py -i <input_sample_info_csv> -r <ref_dir> -o <output_dir>`


Step-2: check the pipeline using "dry-run"<BR/>

`sh script/dry-run.sh`<BR/>

Step-3: submit your slurm job.<BR/>

`sbatch script/run_snakemake.slurm`<BR/>




### 2. Local python analysis

The second portion of WideVariant filters candidate SNVs based on data arrays generated in the first portion and generates a high-quality SNV table and a parsimony tree. This step is implemented with a custom python script. More information can be found [here](readme_files/readme_local_main.md).


## Tutorial Table of Contents

[Main WideVariant pipeline README](README.md)
* [Snakemake pipeline](readme_files/readme_snake_main.md)
	* [Overview and how to run the snakemake pipeline](readme_files/readme_snake_run.md)
	* [Technical details about the snakemake pipeline](readme_files/readme_snake_rules.md)
	* [Wishlist for snakemake pipeline upgrades](readme_files/readme_snake_wishlist.md)
	* [Helpful hints for using the command line](readme_files/readme_snake_basics.md)
* [Local analysis](readme_files/readme_local_main.md)
	* [How to run the local analysis script](readme_files/readme_local_run.md)
	* [Wishlist for local analysis upgrades](readme_files/readme_local_wishlist.md)
	* [Python best practices](readme_files/readme_local_best.md)


## Example use cases

Previous iterations of this pipeline have been used to study:
* [_C. acnes_ biogeography in the human skin microbiome](https://www.sciencedirect.com/science/article/pii/S1931312821005783)
* [Adaptive evolution of _S. aureus_ on patients with atopic dermatitis](https://www.biorxiv.org/content/10.1101/2021.03.24.436824v3)
* [Adaptive evolution of _B. fragilis_ on healthy people](https://www.sciencedirect.com/science/article/pii/S1931312819301593)


