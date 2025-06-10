# <img src="https://github.com/liaoherui/AccuSNV/blob/main/readme_files/logo.png" width = "100" height = "100" >  High-accuracy SNV calling for bacterial isolates using AccuSNV 
-------------------------------------------------

### Version: V0.0.0.1 (Last update on 2025-04-26)
-------------------------------------------------

Note: This tool is powered by Lieberman Lab SNV calling pipeline - [WideVariant](https://github.com/liebermanlab/WideVariant).

## Install

Git clone:<BR/>
`git clone https://github.com/liaoherui/AccuSNV.git`<BR/>

### Option-1 (via pre-built conda env - Linux only, recommended!)

`cd AccuSNV/snake_pipeline`<BR/>

If you don't have `gdown`, pleae install it first:<BR/>
`pip install gdown`

Download pre-built environments:<BR/>
`sh download_install_env.sh`<BR/><BR/>
Note: Please ignore the error message: `tar: Exiting with failure status due to previous errors`. You can still use the environment despite receiving this error message.

Activate the pre-built environment<BR/>
`source accusnv_env/bin/activate`

Change the permission of the file:<BR/>
`chmod 777 slurm_status_script.py`<BR/>


### Option-2 (via .yaml file)

`cd AccuSNV/snake_pipeline`<BR/>

Build the conda environment:<BR/>
`conda env create -n accusnv_env --file accusnv.yaml` or <BR/>`mamba env create -n accusnv_env --file accusnv.yaml` <BR/>

Activate the conda environment:<BR/>
`conda activate accusnv_env`<BR/>

Copy conda-env-based Snakefile:<BR/>
`cp Snakefiles_diff_options/Snakefile_conda_env.txt  ./Snakefile`<BR/>

Change the permission of the file:<BR/>
`chmod 777 slurm_status_script.py`<BR/>

### Option-3 (via Bioconda, under processing)

------------------------------------------------------------------------------------
Once you finish the install, you can test the tool with the command lines below :<BR/>

Test snakemake pipeline - Under `snake_pipeline` folder:<BR/>
`sh test_run.sh`<BR/>
`sh scripts/dry-run.sh`<BR/>
`sbatch scripts/run_snakemake.slurm`<BR/>

Test downstream analysis - Under `local_analysis` folder:<BR/>
`sh test_local.sh`<BR/>

## Overview

This pipeline and toolkit is used to detect and analyze single nucleotide differences between closely related bacterial isolates. 

* Noteable features
	* Avoids false negatives from low coverage and false positives through a deep learning method, while also enabling visualization of raw data.
	* Enables easy evolutionary analysis, including phylogenetic construction, nonsynonmous vs synonymous mutation counting, and parallel evolution, etc.


* Inputs (to Snakemake cluster step): 
	* short-read sequencing data of closely related bacterial isolates
	* an annotated reference genome
* Outputs (of downstream analysis step): 
	* table of high-quality SNVs that differentiate isolates from each other
	* parsimony tree of how the isolates are related to each other
   	* More details can be found in [here](#output)

The pipeline is split into two main components, as described below. 

### 1. Snakemake pipeline

The first portion of AccuSNV aligns raw sequencing data from bacterial isolates to a reference genome, identifies candidate SNV positions, and creates useful data structure for supervised local data filtering. This step is implemented in a workflow management system called [Snakemake](http://snakemake.readthedocs.io) and is executed on a [SLURM cluster](https://slurm.schedmd.com/documentation.html). More information is available [here](readme_files/readme_snake_main.md).

<!--- #### 1.1 Update - 2025-02-21: A user-friendly Python script is now available to help users run the pipeline more easily. Instructions are provided below:


Make sure to configure your `config.yaml` file and `script/run_snakemake.slurm` before starting the steps below.. -->

Please ensure the right permission of the file `slurm_status_script.py`:

`chmod 777 slurm_status_script.py`<BR/>

Step-1: run the python script: <BR/>

`python accusnv_snakemake.py -i <input_sample_info_csv> -r <ref_dir> -o <output_dir>`

One example with test data can be found in `snake_pipeline/test_run.sh`

If you cloned the repository (e.g. a new download) and have already downloaded the pre-built Conda environment (e.g., /path/snake_pipeline/accusnv_sub), there's no need to download it again. Just try:

`python accusnv_snakemake.py -i <input_sample_info_csv> -c /path/snake_pipeline/accusnv_sub -r <ref_dir> -o <output_dir>`


Step-2: check the pipeline using "dry-run"<BR/>

`sh scripts/dry-run.sh`<BR/>

Step-3: submit your slurm job.<BR/>

`sbatch scripts/run_snakemake.slurm`<BR/>

Note: If you need to modify any slurm job configuration, you can edit the config.yaml file generated in your output folder: `<output_dir>/conf/config.yaml`



### 2.1. Local python analysis

Note: This step has been incorporated into the Snakemake pipeline and will be executed automatically by default. However, you can still use this local Python script to rerun the analysis with different parameters if needed.

`python new_snv_script.py -i <input_mutation_table> -c <input_raw_coverage_matrix> -r <ref_dir> -o <output_dir>`

One example with test data can be found in `local_analysis/test_local.sh`

The second portion of AccuSNV filters candidate SNVs based on data arrays generated in the first portion and generates a high-quality SNV table and a parsimony tree. This step utilizes deep learning and is implemented with a custom Python script. More information can be found [here](readme_files/readme_local_main.md).

### 2.2. Local downstream analysis

Based on the identified SNVs and the output final mutation table (in .npz format), AccuSNV offers a set of downstream analysis modules (e.g. dN/dS calculation). You can run these modules using the command below.

`python accusnv_downstream.py -i  test_data/candidate_mutation_table_final.npz -r ../snake_pipeline/reference_genomes/Cae_ref -o cae_accusnv_ds_pe`


### Full command-line options

Snakemake pipeline - accusnv_snakemake.py 
```
AccuSNV - SNV calling tool for bacterial isolates using deep learning.

options:
  -h, --help            show this help message and exit
  -i INPUT_SP, --input_sample_info INPUT_SP
                        The dir of input sample info file --- Required
  -t TF_SLURM, --turn_off_slurm TF_SLURM
                        If set to 1, the SLURM system will not be used for automatic job
                        submission. Instead, all jobs will run locally or on a single
                        node. (Default: 0)
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
  -e EXCLUDE_SAMP, --excluse_samples EXCLUDE_SAMP
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
                        The dir of your reference genomes
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
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        The output dir
```


## Output


<img src="https://github.com/liaoherui/AccuSNV/blob/main/readme_files/output_downstream_new_single.png" width = "700" height = "650" >  

The main output folder structure of Snakemake pipeline is shown below:

```
1-Mapping - Alignment temporary files
2-Case - candidate mutation tables for 3-AccuSNV
3-AccuSNV - Main output of Snakemake pipeline
```


Important and major output files:
Header    |Description	
------------ | ------------- 
candidate_mutation_table_final.npz | NPZ table used for downstream analysis modules.
snv_table_merge_all_mut_annotations_final.tsv | Text report - contain identified SNVs and related information.
snv_qc_heatmap_*.png | QC figures
snv_table_with_charts_final.html | Html report - display the comprehensive information about identified SNVs. Note, if you want to see the bar charts in the html file, make sure you have the folder "bar_charts" under the same folder with the html file.





<!--- ## Tutorial Table of Contents

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
* [Adaptive evolution of _B. fragilis_ on healthy people](https://www.sciencedirect.com/science/article/pii/S1931312819301593) -->


