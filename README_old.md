[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](https://anaconda.org/bioconda/accusnv)

# <img src="https://github.com/liaoherui/AccuSNV/blob/main/readme_files/logo.png" width = "100" height = "100" >  High-accuracy SNV calling for bacterial isolates using AccuSNV 

AccuSNV is a computational tool designed to identify single nucleotide variants (SNVs) in short-read whole genome sequencing (WGS) data from bacterial isolates. By leveraging deep learning, it classifies SNVs as either true or false, improving the accuracy of variant detection. The tool takes WGS data and a reference genome as input, and outputs a high-quality mutation table along with text and HTML reports. Additionally, it facilitates detailed downstream analysis, including phylogenetic tree construction and evolutionary analysis, among other features.

The workflow of AccuSNV:

# <img src="https://github.com/liaoherui/AccuSNV/blob/main/readme_files/method_fix.jpg" width = "800" height = "500" >  

-------------------------------------------------

### Version: V1.0.0.5 (Last update on 2025-08-18)
-------------------------------------------------


Note: This tool is powered by Lieberman and Key Lab SNV calling pipeline - [WideVariant](https://github.com/liebermanlab/WideVariant).

## Install

Git clone:<BR/>
`git clone https://github.com/liaoherui/AccuSNV.git`<BR/>

### Option-1 (via Bioconda)

`mamba create -n accusnv -c conda-forge -c bioconda accusnv` or <BR/>
`conda create -n accusnv -c conda-forge -c bioconda accusnv`<BR/>

then, `conda activate accusnv`

⚠️ It should be noted that some commands have been replaced if you install AccuSNV using bioconda. (See below)

Command (Not bioconda)    |	Command (bioconda)
------------ | ------------- 
python new_snv_script.py -h | accusnv -h
python accusnv_snakemake.py -h | accusnv_snakemake -h
python accusnv_downstream.py -h | accusnv_downstream -h

The user must only change the initial python line. (See below) 

`accusnv_snakemake -i <input_sample_info_csv> -r <ref_dir> -o <output_dir>`


⚠️ : please ensure that your working directory **does not contain** any files or folders with the same names as those listed below (or if they are present, ensure they are identical files to those in this GitHub repository): <BR/> 

Files: `config.yaml`,`experiment_info.yaml`,`Snakefile` <BR/>
Folders: `scripts` (and all files under `snake_pipeline/scripts` on Github repo)<BR/>


If you install the tool via bioconda, you can test the tool with the command lines below :<BR/>

`cd conda_test_dir`<BR/>
`sh test_run.sh`<BR/>
`sh scripts/dry_run.sh`<BR/>
`sbatch scripts/run_snakemake.slurm`<BR/>

⚠️ : if you still want to run Snakemake directly from the `snake_pipeline` folder within the conda or bioconda environment, please run:

`cd snake_pipeline`<BR/>
`cp Snakefiles_diff_options/Snakefile_conda_env.txt  ./Snakefile`

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


### Option-3 (via pre-built conda env - Linux only)

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

------------------------------------------------------------------------------------
Once you finish the install (via **Option-2** or **Option-3**), you can test the tool with the command lines below :<BR/>

Test snakemake pipeline - Under `snake_pipeline` folder:<BR/>
`sh test_run.sh`<BR/>
`sh scripts/dry_run.sh`<BR/>
`sbatch scripts/run_snakemake.slurm`<BR/>

Test downstream analysis - Under `local_analysis` folder:<BR/>
`sh test_local.sh`<BR/>

<!--- ### Interactive exploration via Spyder:<BR/>

Open `local_analysis/accusnv_downstream.ipynb` in Jupyter.<BR/>

modify the `ref_dir`, `input_mat`, and any optional filtering parameters (e.g. `fn_min_cov`, `fn_min_qual`, `fn_min_freq`, `max_indel`, `min_freq`, `min_med_cov`, `exclude_recomb`), and run the script to inspect results in Spyder.<BR/> -->

### Quick Tests for the downstream analysis module on your own PC:

(Note: if you prefer Linux/HPC, ignore this!)

Refers to [2.2. Local downstream analysis](#custom-anchor)

- [macOS commands (run using command lines)](readme_files/test_run_mac.md)
- [Run accusnv_downstream in IDE (Spyder)](readme_files/test_run_spyder.md)

<!--- - [Windows commands](readme_files/test_run_windows.md)
- [Linux without Slurm](readme_files/test_run_linux_local.md)
- [Linux with Slurm](readme_files/test_run_linux_slurm.md)

These step-by-step guidelines demonstrate running the snakemake pipeline (for Linux system only) along with the downstream analysis commands (`accusnv_snakemake` and `accusnv_downstream`). -->

## Overview

This pipeline and toolkit is used to detect and analyze single nucleotide differences between bacterial isolates from WGS data. 

* Noteable features
	* Avoids false negatives from low coverage and false positives through a deep learning method, while also enabling visualization of raw data.
	* Enables easy evolutionary analysis, including phylogenetic construction, nonsynonmous vs synonymous mutation counting, and parallel evolution, etc.


* Inputs (to Snakemake cluster step): 
	* short-read sequencing data of bacterial isolates
	* an annotated reference genome
* Outputs (of downstream analysis step): 
	* table of high-quality SNVs that differentiate isolates from each other
	* parsimony tree of how the isolates are related to each other
   	* More details can be found in [here](#output)

The pipeline is split into two main components, as described below. 

### 1. Snakemake pipeline

The first portion of AccuSNV aligns raw sequencing data from bacterial isolates to a reference genome, identifies candidate SNV positions, and creates useful data structure for model classification. This step is implemented in a workflow management system called [Snakemake](http://snakemake.readthedocs.io) and is executed on a [SLURM cluster](https://slurm.schedmd.com/documentation.html). More information is available [here](readme_files/readme_snake_main.md).

<!--- #### 1.1 Update - 2025-02-21: A user-friendly Python script is now available to help users run the pipeline more easily. Instructions are provided below:


Make sure to configure your `config.yaml` file and `scripts/run_snakemake.slurm` before starting the steps below.. -->

⚠️: **Please ensure the right permission of the file `slurm_status_script.py`**:

`chmod 777 slurm_status_script.py`<BR/>

⚠️ : If you installed the tool via **Bioconda** and want to run Snakemake directly from the snake_pipeline folder (instead of from an empty directory as described in Option-1 of the installation section), please run the following command before doing anything else:

`cd snake_pipeline`<BR/>
`cp Snakefiles_diff_options/Snakefile_conda_env.txt  ./Snakefile`

✅ Step-1: run the python script: <BR/>

`python accusnv_snakemake.py -i <input_sample_info_csv> -r <ref_dir> -o <output_dir>`

or use (if you install the tool via **bioconda**):

`accusnv_snakemake -i <input_sample_info_csv> -r <ref_dir> -o <output_dir>`

(⚠️ For bioconda installation, we strongly recommend running the command line above in a clean, empty folder. You can use `mkdir work_dir` to build such folder.)

✅ Step-2: check the pipeline using "dry-run"<BR/>

`sh scripts/dry-run.sh`<BR/>

✅ Step-3: submit your slurm job.<BR/>

`sbatch scripts/run_snakemake.slurm`<BR/>

⚠️: If you need to modify any slurm job configuration, you can edit the config.yaml file generated in your output folder: `<output_dir>/conf/config.yaml`

⚠️: Job Interruption Warning: If your job stops before completing, you can check whether any tasks are still pending by running: `sh scripts/dry-run.sh`. If there are unfinished jobs, you can re-submit them using: `sbatch scripts/run_snakemake.slurm`.

⚠️⚠️⚠️: Job interruptions may be caused by issues with specific compute nodes or limitations of the cluster being used (e.g. Timelimit or QOSMaxSubmitJobPerUserLimit). To avoid such interruptions, consider: 

[1]. Use only one partion (modify `- partition="A,B,C"` to `- partition="A"` in `config.yaml`) and set maximum job in `run_snakemake.slurm` or `config.yaml` (see [2] and [3]). 
	
[2]. modifying your `scripts/run_snakemake.slurm` file (e.g. `snakemake --jobs <maxSubmitJob> --max-jobs-per-second 0.5  --max-status-checks-per-second 0.2 xxx`) or 

[3]. modifying your config.yaml file (`<output_dir>/conf/config.yaml`, e.g. change `jobs: 400` to `jobs: <maxSubmitJob>`, and add `max-jobs-per-second: 0.5`, and `max-status-checks-per-second: 0.2`). 
	
If still interruption, please reach out via herui728@mit.edu.

----------------------------------
Notes for Step-1: 

One example (This example uses commands like `python accusnv_snakemake.py xxx`. If you installed the tool via Bioconda, please replace those with: `accusnv_snakemake xxx`)  with test data can be found in `snake_pipeline/test_run.sh`

If you cloned the repository (e.g. a new download) and have already downloaded the pre-built Conda environment (e.g., /path/snake_pipeline/accusnv_sub), there's no need to download it again. Just try:

`python accusnv_snakemake.py -i <input_sample_info_csv> -c /path/snake_pipeline/accusnv_sub -r <ref_dir> -o <output_dir>`

One example file for `<input_sample_info_csv>` can be found at `snake_pipeline/samples.csv`. More information about the input csv and the reference genome file can be found at [here](https://github.com/liaoherui/AccuSNV/blob/main/readme_files/readme_snake_run.md#modify-files-for-your-project)

----------------------------------


### 2.1. Local python analysis

⚠️: This step has been incorporated into the Snakemake pipeline and will be executed automatically by default. However, you can still use this local Python script to rerun the analysis with different parameters if needed.

`python new_snv_script.py -i <input_mutation_table> -c <input_raw_coverage_matrix> -r <ref_dir> -o <output_dir>`

or use (if you install the tool via bioconda):

`accusnv -i <input_mutation_table> -c <input_raw_coverage_matrix> -r <ref_dir> -o <output_dir>`

One example with test data can be found in `local_analysis/test_local.sh`

The second portion of AccuSNV filters candidate SNVs based on data arrays generated in the first portion and generates a high-quality SNV table and a parsimony tree. This step utilizes deep learning and is implemented with a custom Python script. More information can be found [here](readme_files/readme_local_main.md).

<a id="custom-anchor"></a>
### 2.2. Local downstream analysis

Based on the identified SNVs and **the output final mutation table (in .npz format, e.g. candidate_mutation_table_final.npz under the folder 3-AccuSNV) from Snakemake pipeline**, AccuSNV offers a set of downstream analysis modules (e.g. dN/dS calculation). You can run these modules using the command below.

(Note, .npz file under the folder **2-Case** can by only used as input to Local python analysis - `new_snv_script.py`)

`python accusnv_downstream.py -i  test_data/candidate_mutation_table_final.npz -r ../snake_pipeline/reference_genomes/Cae_ref -o cae_accusnv_ds_pe`

or use (if you install the tool via bioconda):

`accusnv_downstream -i  test_data/candidate_mutation_table_final.npz -r ../snake_pipeline/reference_genomes/Cae_ref -o cae_accusnv_ds_pe`


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


## Output


<img src="https://github.com/liaoherui/AccuSNV/blob/main/readme_files/output_downstream_new_single.jpg" width = "700" height = "650" >  

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

A demo output HTML report of AccuSNV can be found at https://heruiliao.github.io/

 ## -Contact-
  
 If you have any questions, please post an issue on GitHub or email us: herui728@mit.edu

 ## References:

how to cite this tool:
```
Liao, H., Conwill, A., & Light-Maka, Ian., et al. High-accuracy SNV calling for bacterial isolates using deep learning with AccuSNV, BioRxiv, 2025; https://www.biorxiv.org/content/10.1101/2025.09.26.678787v2
```


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


