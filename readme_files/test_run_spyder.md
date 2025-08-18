# Test AccuSNV via Spyder

This page shows how to test AccuSNV downstream analysis module (see 2.2 in main README) via Spyder on macOS.


## 1. Download the test folder for Spyder

Download link is here: [spyder_test.zip](https://github.com/liaoherui/AccuSNV/raw/refs/heads/main/spyder_test.zip)

Unzip the files, and open the file `accusnv_downstream_IDE.py` with Spyder.



## 2. Add the bioconda channel and install AccuSNV in Spyder

--- **Note**: If you have already installed the accusnv conda environment with Spyder (if Spyder is not installed, please run `conda install spyder`), you can skip this step and go directly to “3. Configure the Spyder environment.” ---

Check the configuration of the current environment:

```bash
conda config --show-sources
```
<img src="https://github.com/liaoherui/AccuSNV/blob/main/readme_files/spyder_step1.jpg" width = "900" height = "800" >

Add bioconda to Spyder's configuration file (replace the directory with the path to your own .condarc file obtained by `conda config --show-sources`):

```bash
conda config --add channels bioconda --file /path/tp/Library/spyder-6/.condarc
```

Verify whether the channel was successfully added

```bash
conda config --show channels
```

<img src="https://github.com/liaoherui/AccuSNV/blob/main/readme_files/spyder_test2.jpg" width = "900" height = "800" >

Install AccuSNV via bioconda:

```bash
conda create -n accusnv_spyder python=3.9 spyder accusnv -c bioconda -c conda-forge -y
```

<img src="https://github.com/liaoherui/AccuSNV/blob/main/readme_files/spyder_test4.jpg" width = "900" height = "800" >

## 3. Configure the Spyder environment


 Set the Python interpreter to your pre-built accusnv_spyder conda environment. (See screenshot below — in this example)

<img src="https://github.com/liaoherui/AccuSNV/blob/main/readme_files/spyder_test3.jpg" width = "900" height = "800" >

## 4. Run the test script
1. Close Spyder (to ensure the interpreter setting is saved).
2. Reopen Spyder.
   
Option-1:
-> Open the file `accusnv_downstream_IDE.py` with Spyder.
-> Run the script

Option-2:
-> In Spyder, run: 

```
from accusnv import accusnv_downstream_func
# ---------------------------------------------------------------------------
# Variables for interactive use
# ---------------------------------------------------------------------------
# When running this script in an interactive environment (e.g. Spyder or
# Jupyter), set ``ref_dir`` to the directory containing ``genome.fasta``
# and ``input_mat`` to the path of your candidate mutation table (NPZ file).
# ``output_dir`` determines where results will be written.

# The parameters below control manual filtering behaviour:
#   - ``fn_min_cov``: minimum forward+reverse reads required per sample
#     (default 1).
#   - ``fn_min_qual``: minimum quality score per sample (default 30).
#   - ``fn_min_freq``: minimum major allele frequency per sample
#     (default 0.75).
#   - ``max_indel``: maximum fraction of reads supporting
#     an indel in a sample (turned off by default; reference ``-d 0.33``).
#   - ``min_freq``: across samples, maximum fraction of
#     undefined bases per position (turned off by default; reference
#     ``-g 0.25``).
#   - ``min_med_cov``: minimum median coverage across samples
#     per position (turned off by default; reference ``-m 5``).
#   - ``eb``: set to ``1`` to exclude SNVs from potential
#     recombination events (default includes them).
#   - ``exclude_sample_ids``: comma-separated sample IDs to remove from analysis.
#   - ``exclude_position_ids``: comma-separated genomic positions to exclude.


accusnv_downstream_func.analyze(ref_dir="reference_genomes/Cae_ref",
                                input_mat="test_data/candidate_mutation_table_final.npz",
                                output_dir = "accusnv_downstream_out",
                                fn_min_cov = 1,
                                fn_min_qual = 30,
                                fn_min_freq = 0.75,
                                max_indel = None,
                                min_freq = None,
                                min_med_cov = None,
                                eb = 0,
                                exclude_sample_ids = "",
                                exclude_position_ids = "")
```


   

## 5. Expected output

After running, you should see an output directory: `accusnv_downstream_out/`

The directory should contain the generated output files.
