# Test AccuSNV on macOS

This page shows how to verify the Bioconda installation on macOS using the bundled test data.

## 1. Install with Bioconda

```bash
conda create -n accusnv_env -c conda-forge -c bioconda accusnv
conda activate accusnv_env
```
If the command line doesn't work, you may consider:

```bash
CONDA_SUBDIR=osx-64 conda create -n accusnv_env -c conda-forge -c bioconda accusnv_env
conda activate accusnv_env
```

## 2. Clone the repository

```bash
git clone https://github.com/liaoherui/AccuSNV.git
cd AccuSNV
```


## 3. Run the downstream analysis

```bash
accusnv_downstream -i local_analysis/test_data/candidate_mutation_table_final.npz \
  -r snake_pipeline/reference_genomes/Cae_ref -o mac_downstream_test
```

## 5. Expected output

The directories `mac_downstream_test/` should contain output files .