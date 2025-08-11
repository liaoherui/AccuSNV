# Test AccuSNV via Spyder

This page shows how to verify the Bioconda installation on macOS using the bundled test data.

## 1. Install with Bioconda

```bash
conda create -n accusnv_env -c conda-forge -c bioconda accusnv
conda activate accusnv_env
conda install spyder-kernels=3.0
```
If the command line doesn't work, you may consider:

```bash
CONDA_SUBDIR=osx-64 conda create -n accusnv_env -c conda-forge -c bioconda accusnv_env
conda activate accusnv_env
conda install spyder-kernels=3.0
```

## 2. Clone the repository

```bash
git clone https://github.com/liaoherui/AccuSNV.git
cd AccuSNV
```


## 3. Configure the Spyder environment

Ensure that the spyder-kernels package is installed in your Conda environment. If not, run:

```bash
conda activate accusnv_env
conda install spyder-kernels=3.0
```

Then:

1. Open Spyder.

2. Set the Python interpreter to your pre-built accusnv conda environment. (See screenshot below — in this example, the Conda environment is named widevariant.)

<img src="https://github.com/liaoherui/AccuSNV/blob/main/readme_files/Spyder_env.jpg" width = "900" height = "850" >

## 4. Run the test script

1. Close Spyder (to ensure the interpreter setting is saved).
2. Reopen Spyder.
3. Open the file `accusnv_downstream_IDE.py` with Spyder.
4. Run the script

## 5. Expected output

After running, you should see an output directory: `accusnv_downstream_out/`

The directory should contain the generated output files.
