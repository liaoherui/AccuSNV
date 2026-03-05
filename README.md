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
`cd AccuSNV/snake_pipeline`<BR/>
`chmod 777 slurm_status_script.py`<BR/>

Install via bioconda:<BR/>
`mamba create -n accusnv -c conda-forge -c bioconda accusnv` or <BR/>
`conda create -n accusnv -c conda-forge -c bioconda accusnv`<BR/>

then, `conda activate accusnv`




