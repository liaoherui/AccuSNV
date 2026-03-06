# Input files of AccuSNV

## 1. Input CSV file of AccuSNV

`<samples.csv>` tells AccuSNV what to run, where your reads are, and how samples are grouped.

### Required header (exact order)

```
Path,Sample,FileName,Reference,Group,Outgroup,Type
````

### Column meaning

- `Path`: Folder containing raw read files for this sample.
- `Sample`: Unique sample ID (used in output filenames and plots).
- `FileName`: Read file prefix (without `_1/_2` and without extension). Example: if files are `strainA_1.fastq.gz` and `strainA_2.fastq.gz`, use `strainA`.
- `Reference`: Reference genome folder name (under your reference genome directory).
- `Group`: Samples with the same Group are analyzed together in one AccuSNV group output.
- `Outgroup`: `0` = ingroup sample, `1` = outgroup sample.
- `Type`: Sequencing type: `PE` or `SE`.

## 2. Example 

Suppose the input directory is like this:
```
/my_project/
├── raw_reads/
│   ├── strain1_1.fastq.gz
│   ├── strain1_2.fastq.gz
│   ├── strain2_1.fastq.gz
│   ├── strain2_2.fastq.gz
│   ├── strain3_1.fastq.gz
│   ├── strain3_2.fastq.gz
│   ├── strain4_1.fastq.gz
│   └── strain4_2.fastq.gz
├── reference_genomes/
│   └── Cae_ref/
│       ├── genome.fasta
│       └── genome.gff
└── input_sample.csv
```
Then, your `input_sample.csv` should look like this (if no outgroup sample):

```
Path,Sample,FileName,Reference,Group,Outgroup,Type
/my_project/raw_reads/,strain1,strain1,Cae_ref,group_pe_test,0,PE
/my_project/raw_reads/,strain2,strain2,Cae_ref,group_pe_test,0,PE
/my_project/raw_reads/,strain3,strain3,Cae_ref,group_pe_test,0,PE
/my_project/raw_reads/,strain4,strain4,Cae_ref,group_pe_test,0,PE
```

Finally, follow the command lines in the **Usage** section to run the tool.

For example, based on the case above, you can start with the following command (on your laptop):

`python accusnv_snakemake.py -f 1 -i input_sample.csv -r reference_genomes -o my_test`
