# Input files of AccuSNV

## 1. Input CSV file of AccuSNV

`<samples.csv>` tells AccuSNV what to run, where your reads are, and how samples are grouped.

### Example format

```
Path,Sample,FileName,Reference,Group,Outgroup,Type
/user/project/reads/,strain1,strain1,Cae_ref,group_pe_test,0,PE
/user/project/reads/,strain2,strain2,Cae_ref,group_pe_test,0,PE
/user/project/reads/,strain3,strain3,Cae_ref,group_pe_test,0,PE
/user/project/reads/,strain4,strain4,Cae_ref,group_pe_test,0,PE
```

### Required header (exact order)

```
Path,Sample,FileName,Reference,Group,Outgroup,Type
```

### Column meaning

- `Path`: Folder containing raw read files for this sample. Symlinked FASTQ files are supported; matching uses the symlink filename in this folder.
- `Sample`: Sample ID (used in output filenames and plots). A sample can be listed on several rows to put it in more than one group, e.g. as the outgroup of each of them, as long as those rows give the same `Path`, `FileName` and `Type`. Reads are trimmed once and mapped once per reference.
- `FileName`: Read file prefix (without `_1/_2`, `_R1/_R2`, and without extension). Example: if files are `strainA_1.fastq.gz` and `strainA_2.fastq.gz`, use `strainA`. The prefix is matched exactly, so `strainA_1` will not also match `strainA_10`.
- `Reference`: Reference genome folder name (under your reference genome directory). Avoid names that start with `ref_` or contain `_ref_`, because AccuSNV uses `_ref_` as an internal filename delimiter; use names such as `clade1_ref` instead.
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
Then, the `input_sample.csv` should look like this (if no outgroup sample):

Each reference folder must contain a file named exactly `genome.fasta`. AccuSNV can generate BWA and samtools indexes during the Snakemake run (`genome.fasta.bwt`, `.amb`, `.ann`, `.pac`, `.sa`, and `.fai`); if the reference directory is not writable, pre-create them with `bwa index genome.fasta` and `samtools faidx genome.fasta`.

```
Path,Sample,FileName,Reference,Group,Outgroup,Type
/my_project/raw_reads/,strain1,strain1,Cae_ref,group_pe_test,0,PE
/my_project/raw_reads/,strain2,strain2,Cae_ref,group_pe_test,0,PE
/my_project/raw_reads/,strain3,strain3,Cae_ref,group_pe_test,0,PE
/my_project/raw_reads/,strain4,strain4,Cae_ref,group_pe_test,0,PE
```

Finally, follow the command lines in the **Usage** section to run the tool.

For example, based on the case above, you can start with the following command (on your laptop):

`accusnv -m local -i input_sample.csv -r reference_genomes -o my_test`
