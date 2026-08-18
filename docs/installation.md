# Installation

## Conda installation

The simplest way to install AccuSNV is via bioconda, in a new environment with Python 3.12:

```
conda create -n accusnv python=3.12
conda activate accusnv
conda install -c bioconda accusnv
```

(you can use mamba, micromamba, or miniconda - micromamba is a very fast conda alternative)

If you run that, you should be good to go. Try running `accusnv -h` to see the help menu, and continue to [quick start](quickstart.md). Everything below is additional detail.

## Pip installation

If you install the non-Python dependencies for AccuSNV separately on your system (see below), you can install the AccuSNV python package via pip as well:

```
pip install accusnv
```

## Dependencies

These are not Python packages, so install them separately and make sure they are on your `PATH` if you have not done the conda install (`conda install accusnv` installs these automatically).

AccuSNV checks for all of these before it starts and names any that are missing, so you find out immediately if a dependency is unavailable.

| Tool                      | Used for                                                                               |
| ------------------------- | -------------------------------------------------------------------------------------- |
| `bwa` (or `bowtie2`)      | Aligning reads to the reference                                                        |
| `samclip`                 | Dropping soft-clipped `bwa` alignments (not needed with `bowtie2` or `--skip_samclip`) |
| `samtools`                | Sorting, duplicate marking, pileups                                                    |
| `bcftools`                | Variant calling and call quality scores                                                |
| `tabix`                   | Indexing the VCF files                                                                 |
| `sickle`                  | Quality trimming                                                                       |
| `cutadapt`                | Adapter trimming                                                                       |
| `dnapars` (from `phylip`) | Building the maximum-parsimony tree                                                    |

You can also install these dependencies by hand with conda:

```bash
conda install -c conda-forge -c bioconda bwa samclip samtools bcftools tabix sickle cutadapt phylip
```

`dnapars` is only needed for the parsimony tree, so it is not checked when you pass `--skip_trees` or `--skip_all_downstream`, or when you set `use_nj_tree: true` in `pipeline.yaml` to build a neighbor-joining tree with Biopython instead.

`samclip` is only used on `bwa` alignments, so it is not checked when the aligner is `bowtie2` or
when you pass `--skip_samclip`.

## Using a different environment

If the binaries are in one conda environment and you run `accusnv` from another, pass the
activation command with `-e`:

```bash
accusnv -e 'conda activate accusnv' -i samples.csv -r reference_genomes -o out
```

Every workflow rule will then run that command first. AccuSNV skips its own dependency check in this case, since it cannot see inside the other environment. So in this case, if something is missing, you will find out when the rule that needs it fails.

This is also the answer on clusters where compute nodes do not inherit the login node's
environment. Passing `-e 'conda activate accusnv'` is usually all that is needed.

## Troubleshooting

Below is a list of conda issues you may run into during installation.

1. **Snakemake 8 requires Python >3.11.** AccuSNV requires Snakemake version >8, which in turn requires a Python version higher than 3.11. Older Python versions simply will not work.

2. **The Cutadapt package in bioconda requires Python <3.13.** If you are running into an issue specifically mentioning an inability to resolve the cutadapt dependency, this is most likely due to an outdated cutadapt in bioconda (as of Aug 2026). You can try creating a fresh conda environment without installing cutadapt, and instead install it separately with pip inside the conda environment: `pip install cutadapt`.
