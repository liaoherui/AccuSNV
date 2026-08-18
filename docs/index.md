# AccuSNV

AccuSNV is a computational pipeline that identifies **single nucleotide variants (SNVs)** in short-read whole genome sequencing data **between genomes** in a group of bacterial isolates.

AccuSNV takes FASTQ files from *three or more* bacterial isolates and a reference genome, and returns the genomic SNVs at which those isolates differ from each other, instead of just differences from references. 

The pipeline then builds a parsimony **phylogenetic tree** from these SNVs, calculates the mutational distance to the most recent common ancestor (**dMRCA**), reports the impact on protein sequence for each SNV, and calculates ***dN/dS*** ratios.

AccuSNV classifies candidate SNV sites as true or false SNVs using a *convolutional neural network* (CNN) trained on real sequencing data from bacterial isolates to distinguish true variation from mapping errors, indels, or sequencing noise. It also runs a set of rule-based quality checks on each site, inherited from the [WideVariant](https://github.com/liebermanlab/WideVariant) pipeline. Both the CNN and simple rules are reported for every position, so you can always see why a SNV was kept or dropped.

## Where to start

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Getting started 

[Install AccuSNV](installation.md), then run the [two-minute test dataset](quickstart.md) that ships with the repository. The test dataset runs within a few minutes, to make sure the pipeline is working on your system.
:::

:::{grid-item-card} Setting up your own data
[Input files](inputs.md) covers the sample sheet, how AccuSNV finds your FASTQ files, and how to lay out reference genomes. [Running AccuSNV](running.md) covers local runs, cluster runs, and re-running just the evolutionary analyses.
:::

:::{grid-item-card} Reading your results
Start with [the final SNV table](snv_table.md), which documents every column. [Output files](outputs.md) is the map of everything else the run wrote.
:::

:::{grid-item-card} Perform evolutionary analyses
[How SNVs are called](filters.md) explains the neural network, the nine quality checks, and how
the two verdicts combine. [Recombination](recombination.md) and [*dN/dS*](dnds.md) explain the two analyses that people most often need to interpret carefully.
:::

::::

## The basics

Set up the AccuSNV environment:

```bash
conda create -n accusnv python=3.12
conda activate accusnv
conda install -c conda-forge -c bioconda accusnv
```

Run it in local mode on the test data:

```
accusnv -m local -i Test_data/samples_cae_test_pe.csv -r Test_data/reference_genomes -o cae_accusnv_output
```

Run it on an **HPC system** with a SLURM scheduler:

```
accusnv -m slurm -sp <partition> -i Test_data/samples_cae_test_pe.csv -r Test_data/reference_genomes -o cae_pe_test_snakemake
```

## Output files

AccuSNV creates a number of potentially useful intermediate and output files. You can read about all of them under Output files. However, the most important ones are:

| File                                                               | Description                                                                                 |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| `group_<group>_snv_table_final.tsv`                                | SNV calls, one row per position, with gene and protein annotation. This is the main result. |
| `group_<group>_snv_dashboard.html`                                 | An interactive web page for examining SNVs and showing them on an interactive tree.         |
| `group_<group>_snv_table_unfiltered.tsv`                           | A table that also includes sites that were considered, but rejected, as potential SNVs.     |
| `group_<group>_snv_tree_final.nwk.tree`                            | A maximum-parsimony tree of the isolates in Newick format.                                  |
| `3-Analysis/group_<group>/dNdS_out/`                               | Genome-wide and per-gene *dN/dS* calculations.                                              |
| `3-Analysis/group_<group>/phylogeny/snv_table_tree_distances.tsv`  | Per-sample distance to the inferred common ancestor (dMRCA).                                |

## Citation

> Liao, Herui, Arolyn Conwill, Ian Light-Maka, Martin Fenk, Alyssa H. Mitchell, Evan B. Qu, Paul Torrillo, Jacob S. Baker, Felix M. Key, and Tami D. Lieberman.
> "[High-accuracy SNV calling for bacterial isolates using deep learning with AccuSNV](https://genome.cshlp.org/content/early/2026/07/02/gr281341125)."
> *Genome Research*, June 2026, Vol. 36, No. 6.
> [doi:10.1101/gr.281341.125](https://doi.org/10.1101/gr.281341.125)

If you have any questions and bug reports, please report them on the [GitHub issue tracker](https://github.com/liaoherui/AccuSNV/issues), or email Herui Liao (herui728@mit.edu) or the maintainer Alex Crits-Christoph (crits@mit.edu).

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: Setting up a run
:hidden:

inputs
parameters
running
```

```{toctree}
:maxdepth: 2
:caption: How AccuSNV works
:hidden:

pipeline
filters
```

```{toctree}
:maxdepth: 2
:caption: Interpreting results
:hidden:

outputs
snv_table
phylogeny
dnds
recombination
```

```{toctree}
:maxdepth: 2
:caption: Help
:hidden:

troubleshooting
```
