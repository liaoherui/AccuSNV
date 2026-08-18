# Pipeline overview

The AccuSNV pipeline is a Snakemake workflow with the following overarching steps:

```{figure} _static/figures/pipeline_overview.svg
:alt: The AccuSNV pipeline
:width: 100%

The AccuSNV pipeline stages.
```

## Stage 1: mapping (`1-Mapping/`)

First, Reads are trimmed using `cutadapt`, which removes the 3' adapter (`adapter_sequence`, Nextera by default), then `sickle` quality-trims at Q20 and drops anything shorter than 50 bp. 

You can then choose your aligner to be either `bwa` or `bowtie2` (bwa by default). The aligner then maps the trimmed reads to the reference for that sample's group, building the index on the first run if it is not already there. Bwa output is by default piped through `samclip --max 0`, which throws away any read with a soft-clipped end. Pass `--skip_samclip` to not do this. `bowtie2` does not need this, so `samclip` is not run for it.

The alignments then are proceed with `samtools`, which sorts by name, fixes mate information, sorts by coordinate, and removes duplicates with `markdup -r`. Reads within `markdup_optical_distance` pixels of each other count as optical duplicates. The SAM and all the intermediate BAMs are deleted once the final sorted BAM exists.

Next, `samtools mpileup` produces a full pileup, which AccuSNV converts into a per-position matrix holding, for every base of the genome, the number of forward and reverse reads supporting each of A, T, C and G, their average base quality, mapping quality and distance from the read end, and the number of reads supporting an insertion or a deletion nearby. 

Next, `bcftools mpileup | bcftools call` produces a VCF for the whole genome. The FQ score at every position becomes that sample's call-quality vector. Positions where a single-base substitution was called with FQ below `max_fq` (default -30) become that sample's list of *candidate variant positions*.



## Stage 2: SNV filtering (`2-SNV-filtering/`)

In the second stage, every ingroup sample's list of candidate positions is merged into one list for the entire group.

Then every sample is genotyped at every one of the candidate positions nominated by any sample. The candidate mutation table holds, per sample and per position, the eight read counts (A, T, C, G on the forward strand, then the same four on the reverse), the call quality, and the indel counts. 

Next, both the CNN and WideVariant rule-based filters are run for SNV calling. Every candidate position is scored by the CNN, and through the WideVariant filters, and given a final label, as described in [How SNVs are called](filters.md). 

Finally, each passing SNV is annotated using the reference annotations: which gene it is in, where in that gene it is, and what the codon and amino acid changes are. The final SNV calls with annotations are provided in `snv_table_final.tsv`.

## Stage 3: evolutionary analysis (`3-Analysis/`)

The evolutionary analyses run by default are:

1. dN/dS is computed genome-wide and per gene ([dN/dS](dnds.md)).

2. `dnapars` builds a maximum-parsimony tree using passing SNVs ([Phylogeny and dMRCA](phylogeny.md))

3. dMRCA of each sample to the inferred ancestor is calculated, along with the median, minimum and maximum distance to the ancestor across the ingroup.

4. An interactive HTML report is generated for SNV curation and tree viewing.

## The neural network

```{figure} _static/figures/cnn_architecture.jpg
:alt: The AccuSNV convolutional neural network architecture
:width: 100%

The CNN architecture overview
```

For a detailed description of the CNN, please read the AccuSNV paper: [High-accuracy SNV calling for bacterial isolates using deep learning with AccuSNV, Genome Research, 2026](https://genome.cshlp.org/content/early/2026/07/02/gr281341125).

## All Snakemake rules

The Snakemake rules in order are:

| Rule name                  | Runs per... | It runs:                                                      |
| -------------------------- | ----------- | ------------------------------------------------------------- |
| `cutadapt`                 | sample      | `cutadapt`                                                    |
| `sickle`                   | sample      | `sickle`                                                      |
| `create_mapping_index`     | reference   | `bwa index` or `bowtie2-build`                                |
| `mapping`                  | sample      | `bwa mem` (piped through `samclip`) or `bowtie2`              |
| `sam2bam`                  | sample      | `samtools` sort, fixmate, markdup, index                      |
| `samtools_idx`             | reference   | `samtools faidx`                                              |
| `mpileup2vcf`              | sample      | `samtools mpileup`, `bcftools mpileup`/`call`/`view`, `tabix` |
| `vcf2quals`                | sample      | `accusnv.preprocessing.vcf2quals_snakemake`                   |
| `variants2positions`       | sample      | `accusnv.preprocessing.variants2positions`                    |
| `pileup2diversity`         | sample      | `accusnv.preprocessing.pileup2diversity`                      |
| `combine_positions`        | group       | merges the per-sample candidate lists                         |
| `candidate_mutation_table` | group       | `accusnv.preprocessing.build_candidate_mutation_table`        |
| `calling_accusnv`          | group       | `accusnv.accusnv.accusnv` (the CNN and the checks)            |
| `annotate_snvs`            | group       | `accusnv.downstream.annotate`                                 |
| `dnds`                     | group       | `accusnv.downstream.dnds`                                     |
| `build_tree`               | group       | `accusnv.downstream.tree_building`                            |
| `report_html`              | group       | `accusnv.downstream.generate_dashboard`                       |
