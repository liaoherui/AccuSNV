# Troubleshooting

Please read your `accusnv.log`, `accusnv.full.log`, and `accusnv.snakemake.log` in your output directory to fully investigate a particular bug.

If you have any issues with AccuSNV, open an issue at [github.com/liaoherui/AccuSNV/issues](https://github.com/liaoherui/AccuSNV/issues), and we will respond quickly! Include `accusnv.log`, `accusnv.full.log`, your `samples.csv` (with paths redacted if you need to), and the `configs/pipeline.yaml` from the run.



Here are some possible issues you could run into, and some potential fixes.

**"Could not find dependencies: dnapars, sickle"**

This means that some of the external dependencies are not available on your system PATH. You might need to refer to a specific conda environment with `-e 'conda activate <env>'`, or double check your installation. When running in the accusnv environment, you should be able to run `samtools`, `bcftools`, `dnapars`, `sickle`, `cutadapt`, and `bwa` or `bowtie2`.

**"could not find both a forward and reverse read for sample X in directory Y"**
From the details in the sample table passed to AccuSNV, it could not find a pair of FASTQ files in the `Path` column for a sample, for files starting with `FileName`. Check the extension is either `.fastq.gz`, `.fq.gz`, `.fastq`, or `.fq`, and that the R1 and R2 read files are distinguishable by a `1` or `2` joined with `_`, `.`, or `-`. Finally, a sample marked `PE` with only one file will also give this message if it should be `SE` (single end) in the sample table.

**"No reference FASTA file found in: ..."**
In your sample table, the `Reference` column is a subfolder that does not contain a FASTA. AccuSNV looks for a FASTA that ends in `.fasta`, `.fa`, or `.fna` in that folder. "Multiple reference FASTA files found" means it found several, while there should just be one per subfolder in the reference genomes directory.

**Permission errors while indexing the reference**
The first run creates `bwa` or `bowtie2` and `samtools` indexes next to the FASTA in the reference directory. If the reference directory is read-only, build them yourself first with `bwa index genome.fasta` and `samtools faidx genome.fasta`.

**Slurm jobs die with out-of-memory or time-limit errors**
If this happens, your jobs are running out of memory even on the maximum memory tier that AccuSNV allows for a job. They should be resubmitted with custom memory and runtime parameters in your `config.yaml`. See [the resources section on the Parameters page](parameters.md).

**There were too few or missing SNVs in your output**

Read the <u>filter</u> lines in `accusnv.log`, which tell you which filters and checks removed SNVs. Also check `snv_table_unfiltered.tsv`, which should contain detailed information on eliminated potential SNVs.

Finally, have a look to see if the missing SNVs appear in the interactive dashboard HTML file in your output directory, which you can use to investigate why they may have been eliminated.

Missing SNVs that should actually pass filters can be re-included with the `--include_positions` parameter.

**There were more SNVs than expected in your output**

Have a look at `2-SNV-filtering/group_<group>/snvs_per_sample.tsv`. If there is one sample with an order of magnitude more than the rest, it may be contaminated or a highly divergent strain. You may wish to re-run without this sample.

**Every SNV is intergenic**

If this happens, then the GFF annotation file was likely not found or could not be parsed. Check that a `.gff` is included in the same reference subdirectory as your FASTA, and that `accusnv.log` reports loading genes:

```text
[calling] Loaded 501 genes from the reference annotations across 1 contig(s)
```

If that line has 0 genes, CDS features are not in the GFF.
