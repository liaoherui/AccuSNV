# Running AccuSNV

A basic run of AccuSNV is [pretty simple](quickstart.md), but there are many options, parameters, and advanced scenarios that are possible, as covered on this page.

## The command-line options

```text
usage: accusnv [-h] [-i CSV] [-r DIR] [-o DIR] [-c FILE] [-p FILE]
        [-m {dryrun,slurm,local}] [-j N] [-sp PARTITIONS] 
        [-e CMD] [...]
```

### Inputs and outputs

| Option                          | Description                                                                                 |
| ------------------------------- | ------------------------------------------------------------------------------------------- |
| `-i`, `--input_sample_info CSV` | The [sample sheet](inputs.md#the-sample-sheet). Required.                                   |
| `-r`, `--ref_dir DIR`           | The [reference genome directory](inputs.md#the-reference-genome-directory). Required.       |
| `-o`, `--output_dir DIR`        | Where to write everything. Default value is `accusnv_output`. Created if it does not exist. |

### Run modes

| Option                              | Description                                                                                                                                                                                                                                                                                                                                                       |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-m`, `--mode {local,slurm,dryrun}` | Whether to run a job in local mode or slurm mode (for an HPC slurm system). Default is `local`.                                                                                                                                                                                                                                                                   |
| `-j`, `--cores N`                   | Cores each `cutadapt`, samtools, and aligner job runs with, in every mode. In a local run it is also the total core budget. Default: 4.                                                                                                                                                                                                                           |
| `-sp`, `--partition PARTITIONS`     | Slurm partition(s) to submit to. Comma-separated for more than one, for example `-sp short,long`. By default, `sbatch` uses your account's cluster default. Ignored outside slurm mode.                                                                                                                                                                           |
| `-e`, `--env CMD`                   | A command run to activate an environment at the start of every job, e.g. `'conda activate accusnv'`. Default is to inherit the environment `accusnv` itself was launched from. This isn't necessary in most setups. Better practice is to activate the environment first (`conda activate accusnv`) and then run accusnv in that environment without this option. |

### Config files

| Option                       | Description                                                                                                                                  |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `-p`, `--pipeline_file FILE` | Pipeline parameters: including the aligner, cores, adapter sequence, and all filter cutoffs. See [Parameters](parameters.md).                |
| `-c`, `--config_file FILE`   | Snakemake execution settings: executor, job limit, per-rule resources. This file is useful for specifying custom resource requests per rule. |

If you do not pass these two files, the default files will be used and created in `<output_dir>/configs/` on your run. So the way to change anything is to run once (e.g. with `-mode dryrun`), copy the generated file, edit it, and pass it back with these parameters for future runs. When you do pass these files via these arguments, they are copied into `<output_dir>/configs/` for your given run. Alternatively, you can edit the files in place in an existing run's output directory.

Any parameters you pass via the command line arguments (e.g., `-sp` for partitions) overwrite the copies of these in your `<output_dir>/configs/` folder. **Command line arguments always take precedence over any existing parameters in your config files for a given run**.

### Turning particular stages on and off

There are several downstream analyses in AccuSNV that you may or may not want to run. You can specify these with:

| Option                  | Description                                                                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `--skip_all_downstream` | Stop after the annotated SNV tables. Avoids creating dN/dS, tree, and report.                                                            |
| `--skip_report`         | Skip the HTML report.                                                                                                                    |
| `--skip_dnds`           | Skip the dN/dS calculation.                                                                                                              |
| `--skip_trees`          | Skip the parsimony tree. The dMRCA distance tables are still written.                                                                    |
| `--skip_samclip`        | Stop piping `bwa` alignments through `samclip`, which by default drops any read with a soft-clipped end. Not relevant with `bowtie2`.    |
| `--skip_recombination`  | Do not flag recombinant SNVs, which means they will now get counted in dN/dS, the tree and dMRCA. See [Recombination](recombination.md). |
| `--build_snv_trees`     | Write one NEXUS tree per SNV, tips colored by base call, for viewing in FigTree. Off by default.                                        |
| `--downstream_only`     | Re-run only the evolutionary analyses on an output directory that already has SNV calls in it.                                           |

Each of these also exists as a `true`/`false` key in `pipeline.yaml`. Setting the flag on the
command line sets the corresponding key to `true` for that run in the yaml file.

### Manually excluding and including particular sites

You can include or exclude specific genomic positions:

| Option                | Description                                                                                                     |
| --------------------- | --------------------------------------------------------------------------------------------------------------- |
| `--exclude_positions` | File of genome_pos values, one per line, to exclude from the SNV set, regardless of whatever the model decided. |
| `--include_positions` | File of genome_pos values, one per line, to keep in the SNV set regardless of whatever the model decided.       |

This would typically be done after you've run AccuSNV once already, and then manually examined, curated and filtered your SNVs. You can take the `genome_pos` column from the final SNV table output file for each site that you wish to include or exclude, and add them to the files.

## Running in local mode

```bash
accusnv -m local -j 8 -i samples.csv -r reference_genomes -o my_output
```

Everything then runs on the current machine, with Snakemake using up to `-j` cores. This will work fine for small runs, such as <10 isolates. For dozens of isolates, runs will begin to take up to several hours.

## Running via Slurm and Slurm resource requesting

```bash
accusnv -m slurm -sp short -i samples.csv -r reference_genomes -o my_output
```

When running via Slurm, each rule is its own job. AccuSNV uses Snakemake to request memory and a time limit per job, using a table of tiers with expected run times and memory usage per job. This is done to reduce the requested memory needed by your jobs, reducing their load on your HPC cluster.

If a job runs out of memory or hits its time limit, Snakemake resubmits it at the next tier up, up to three times. That is the `retries: 3` setting in `config.yaml`; set it to `0` to switch the behavior off. The maximum number of jobs that will be run at the same time is the `jobs:` setting (100 by default). The tiers are in `src/accusnv/workflow/resources.py`.

To manually change the resource requests for a particular step of the pipeline, edit your `<output_dir>/configs/config.yaml` and pass it back with `-c`. For example:

```
set-resources:
  candidate_mutation_table:
    mem_mb: 200000
    runtime: 480
```

This example sets the maximum memory for the candidate SNV table building step to 200 GB, higher than the defaults.

## Snakemake dry run

```bash
accusnv -m dryrun -i samples.csv -r reference_genomes -o my_output
```

The dry run mode of Snakemake builds the job graph and prints what would run, without actually running anything. You can do this to quickly check that your sample sheet parses, your FASTQ files are all findable, and the dependencies are available before submitting a cluster job.

## Re-running only the downstream analyses

If you want to try different downstream annotation and analyses, or add an analysis an earlier run skipped, you do not have to re-run everything: just run with `--downstream_only`, pointing it to your previous output folder:

```bash
accusnv --downstream_only -i samples.csv -r reference_genomes -o my_output
```

AccuSNV then reads the SNV calls already in `2-SNV-filtering/group_<group>/`, and re-runs annotation, dN/dS, tree building and the report. You can modify any of the settings these stages use in `pipeline.yaml`, and then re-run with `--downstream_only` to explore different analysis choices, including `--exclude_positions` or `--include_positions`.

## Reading output logs for debugging

AccuSNV extensively logs all steps. The primary and simple log file is `accusnv.log`. This log has one line per step per sample, saying what happened in that step, with any warning or error. For example:

```text
[variants2positions] strain2: 4 candidate variant positions kept (0.0008% of the 500,000 bp genome)
[pileup2diversity] strain2: 99.7% of the genome covered by at least one read, median depth 14.0x
[calling] Group group_pe_test: CNN scored 10 positions and called 10 of them real SNVs
[calling] Cov_filter (under 5 reads on either strand): 10 positions still varied between samples
          going in, this filter dropped 3 of them, leaving 7
```

However, in some cases you may be interested in the more complete log files. These are:

`accusnv.full.log` adds additional per-sample detail and shows you all of the output from dependencies like `bwa`, `samtools`, `bcftools`, `cutadapt` and `sickle` as well.

`accusnv.snakemake.log` contains all of the detailed output from Snakemake. 
