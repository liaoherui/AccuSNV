# Adjusting parameters

AccuSNV is set up with parameters that have been optimized for accuracy in SNV calling, and in many cases, you may not have to adjust these defaults. However, depending on your particular data and run environment, you may have to, and all of the parameters behind an AccuSNV run are organized into two `yaml` config files:

**pipeline.yaml:** contains all pipeline parameters.

**config.yaml:** contains all runtime parameters (for snakemake and SLURM)

By default, both config files are created in `<output_dir>/configs/` with their defaults filled in on every run. Alternatively, you can pass `-c [config.yaml]` or `-p [pipeline.yaml]` to `accusnv` to use previously modified versions of these files. If you do so, your files are copied into `<output_dir>/configs/` for your new run. Further, any command line options passed to AccuSNV are written into those config files. This way, the config files in a given output directory are always indicative of the values run during that run.

To modify parameters, you can run once (with `accusnv -m dryrun`), copy the generated file, edit it, and pass it back:

```bash
accusnv -p pipeline.yaml -c config.yaml -i samples.csv -r reference_genomes -o my_output
```

## Pipeline parameters: pipeline.yaml

### Logging

| Key             | Default            | Description                                                                 |
|:--------------- | ------------------ | --------------------------------------------------------------------------- |
| `log_file`      | `accusnv.log`      | The summary log file name found in `<outdir>`                               |
| `full_log_file` | `accusnv.full.log` | The extended log file name that contains the output of every external tool. |

### Read trimming and mapping

| Key                        | Default       | Description                                                                                                                                                                                                                                            |
| -------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `aligner`                  | `bwa`         | `bwa` or `bowtie2`.                                                                                                                                                                                                                                    |
| `skip_samclip`             | `false`       | Prevents piping `bwa` alignments through `samclip --max 0`, which drops any read with a soft-clipped end. `bowtie2` never uses `samclip`. In testing, the impact of samclip has been found to be minor either way, but inclusion is more conservative. |
| `cores`                    | `4`           | Number of cores each `cutadapt`, samtools, and mapping job uses. In a local run it is also the total cores for the whole run. `accusnv -j` overrides/writes this.                                                                                      |
| `adapter_sequence`         | `CTGTCTCTTAT` | The 3' adapter `cutadapt` trims. The default is the Nextera adapter, which may need to be changed for TruSeq or other kits.                                                                                                                            |
| `sickle_quality`           | `20`          | Phred quality threshold for `sickle` trimming.                                                                                                                                                                                                         |
| `sickle_min_length`        | `50`          | Reads shorter than this after trimming are dropped.                                                                                                                                                                                                    |
| `bowtie2_maxins`           | `2000`        | Maximum insert size (`-X`) for paired-end `bowtie2`. Ignored with `bwa`.                                                                                                                                                                               |
| `markdup_optical_distance` | `100`         | Optical-duplicate pixel distance for `samtools markdup`. Consider adjusting to `2500` for patterned flowcells such as NovaSeq.                                                                                                                         |

### Pileup and variant calling

| Key                 | Default | Description                                                                                                                          |
| ------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `mpileup_min_mapq`  | `30`    | Minimum read mapping quality for `samtools`/`bcftools mpileup`.                                                                      |
| `mpileup_max_depth` | `3000`  | Maximum per-file pileup depth.                                                                                                       |
| `variant_min_af`    | `0.75`  | Minimum alternate-allele fraction for `bcftools` to emit a candidate variant.                                                        |
| `max_fq`            | `-30`   | The `bcftools` FQ score below which a position counts as variable. FQ is on a negative scale, so more negative is stronger evidence. |

### Sample-level filters

| Key            | Default | Description                                                                                                                                                                                       |
| -------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `min_cov_samp` | `45`    | Remove a *sample* when it has no read coverage at more than this percentage of the candidate positions. Set to `100` to keep every sample.                                                        |
| `exclude_samp` | `null`  | Either a comma-separated list of sample names to drop, **or a number of max SNVs per sample**. When it is an integer, the pipeline will drop any sample *with at least that many candidate SNVs*. |

### Per-basecall filters

These four filters decide whether one sample's base call at one position is trusted for nomination of a candidate SNV. 

| Key                          | Default | Description                                                                                                                                                                                                                                                             |
| ---------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `call_min_qual`              | `30`    | Minimum call quality (negative `bcftools` FQ score, so higher is better). (`Qual_filter`)                                                                                                                                                                               |
| `min_cov_filt`               | `5`     | Minimum forward reads *and* minimum reverse reads at the position. (`Cov_filter`)                                                                                                                                                                                       |
| `call_min_major_allele_freq` | `0.85`  | The most common base must account for at least this fraction of the reads. (`MAF_filter`)                                                                                                                                                                               |
| `call_max_indel_frac`        | `0.33`  | Drop the call when more than this fraction of the covering reads contain an insertion or deletion. (`Indel_filter`)                                                                                                                                                     |
| `min_mut_qual`               | `1`     | Minimum "mutation quality" for a SNV. The mutation quality is defined by AccuSNV as the best FQ score for a given site, for the more weakly supported variant at that site. Effectively, it discards sites where there is no strong evidence that two samples disagree. |

`min_cov_filt` is the filter that matters most at low depth. In low coverage samples, you may consider reducing this. 

### Across-sample position filters

These filters check SNV position quality across the whole group of samples.

| Key                            | Default | Description                                                                                                                                        |
| ------------------------------ | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `max_frac_ambiguous_samples`   | `1`     | Drop a position when more than this fraction of samples have no confident base call. The default is `1` which means never applied. (`MFAS_filter`) |
| `min_median_coverage_position` | `5`     | Drop a position whose median read depth across samples is below this. (`MMCP_filter`)                                                              |
| `max_mean_copynum`             | `4`     | Drop a position whose depth averages more than this many times the genome-wide median. (`CPN_filter`)                                              |
| `max_max_copynum`              | `7`     | As above, but for any single sample. (`CPN_filter`)                                                                                                |

### Overruling SNVs rejected by the CNN

When the AccuSNV CNN rejects a candidate site as a SNV but it still passes all of the rule filters above, AccuSNV keeps the SNV only if the positions with the variant do not have mixed read support (the `Fraction_ambiguous_samples` column). This cutoff is by default <25% if there are less than 20 samples in the group, and 10% if there are more than 20 samples in the group.

| Key                    | Default | Description                                                                                           |
| ---------------------- |:------- | ----------------------------------------------------------------------------------------------------- |
| `rebuild_sample_count` | `20`    | Group size above which the stricter cutoff applies.                                                   |
| `rebuild_cutoff_many`  | `0.1`   | The `Fraction_ambiguous_samples` cutoff used when there are more samples than `rebuild_sample_count`. |
| `rebuild_cutoff_few`   | `0.25`  | The `Fraction_ambiguous_samples` cutoff used otherwise.                                               |

### Manual SNV curation options

| Key                 | Default | Description                                                                                                                |
| ------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------- |
| `exclude_positions` | `""`    | Path to a file of `genome_pos` values, one per line, to exclude from the SNV set regardless of whatever the model decided. |
| `include_positions` | `""`    | Path to a file of `genome_pos` values, one per line, to keep in the SNV set regardless of whatever the model decided.      |

`accusnv --exclude_positions` and `--include_positions` create these parameter key values.  Both are applied after SNV calling, so `--downstream_only` gets these SNVs without re-calling.

### Recombination detection

| Key                     | Default | Description                                                          |
| ----------------------- | ------- | -------------------------------------------------------------------- |
| `recomb_distance_bp`    | `1000`  | Only SNV pairs closer together than this are tested for correlation. |
| `recomb_corr_threshold` | `0.75`  | Correlation above which a pair is flagged as recombinant.            |
| `skip_recombination`    | `false` | Turn detection off entirely, which means nothing is ever flagged.    |

See [Recombination](recombination.md) for what these actually do and when to change them.

### Annotation

| Key                              | Default | Description                                                                                                                                                       |
| -------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `annotate_min_major_allele_freq` | `0.75`  | The major-allele cutoff used when base calls are rebuilt for annotation. Looser than the calling cutoff so the per-sample columns show a base rather than an `N`. |

The promoter window, which is how far upstream of a gene a SNV can be and still be reported as type `P` for promoter, is fixed at 250 bp, and not currently a flag.

### Very large runs: "fast mode"

| Key                   | Default  | Description                                               |
| --------------------- | -------- | --------------------------------------------------------- |
| `fast_mode_positions` | `100000` | Above this many candidate positions, switch to fast mode. |

In fast mode AccuSNV scores every position with the CNN and then stops. It writes
`snv_table_cnn_raw.tsv` and `candidate_mutation_table_final.npz`, and skips annotation, dN/dS, the tree and the report. If this threshold is crossed in a group of isolates, the user may wish to select more closely related groups of isolates for more accurate SNV calling.

### Optional stages

| Key                   | Default | Description                                                                              |
| --------------------- | ------- | ---------------------------------------------------------------------------------------- |
| `skip_report`         | `false` | Skip the HTML report.                                                                    |
| `skip_dnds`           | `false` | Skip dN/dS.                                                                              |
| `skip_trees`          | `false` | Skip the parsimony tree. dMRCA is still estimated.                                       |
| `use_nj_tree`         | `false` | Build a neighbor-joining tree with Biopython instead of a parsimony tree with `dnapars`. |
| `build_snv_trees`     | `false` | Write one NEXUS tree per SNV with tips colored by base call.                             |
| `skip_all_downstream` | `false` | Stop after the annotated SNV tables.                                                     |
| `downstream_only`     | `false` | Re-run only the evolutionary analyses.                                                   |

## Runtime configuration: config.yaml

`config.yaml` is the Snakemake execution profile, plus the input and output paths that AccuSNV will fill in from your `-i`, `-r` and `-o` arguments.

```yaml
rerun-incomplete: true
latency-wait: 30
keep-going: true
printshellcmds: true

# Resubmit a failed job at the next resource tier up. 0 disables.
retries: 3

# Slurm execution
executor: slurm
jobs: 100
default-resources:
  - slurm_partition="partition_a,partition_b"
  - mem_mb=4000
  - runtime=60
```

| Key                 | Description                                                                                                                                                                      |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rerun-incomplete`  | Redo any job whose output is half-written, for example after a killed run. Almost certainly best to keep as true.                                                                |
| `latency-wait`      | Seconds to wait for a file to appear on a shared filesystem before declaring it missing. You probably only need to increase this on an overloaded or old system.                 |
| `keep-going`        | Carry on with jobs that do not depend on a failed one, instead of stopping the whole run.                                                                                        |
| `retries`           | How many times a failed job is resubmitted, each time with a higher resource tier.                                                                                               |
| `jobs`              | Maximum Slurm jobs running at once.                                                                                                                                              |
| `default-resources` | The resources for every job. By default, resources are decided per job and per tier by `resources.py`. See below for how to edit this.                                           |
| `cores`             | This is passed from the CLI parameter `-j` and written in `pipeline.yaml`. It is repeated here for Snakemake - if you want to edit this, edit `-j` or `pipeline.yaml`, not here. |

The `config:` block at the bottom includes `sample_table`, `outdir` and `env`. AccuSNV fills those in automatically from the command line parameters, so you do not need to touch them.

### Slurm job resource allocation

The memory and run-time resources requested for each job are determined for each step (Snakemake rule) by `resources.py`. By default, Snakemake will start with a reasonably small memory and run-time request for a given job. If it then runs out of time or memory, it will restart that job at a higher 'tier', requesting more memory and run-time.

The tiers are a plain Python dictionary in `src/accusnv/workflow/resources.py` if you need to
adjust them for your cluster. Here are their default values, in MB of RAM and minutes:

```
TIERS = {
    'create_mapping_index':     [(2000, 60), (8000, 60), (320000, 240)],
    'mapping':                  [(8000, 120), (32000, 120), (64000, 240)], ## medium amount of RAM
    'cutadapt':                 [(2000, 30), (8000, 60), (32000, 240)],
    'sickle':                   [(2000, 30), (8000, 60), (32000, 240)],
    'sam2bam':                  [(4000, 30), (16000, 60), (64000, 240)],
    'mpileup2vcf':              [(1000, 30), (4000, 60), (64000, 240)],
    'vcf2quals':                [(2000, 30), (4000, 60), (64000, 240)],
    'variants2positions':       [(1000, 30), (4000, 60), (64000, 240)],
    'upstream_rejects':         [(1000, 30), (4000, 60), (64000, 240)],
    'pileup2diversity':         [(8000, 30), (32000, 60), (128000, 240)],
    'combine_positions':        [(1000, 30), (4000, 60), (32000, 240)],
    'combine_upstream_rejects': [(1000, 30), (4000, 60), (32000, 240)],
    'candidate_mutation_table': [(32000, 30), (64000, 60), (128000, 240)], ## can use a LOT of RAM
    'calling_accusnv':          [(16000, 30), (64000, 60), (128000, 240)], ## can use a decent amount of RAM
    'annotate_snvs':            [(4000, 30), (16000, 60), (64000, 240)],
    'dnds':                     [(2000, 30), (16000, 60), (64000, 240)],
    'build_tree':               [(2000, 30), (16000, 60), (64000, 240)],
    'report_html':              [(2000, 30), (4000, 60), (32000, 240)],
}
```

If you need to change the requests for a particular stage, you can do that by adding a section to the `config.yaml` that will override these parameters for a particular pipeline stage. Here is an example for how to increase the memory required by the candidate_mutation_table stage:

```
set-resources:
  candidate_mutation_table:
    mem_mb: 200000
    runtime: 480
```
