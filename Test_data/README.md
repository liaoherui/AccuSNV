# Test data

The two original AccuSNV test datasets, paired-end and single-end, with the outputs they are
expected to produce. Both use four *Cutibacterium acnes* isolates mapped against the same
reference.

## Layout

```
Test_data/
    samples_cae_test_pe.csv        sample sheet for the paired-end test
    samples_cae_test_se.csv        sample sheet for the single-end test
    reference_genomes/Cae_ref/     reference: genome.fasta and genome.gff
    reads_pe_test/strain1..4/      paired-end reads (strainN_1.fq.gz, strainN_2.fq.gz)
    reads_se_test/strain1..4/      single-end reads (strainN.fq.gz)
    expected_output/pe_test/       the results the paired-end test should produce
    expected_output/se_test/       the results the single-end test should produce
```

The paths in both sample sheets are relative to the repository root, so run the commands below
from there.

## Running the tests

```
accusnv -m local -i Test_data/samples_cae_test_pe.csv -r Test_data/reference_genomes -o pe_out
accusnv -m local -i Test_data/samples_cae_test_se.csv -r Test_data/reference_genomes -o se_out
```

Each takes a few minutes on a laptop. `-j <cores>` sets how many cores to use.

The reference directory needs to be writable the first time, because the run builds the bwa and
samtools indexes there. If it is not, build them first with `bwa index genome.fasta` and
`samtools faidx genome.fasta`.

## Checking the results

Each `expected_output/` folder holds the text results of a full run:

| File | Where the run writes it |
| --- | --- |
| `snv_table_final.tsv` | `group_<group>_snv_table_final.tsv` |
| `snv_table_unfiltered.tsv` | `group_<group>_snv_table_unfiltered.tsv` |
| `snv_table_cnn_raw.tsv` | `2-SNV-filtering/group_<group>/snv_table_cnn_raw.tsv` |
| `snvs_per_sample.tsv` | `2-SNV-filtering/group_<group>/snvs_per_sample.tsv` |
| `snv_tree_final.nwk.tree` | `group_<group>_snv_tree_final.nwk.tree` |
| `dnds_genomewide.tsv` | `3-Analysis/group_<group>/dNdS_out/dnds_genomewide.tsv` |
| `dnds_per_gene.tsv` | `3-Analysis/group_<group>/dNdS_out/dnds_per_gene.tsv` |
| `snv_table_tree_distances.tsv` | `3-Analysis/group_<group>/phylogeny/snv_table_tree_distances.tsv` |
| `snv_table_simple_stats.tsv` | `3-Analysis/group_<group>/phylogeny/snv_table_simple_stats.tsv` |

So to compare the paired-end run:

```
diff Test_data/expected_output/pe_test/snv_table_final.tsv pe_out/group_pe_test_snv_table_final.tsv
diff Test_data/expected_output/pe_test/dnds_genomewide.tsv pe_out/3-Analysis/group_pe_test/dNdS_out/dnds_genomewide.tsv
```

## What these runs produce

| | paired-end | single-end |
| --- | --- | --- |
| samples | 4 | 4 |
| SNVs in the final table | 38 | 39 |
| nonsynonymous / synonymous | 19 / 11 | 26 / 10 |
| dN/dS | 0.5489 (95% CI 0.2482-1.2767) | 0.8262 (95% CI 0.3855-1.9200) |

The interactive dashboard and the PNG figures are not included here. They are large, and the
dashboard embeds compressed read counts that do not compare line by line. Open
`<out_dir>/group_<group>_snv_dashboard.html` from your own run instead.

## How far these files have been checked

The expected outputs were produced by a full run of the current code.

Both tests have been reproduced. A second, independent full run from clean produced all nine
files byte for byte identical to the ones here, for each test. For the paired-end test,
re-running the downstream stages on top of that with `--downstream_only` reproduced them again.

A difference against these files is worth looking into rather than being a failure by itself. A
different bwa, samtools or bcftools version can shift the SNV tables on its own.
