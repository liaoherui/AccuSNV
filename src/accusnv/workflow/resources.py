"""Per-rule memory + runtime tiers, and dynamic selection for the AccuSNV workflow.

Every rule has four tiers - small, medium, large, huge - as (mem_mb, runtime_min).
The starting tier is chosen from a cheap size estimate the Snakefile can compute up front:
  * per-group rules  ->  samples x genome_length   (the driver of the big candidate-table arrays)
  * per-sample rules ->  genome_length
On a retry, Snakemake passes a higher `attempt`, which bumps the choice one tier up - so a job that
runs out of memory or time is automatically resubmitted at the next level.

THRESHOLDS[rule] holds the three driver cut-offs between the four tiers.
"""

# (mem_mb, runtime_min) for small, medium, large, huge. Round numbers set by hand: generous enough
# that no rule starts short, and capped at 120 GB, which is roughly what one node will grant a
# single job. candidate_mutation_table is the rule that reaches the cap first -- its coverage +
# cov_norm arrays run ~24 B/bp/sample and its counts arrays ~344 B/pos/sample -- so for the largest
# cohorts 120 GB is a real ceiling, not a margin, and chunking those arrays is the way past it.
TIERS = {
    # bwa/bowtie2 hold the reference index in RAM (~6 bytes per base), so memory follows genome
    # length. Runtime follows read depth, which we cannot see up front, so it starts generous.
    'create_mapping_index':     [(4000, 60), (8000, 60), (32000, 120), (120000, 240)],
    'mapping':                  [(16000, 120), (32000, 120), (64000, 240), (120000, 720)],
    'cutadapt':                 [(8000, 20), (16000, 60), (32000, 240), (64000, 720)],
    'sickle':                   [(4000, 20), (8000, 60), (16000, 240), (64000, 720)],
    'sam2bam':                  [(2000, 20), (4000, 60), (16000, 240), (64000, 720)],
    'mpileup2vcf':              [(4000, 20), (8000, 60), (16000, 240), (64000, 720)],
    'vcf2quals':                [(4000, 20), (8000, 60), (16000, 240), (64000, 720)],
    'variants2positions':       [(4000, 20), (8000, 60), (16000, 240), (64000, 720)],
    'pileup2diversity':         [(4000, 30), (8000, 60), (32000, 240), (64000, 720)],
    'combine_positions':        [(4000, 20), (8000, 60), (32000, 240), (64000, 720)],
    'candidate_mutation_table': [(4000, 20), (16000, 60), (64000, 240), (120000, 720)],
    'calling_accusnv':          [(8000, 20), (16000, 60), (64000, 240), (120000, 720)],
    'annotate_snvs':            [(4000, 20), (8000, 60), (16000, 240), (64000, 720)],
    'dnds':                     [(4000, 20), (8000, 60), (16000, 240), (64000, 720)],
    'build_tree':               [(4000, 20), (16000, 60), (32000, 240), (64000, 720)],
    'report_html':              [(2000, 30), (4000, 60), (16000, 240), (64000, 720)],
}

# Driver cut-offs (small|medium, medium|large, large|huge). Per-group: samples*bp; per-sample: bp.
# Geometric midpoints between the tiers' representative datasets, so each lands in its own tier.
_G = [27_386_127, 866_025_403, 6_708_203_932]   # per-group boundaries (samples*bp)
_S = [1_581_138, 8_660_254, 15_000_000]         # per-sample boundaries (bp)
THRESHOLDS = {
    'create_mapping_index': _S, 'mapping': _S,
    'cutadapt': _S, 'sickle': _S, 'sam2bam': _S, 'mpileup2vcf': _S, 'vcf2quals': _S,
    'variants2positions': _S, 'pileup2diversity': _S,
    'combine_positions': _G, 'candidate_mutation_table': _G, 'calling_accusnv': _G,
    'annotate_snvs': _G, 'dnds': _G, 'build_tree': _G, 'report_html': _G,
}


def pick(rule, driver, attempt=1):
    """Return (mem_mb, runtime_min): the tier for `driver`, bumped up (attempt-1) times."""
    tiers = TIERS[rule]
    start = sum(driver > cut for cut in THRESHOLDS[rule])
    return tiers[min(start + attempt - 1, len(tiers) - 1)]


def mem(rule, driver, attempt=1):
    return pick(rule, driver, attempt)[0]


def runtime(rule, driver, attempt=1):
    return pick(rule, driver, attempt)[1]
