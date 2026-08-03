"""Per-rule memory + runtime tiers, and dynamic selection for the AccuSNV workflow.

Every non-mapping rule has four tiers - small, medium, large, huge - as (mem_mb, runtime_min).
The starting tier is chosen from a cheap size estimate the Snakefile can compute up front:
  * per-group rules  ->  samples x genome_length   (the driver of the big candidate-table arrays)
  * per-sample rules ->  genome_length
On a retry, Snakemake passes a higher `attempt`, which bumps the choice one tier up - so a job that
runs out of memory or time is automatically resubmitted at the next level. The numbers come from
itest/fit_resources.py (measured on one core, extrapolated to 1000 samples / 15 Mb / 500k SNVs,
with a 1.3x RAM margin and a 5x runtime margin) and are cross-checked against the exact array sizes.

THRESHOLDS[rule] holds the three driver cut-offs between the four tiers.
"""

# (mem_mb, runtime_min) for small, medium, large, huge. Derived by itest/fit_resources.py from
# measured peak RAM on one core, extrapolated to 1000 samples / 15 Mb / 500k SNVs via the exact
# array-byte formulas (validated within ~1.05-1.2x of measurement) with a 1.3x RAM margin.
# candidate_mutation_table dominates: its coverage + cov_norm arrays (~24 B/bp/sample) and counts
# arrays (~344 B/pos/sample) are why the huge tier reaches ~700 GB -- a real ceiling for the largest
# cohorts, and a candidate for future optimization (chunk the cov_norm normalisation).
TIERS = {
    'cutadapt':                 [(1000, 20), (1000, 60), (1000, 240), (1000, 720)],
    'sickle':                   [(500, 20), (500, 60), (500, 240), (500, 720)],
    'sam2bam':                  [(1000, 20), (1000, 60), (1000, 240), (1000, 720)],
    'mpileup2vcf':              [(1000, 20), (1000, 60), (1000, 240), (1000, 720)],
    'vcf2quals':                [(500, 20), (1000, 60), (1500, 240), (1500, 720)],
    'variants2positions':       [(500, 20), (500, 60), (500, 240), (500, 720)],
    'pileup2diversity':         [(1000, 47), (5000, 60), (14000, 240), (14000, 720)],
    'combine_positions':        [(500, 20), (500, 60), (500, 240), (6000, 720)],
    'candidate_mutation_table': [(1000, 20), (13000, 60), (116000, 240), (708000, 720)],
    'calling_accusnv':          [(1000, 20), (4000, 60), (38000, 240), (372000, 720)],
    'annotate_snvs':            [(500, 20), (500, 60), (1000, 240), (17000, 720)],
    'dnds':                     [(500, 20), (500, 60), (1000, 240), (16000, 720)],
    'build_tree':               [(500, 20), (500, 60), (1000, 240), (27000, 720)],
    'report_html':              [(1000, 29), (1000, 60), (1500, 240), (27000, 720)],
}

# Driver cut-offs (small|medium, medium|large, large|huge). Per-group: samples*bp; per-sample: bp.
# Geometric midpoints between the tiers' representative datasets, so each lands in its own tier.
_G = [27_386_127, 866_025_403, 6_708_203_932]   # per-group boundaries (samples*bp)
_S = [1_581_138, 8_660_254, 15_000_000]         # per-sample boundaries (bp)
THRESHOLDS = {
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
