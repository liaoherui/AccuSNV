#!/usr/bin/env python3
"""Rebuild the final AccuSNV SNV table from a candidate matrix."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from copy import deepcopy
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None

from snake_pipeline.modules import snv_module_recoded_with_dNdS as snv


def load_candidate_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        data = {name: handle[name] for name in handle.files}

    required = {"sample_names", "p", "counts", "quals", "indel_counter", "in_outgroup"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Missing arrays in {path}: {', '.join(sorted(missing))}")

    # Convert to expected dtypes
    data["sample_names"] = data["sample_names"].astype(str)
    data["p"] = data["p"].astype(np.int64)
    data["counts"] = data["counts"].astype(np.int64)
    data["quals"] = (data["quals"].astype(np.int64) * -1)
    data["indel_counter"] = data["indel_counter"].astype(np.int64)
    data["in_outgroup"] = data["in_outgroup"].astype(bool)

    for opt_key in ("prob", "label", "recomb", "samples_exclude_bool"):
        if opt_key in data:
            data[opt_key] = data[opt_key]

    return data


def _progress(iterable, *, desc: str | None = None, total: int | None = None):
    """Wrap ``iterable`` in ``tqdm`` if it is available."""

    if tqdm is None:
        return iterable

    return tqdm(iterable, desc=desc, total=total)


def infer_ancestral_nts(my_calls: snv.calls_object, ref: snv.reference_genome_object) -> np.ndarray:
    if not np.any(my_calls.in_outgroup):
        return ref.get_ref_NTs_as_ints(my_calls.p)

    calls_outgroup = my_calls.get_calls_in_outgroup_only()
    ancestral = np.zeros(my_calls.num_pos, dtype=np.int64)

    for idx in _progress(range(my_calls.num_pos), desc="Inferring ancestral NTs", total=my_calls.num_pos):
        column = calls_outgroup[:, idx]
        column = column[column != 0]
        if column.size == 0:
            continue
        alleles, counts = np.unique(column, return_counts=True)
        ancestral[idx] = int(alleles[np.argmax(counts)])

    missing = ancestral == 0
    if np.any(missing):
        ancestral[missing] = ref.get_ref_NTs_as_ints(my_calls.p[missing])

    return ancestral


def compute_filter_tokens(my_cmt: snv.cmt_data_object, min_cov: int) -> Tuple[snv.calls_object, Dict[str, Dict[int, int]], np.ndarray]:
    dpt: Dict[str, Dict[int, int]] = {}
    my_calls = snv.calls_object(my_cmt)

    # Quality filter
    raw = deepcopy(my_calls)
    my_calls.filter_calls_by_element(my_cmt.quals < 30)
    tokens = snv.token_generate(raw.calls.T, my_calls.calls.T, "filter-qual")
    dpt["qual"] = {int(pos): int(val) for pos, val in zip(my_calls.p, tokens)}

    # Coverage filter (per strand)
    raw = deepcopy(my_calls)
    my_calls.filter_calls_by_element(my_cmt.fwd_cov < min_cov)
    my_calls.filter_calls_by_element(my_cmt.rev_cov < min_cov)
    tokens = snv.token_generate(raw.calls.T, my_calls.calls.T, "filter-coverage")
    dpt["cov"] = {int(pos): int(val) for pos, val in zip(my_calls.p, tokens)}

    # Major allele frequency filter
    raw = deepcopy(my_calls)
    my_calls.filter_calls_by_element(my_cmt.major_nt_freq < 0.85)
    tokens = snv.token_generate(raw.calls.T, my_calls.calls.T, "filter-major allele freq")
    dpt["maf"] = {int(pos): int(val) for pos, val in zip(my_calls.p, tokens)}

    # Indel support filter
    raw = deepcopy(my_calls)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_indel = np.sum(my_cmt.indel_stats, axis=2) / my_cmt.coverage
    frac_indel[~np.isfinite(frac_indel)] = 0.0
    my_calls.filter_calls_by_element(frac_indel > 0.33)
    tokens = snv.token_generate(raw.calls.T, my_calls.calls.T, "filter-indel")
    dpt["indel"] = {int(pos): int(val) for pos, val in zip(my_calls.p, tokens)}

    # Fraction ambiguous filter
    raw = deepcopy(my_calls)
    my_calls.filter_calls_by_position(my_calls.get_frac_Ns_by_position() > 1.0)
    tokens = snv.token_generate(raw.calls.T, my_calls.calls.T, "filter-max_fraction_ambigious_samples")
    dpt["mfas"] = {int(pos): int(val) for pos, val in zip(my_calls.p, tokens)}

    # Median coverage per position filter
    raw = deepcopy(my_calls)
    median_cov = np.nan_to_num(np.median(my_cmt.coverage, axis=0))
    my_calls.filter_calls_by_position(median_cov < 5)
    tokens = snv.token_generate(raw.calls.T, my_calls.calls.T, "filter-min_median_coverage_position")
    dpt["mmcp"] = {int(pos): int(val) for pos, val in zip(my_calls.p, tokens)}

    # Copy number filter (approximate using per-sample median coverage across SNVs)
    raw = deepcopy(my_calls)
    sample_median = np.nanmedian(my_cmt.coverage, axis=1)
    sample_median[sample_median == 0] = 1
    copy_num = my_cmt.coverage / sample_median[:, None]
    copy_num = np.nan_to_num(copy_num, nan=0.0)
    mean_copy = np.mean(copy_num, axis=0)
    max_copy = np.max(copy_num, axis=0)
    my_calls.filter_calls_by_position(mean_copy > 4)
    my_calls.filter_calls_by_position(max_copy > 7)
    tokens = snv.token_generate(raw.calls.T, my_calls.calls.T, "filter-copy number")
    dpt["cpn"] = {int(pos): int(val) for pos, val in zip(my_calls.p, tokens)}

    final_tokens = tokens

    return my_calls, dpt, final_tokens


def build_prediction_table(all_positions: Iterable[int],
                           wd_positions: Iterable[int],
                           dpt: Dict[str, Dict[int, int]],
                           dlab: Dict[int, int],
                           dprob: Dict[int, str],
                           recomb_dict: Dict[int, bool],
                           gap_dict: Dict[int, str],
                           my_cmt_zero: snv.cmt_data_object,
                           min_cov: int,
                           output_dir: str) -> str:
    all_p = np.array(sorted(set(int(p) for p in all_positions)), dtype=np.int64)
    wd_set = set(int(p) for p in wd_positions)

    freq_cmt = my_cmt_zero.copy()
    freq_d, check_d = snv.cal_freq_amb_samples(all_p, freq_cmt)
    cutoff = 0.1 if my_cmt_zero.num_samples > 20 else 0.25

    rows = []
    for pos in _progress(all_p, desc="Assembling prediction table", total=all_p.size):
        cnn_label = dlab.get(pos, "skip")
        cnn_prob = dprob.get(pos, "skip")
        wide_variant = "1" if pos in wd_set else "0"
        recomb_flag = "1" if recomb_dict.get(pos, False) else "0"
        gap_flag = gap_dict.get(pos, "0")
        freq_val = float(freq_d[pos])
        qual_token = dpt["qual"].get(pos, 0)
        warr = [str(cnn_label), str(cnn_prob)]
        final_label = snv.dec_final_lab(
            str(cnn_label), warr, wide_variant, recomb_flag, gap_flag,
            freq_val, qual_token, min_cov, check_d[pos], cutoff
        )
        rows.append({
            "genome_pos": pos,
            "Pred_label": final_label,
            "CNN_pred": warr[0],
            "WideVariant_pred": wide_variant,
            "CNN_prob": warr[1],
            "Qual_filter (<30)": dpt["qual"].get(pos, 0),
            "Cov_filter (<5)": dpt["cov"].get(pos, 0),
            "MAF_filter (>0.85)": dpt["maf"].get(pos, 0),
            "Indel_filter (<0.33)": dpt["indel"].get(pos, 0),
            "MFAS_filter (1)": dpt["mfas"].get(pos, 0),
            "MMCP_filter (5)": dpt["mmcp"].get(pos, 0),
            "CPN_filter (4,7)": dpt["cpn"].get(pos, 0),
            "Fix_filter": dpt["fix"].get(pos, 0),
            "Whether_recomb": recomb_flag,
            "Fraction_ambigious_samples": f"{freq_val:.6f}",
            "Gap_filter": gap_flag,
        })

    df = pd.DataFrame(rows)
    df.sort_values("genome_pos", inplace=True)
    path = os.path.join(output_dir, "snv_table_cnn_plus_filter.txt")
    df.to_csv(path, sep="\t", index=False)
    return path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="Path to candidate_mutation_table_final.npz")
    parser.add_argument("--reference", required=True, help="Reference genome directory (with FASTA + GFF)")
    parser.add_argument("--output", default=None,
                        help="Output TSV path. Default: alongside the candidate file with the standard name.")
    parser.add_argument("--min-cov", type=int, default=5,
                        help="Per-strand coverage threshold used by the filters (default: 5)")

    args = parser.parse_args(list(argv) if argv is not None else None)

    data = load_candidate_npz(args.candidate)

    my_cmt = snv.cmt_data_object(
        data["sample_names"],
        data["in_outgroup"],
        data["p"],
        data["counts"],
        data["quals"],
        data["indel_counter"],
    )

    samples_exclude = data.get("samples_exclude_bool")
    if samples_exclude is not None:
        samples_exclude = samples_exclude.astype(bool)
        if samples_exclude.shape[0] != my_cmt.num_samples:
            raise ValueError("samples_exclude_bool length mismatch")
        my_cmt.filter_samples(~samples_exclude)

    calls_filtered, dpt, final_tokens = compute_filter_tokens(my_cmt, args.min_cov)

    reference = snv.reference_genome_object(args.reference)
    calls_ancestral = infer_ancestral_nts(calls_filtered, reference)

    ingroup_mask = ~calls_filtered.in_outgroup
    if not np.any(ingroup_mask):
        raise ValueError("No ingroup samples available for annotation")

    calls_ingroup = calls_filtered.get_calls_in_sample_subset(ingroup_mask)
    quals_ingroup = my_cmt.quals[ingroup_mask, :]
    mut_qual, _ = snv.compute_mutation_quality(calls_ingroup, quals_ingroup)

    recomb = data.get("recomb", np.zeros(my_cmt.p.shape[0], dtype=bool)).astype(bool)
    recomb_dict = {int(pos): bool(val) for pos, val in zip(my_cmt.p, recomb)}
    dpt["recomb"] = recomb_dict

    filter_not_N = calls_ingroup != snv.nts2ints("N")
    filter_not_anc = calls_ingroup != np.tile(calls_ancestral, (np.count_nonzero(ingroup_mask), 1))
    filter_mutqual = np.tile(mut_qual, (np.count_nonzero(ingroup_mask), 1)) >= 1
    filter_not_recomb = np.tile(~recomb, (np.count_nonzero(ingroup_mask), 1))
    fixedmutation = filter_not_N & filter_not_anc & filter_mutqual & filter_not_recomb

    has_mutation = np.any(fixedmutation, axis=0)
    goodpos_idx = np.where(has_mutation)[0]
    tokens_final = snv.generate_tokens_last(final_tokens, goodpos_idx, "filter-fixedmutation")
    dpt["fix"] = {int(pos): int(val) for pos, val in zip(my_cmt.p, tokens_final)}

    label = data.get("label", np.ones(my_cmt.p.shape[0], dtype=int)).astype(int)
    prob = data.get("prob", np.zeros(my_cmt.p.shape[0], dtype=float)).astype(float)
    dlab = {int(pos): int(val) for pos, val in zip(my_cmt.p, label)}
    dprob = {int(pos): ("skip" if np.isnan(val) else f"{float(val):.6f}")
             for pos, val in zip(my_cmt.p, prob)}

    cnn_positions = my_cmt.p[label == 1]
    wd_positions = my_cmt.p[has_mutation]

    my_cmt_zero = my_cmt.copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        prediction_path = build_prediction_table(
            my_cmt.p,
            wd_positions,
            dpt,
            dlab,
            dprob,
            recomb_dict,
            {},
            my_cmt_zero,
            args.min_cov,
            tmpdir,
        )

        annotation_path = os.path.join(tmpdir, "snv_table_mutations_annotations.tsv")
        my_cmt_for_annotation = my_cmt.copy()
        my_cmt_for_annotation.filter_samples(~my_cmt_for_annotation.in_outgroup)
        calls_for_table = snv.ints2nts(calls_filtered.calls)
        mutations_annotated = snv.annotate_mutations(
            reference,
            my_cmt.p,
            np.tile(calls_ancestral, (np.count_nonzero(ingroup_mask), 1)),
            calls_ingroup,
            my_cmt_for_annotation,
            fixedmutation,
            mut_qual.flatten(),
            250,
        )
        snv.write_mutation_table_as_tsv(
            my_cmt.p,
            mut_qual.flatten(),
            list(my_cmt.sample_names),
            mutations_annotated,
            calls_for_table,
            list(my_cmt.sample_names),
            annotation_path,
        )

        draft_path = os.path.join(tmpdir, "snv_table_merge_all_mut_annotations_draft.tsv")
        snv.merge_two_tables(prediction_path, annotation_path, draft_path)

        draft_df = pd.read_csv(draft_path, sep="\t")

    final_df = draft_df[draft_df["Pred_label"].astype(str) != "0"].copy()
    final_df.sort_values("genome_pos", inplace=True)

    default_name = "snv_table_merge_all_mut_annotations_final.tsv"
    output_path = args.output
    if output_path is None:
        out_dir = os.path.dirname(os.path.abspath(args.candidate))
        output_path = os.path.join(out_dir, default_name)
    else:
        # Allow passing a directory as the output argument.
        if output_path.endswith(os.sep) or os.path.isdir(output_path):
            out_dir = output_path.rstrip(os.sep) or "."
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, default_name)
        else:
            out_dir = os.path.dirname(output_path)
            if out_dir and not os.path.isdir(out_dir):
                os.makedirs(out_dir, exist_ok=True)

    final_df.to_csv(output_path, sep="\t", index=False)
    print(f"Final SNV table written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
