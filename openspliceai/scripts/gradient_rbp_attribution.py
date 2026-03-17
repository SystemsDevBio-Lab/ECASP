#!/usr/bin/env python
"""
Gradient-based attribution for a single variant under FiLM conditioning.

Example:
python -m openspliceai.scripts.gradient_rbp_attribution \
  --variant 'chr17:28369751:G>GC' \
  --gene VTN \
  --model runs/stage2_reference/model_best.pt \
  --ref-genome /path/genome.fa \
  --annotation data/grch38_chr.txt \
  --flanking-size 10000 \
  --rbp-expression data/blood_features.json \
  --film-strengths 1 \
  --target DS_AG \
  --rank-metric grad \
  --top-k 20 \
  --output results/vtn_blood_rbp_gradient.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

from openspliceai.predict.predict import model_predict_proba
from openspliceai.rbp.expression import load_rbp_expression
from openspliceai.variant.utils import Annotator, normalise_chrom, one_hot_encode


def parse_variant(variant: str) -> Tuple[str, int, str, str]:
    parts = variant.strip().split(":")
    if len(parts) not in (3, 4):
        raise ValueError(
            "Variant must look like 'chr:pos:REF>ALT' or 'chr:pos:REF:ALT'. "
            f"Got: {variant}"
        )
    chrom = parts[0]
    pos = int(parts[1])
    if len(parts) == 4:
        ref, alt = parts[2], parts[3]
    else:
        ref_alt = parts[2]
        if ">" not in ref_alt:
            raise ValueError(
                "Variant must contain 'REF>ALT'. Quote or escape it in shell, "
                f"e.g. --variant 'chr17:28369751:G>GC'. Got: {variant}"
            )
        ref, alt = ref_alt.split(">", 1)
    if not ref or not alt:
        raise ValueError(f"Variant REF/ALT missing: {variant}")
    return chrom, pos, ref, alt


def load_name_list(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def prepare_sequences(
    ann: Annotator,
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    dist_var: int,
    flanking_size: int,
    dist_ann: Tuple[int, int, int] | None = None,
) -> Tuple[str, str, int]:
    cov = 2 * dist_var + 1
    wid = flanking_size + cov
    chrom_norm = normalise_chrom(chrom, list(ann.ref_fasta.keys())[0])
    seq = ann.ref_fasta[chrom_norm][pos - wid // 2 - 1 : pos + wid // 2].seq
    if seq[wid // 2 : wid // 2 + len(ref)].upper() != ref.upper():
        raise ValueError("Reference allele does not match genome sequence.")
    if len(seq) != wid:
        raise ValueError("Variant too close to chromosome end for the chosen flanking size.")

    pad_size = wid // 2
    if dist_ann is not None:
        pad_start = max(pad_size + dist_ann[0], 0)
        pad_end = max(pad_size - dist_ann[1], 0)
        x_ref = "N" * pad_start + seq[pad_start : wid - pad_end] + "N" * pad_end
    else:
        x_ref = seq
    x_alt = x_ref[:pad_size] + alt + x_ref[pad_size + len(ref) :]
    return x_ref, x_alt, cov


def apply_indel_adjustment(
    y_alt: torch.Tensor,
    ref_len: int,
    alt_len: int,
    cov: int,
) -> torch.Tensor:
    if ref_len > 1 and alt_len == 1:
        del_len = ref_len - alt_len
        return torch.cat(
            [
                y_alt[:, : cov // 2 + alt_len, :],
                torch.zeros((1, del_len, 3), device=y_alt.device, dtype=y_alt.dtype),
                y_alt[:, cov // 2 + alt_len :, :],
            ],
            dim=1,
        )
    if ref_len == 1 and alt_len > 1:
        return torch.cat(
            [
                y_alt[:, : cov // 2, :],
                torch.max(y_alt[:, cov // 2 : cov // 2 + alt_len, :], dim=1).values[:, None, :],
                y_alt[:, cov // 2 + alt_len :, :],
            ],
            dim=1,
        )
    return y_alt


def compute_delta_score(
    models: Sequence[torch.nn.Module],
    x_ref: str,
    x_alt: str,
    strand: str,
    rbp_tensor: torch.Tensor,
    ref_len: int,
    alt_len: int,
    cov: int,
    target: str,
) -> Tuple[torch.Tensor, int]:
    x_ref_arr = one_hot_encode(x_ref)[None, :].transpose(0, 2, 1)
    x_alt_arr = one_hot_encode(x_alt)[None, :].transpose(0, 2, 1)
    x_ref_tensor = torch.tensor(x_ref_arr, dtype=torch.float32, device=rbp_tensor.device)
    x_alt_tensor = torch.tensor(x_alt_arr, dtype=torch.float32, device=rbp_tensor.device)

    if strand == "-":
        x_ref_tensor = torch.flip(x_ref_tensor, dims=[1, 2])
        x_alt_tensor = torch.flip(x_alt_tensor, dims=[1, 2])

    y_ref_preds = [model_predict_proba(m, x_ref_tensor, rbp_tensor) for m in models]
    y_alt_preds = [model_predict_proba(m, x_alt_tensor, rbp_tensor) for m in models]
    y_ref = torch.mean(torch.stack(y_ref_preds), dim=0)
    y_alt = torch.mean(torch.stack(y_alt_preds), dim=0)

    y_ref = y_ref.permute(0, 2, 1)
    y_alt = y_alt.permute(0, 2, 1)

    if strand == "-":
        y_ref = torch.flip(y_ref, dims=[1])
        y_alt = torch.flip(y_alt, dims=[1])

    y_alt = apply_indel_adjustment(y_alt, ref_len, alt_len, cov)
    channel = 1 if target in ("DS_AG", "DS_AL") else 2

    if target in ("DS_AG", "DS_DG"):
        delta = y_alt[:, :, channel] - y_ref[:, :, channel]
    else:
        delta = y_ref[:, :, channel] - y_alt[:, :, channel]

    ds, idx = torch.max(delta, dim=1)
    return ds.squeeze(0), int(idx.item())


def compute_attribution(
    ann: Annotator,
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    gene: str,
    rbp_tensor: torch.Tensor,
    dist_var: int,
    flanking_size: int,
    use_gene_padding: bool,
    target: str,
) -> Tuple[float, int, np.ndarray]:
    genes, strands, idxs = ann.get_name_and_strand(chrom, pos)
    if len(idxs) == 0:
        raise ValueError("No overlapping genes at this variant position.")
    try:
        idx = list(genes).index(gene)
    except ValueError as exc:
        raise ValueError(f"Gene {gene} not found among overlaps: {list(genes)}") from exc

    strand = strands[idx]
    dist_ann = ann.get_pos_data(idxs[idx], pos) if use_gene_padding else None
    x_ref, x_alt, cov = prepare_sequences(
        ann, chrom, pos, ref, alt, dist_var, flanking_size, dist_ann
    )

    rbp_tensor = rbp_tensor.clone().detach().requires_grad_(True)
    ds, idx = compute_delta_score(
        ann.models,
        x_ref,
        x_alt,
        strand,
        rbp_tensor,
        len(ref),
        len(alt),
        cov,
        target,
    )
    ds.backward()
    grad = rbp_tensor.grad.detach().cpu().numpy().reshape(-1)
    return float(ds.item()), idx - cov // 2, grad


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--variant", required=True, help="Variant like chr17:28369751:G>GC")
    parser.add_argument("--gene", required=True, help="Gene symbol to select when multiple genes overlap.")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pt/.pth).")
    parser.add_argument("--ref-genome", required=True, help="Reference genome FASTA.")
    parser.add_argument("--annotation", required=True, help="Annotation table such as data/grch38_chr.txt.")
    parser.add_argument("--flanking-size", type=int, required=True, help="Flanking size (80/400/2000/10000).")
    parser.add_argument("--rbp-expression", required=True, help="Condition vector JSON/NPY/NPZ.")
    parser.add_argument("--film-strengths", default="1", help="Comma-separated list, e.g. 1,5.")
    parser.add_argument("--distance", type=int, default=50, help="Max distance for delta-score window.")
    parser.add_argument("--top-k", type=int, default=20, help="Show top K features.")
    parser.add_argument("--rbp-only", action="store_true", help="Restrict output to a provided curated RBP list.")
    parser.add_argument("--rbp-list", type=Path, help="Optional text file with one RBP gene name per line.")
    parser.add_argument(
        "--no-gene-padding",
        action="store_false",
        dest="use_gene_padding",
        help="Disable transcript boundary padding (default uses gene padding to match variant scoring).",
    )
    parser.set_defaults(use_gene_padding=True)
    parser.add_argument(
        "--target",
        choices=["DS_AG", "DS_AL", "DS_DG", "DS_DL"],
        default="DS_DG",
        help="Delta-score target to attribute (default: DS_DG).",
    )
    parser.add_argument(
        "--rank-metric",
        choices=["gradxinput", "grad"],
        default="gradxinput",
        help="Rank by grad*input or by grad alone.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["abs", "signed"],
        default="abs",
        help="Rank by absolute value or signed value of the selected metric.",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending (useful to surface the most negative signed contributions).",
    )
    parser.add_argument("--output", help="Optional output TSV path.")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gradient-based attribution for a single variant under FiLM conditioning."
    )
    return add_arguments(parser)


def run(args: argparse.Namespace):
    parser = build_parser()

    if args.rbp_only and args.rbp_list is None:
        parser.error("--rbp-only requires --rbp-list.")

    chrom, pos, ref, alt = parse_variant(args.variant)
    rbp_expr = load_rbp_expression(args.rbp_expression)
    rbp_names = rbp_expr.names
    if rbp_names is None:
        rbp_names = [f"feature_{i}" for i in range(rbp_expr.dim)]

    ann = Annotator(
        args.ref_genome,
        args.annotation,
        model_path=args.model,
        model_type="pytorch",
        CL=args.flanking_size,
        rbp_context={
            "tensor": torch.tensor(rbp_expr.values, dtype=torch.float32).unsqueeze(0),
            "names": rbp_expr.names,
        },
        film_strength=1.0,
    )

    for model in ann.models:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

    if ann.rbp_tensor is None:
        raise ValueError("Model requires FiLM conditioning but no RBP tensor is available.")
    rbp_tensor = ann.rbp_tensor

    rbp_filter = None
    if args.rbp_only:
        rbp_filter = set(load_name_list(args.rbp_list))

    strengths = [float(x) for x in args.film_strengths.split(",") if x.strip()]
    target_label = args.target.split("_", 1)[1].lower()
    score_key = f"ds_{target_label}"
    dp_key = f"dp_{target_label}"

    rows = []
    for strength in strengths:
        for model in ann.models:
            if hasattr(model, "film_strength"):
                model.film_strength = strength
        ds, dp, grad = compute_attribution(
            ann,
            chrom,
            pos,
            ref,
            alt,
            args.gene,
            rbp_tensor,
            args.distance,
            args.flanking_size,
            args.use_gene_padding,
            args.target,
        )
        gradxinput = grad * rbp_expr.values
        for name, value, g, gxi in zip(rbp_names, rbp_expr.values, grad, gradxinput):
            if rbp_filter is not None and name not in rbp_filter:
                continue
            rows.append(
                {
                    "film_strength": strength,
                    "feature": name,
                    "value": float(value),
                    "grad": float(g),
                    "abs_grad": float(abs(g)),
                    "grad_x_input": float(gxi),
                    "abs_grad_x_input": float(abs(gxi)),
                    score_key: ds,
                    dp_key: dp,
                }
            )

    def metric_value(row):
        if args.rank_metric == "grad":
            return row["abs_grad"] if args.sort_by == "abs" else row["grad"]
        return row["abs_grad_x_input"] if args.sort_by == "abs" else row["grad_x_input"]

    print(f"[INFO] Variant {args.variant} gene={args.gene}")
    for strength in strengths:
        rows_strength = [row for row in rows if row["film_strength"] == strength]
        rows_strength_sorted = sorted(
            rows_strength,
            key=metric_value,
            reverse=not args.ascending,
        )
        top_rows = rows_strength_sorted[: args.top_k]
        metric_label = "grad*input" if args.rank_metric == "gradxinput" else "grad"
        sort_label = f"abs({metric_label})" if args.sort_by == "abs" else metric_label
        direction_label = "asc" if args.ascending else "desc"
        print(
            f"[INFO] film_strength={strength} target={args.target} "
            f"top {args.top_k} by {sort_label} ({direction_label})"
        )
        for row in top_rows:
            print(
                f"film={row['film_strength']}\t{row['feature']}\tvalue={row['value']:.4f}\t"
                f"grad={row['grad']:.6f}\tgrad*x={row['grad_x_input']:.6f}\t"
                f"{score_key}={row[score_key]:.4f}\t{dp_key}={row[dp_key]}"
            )

    if args.output and rows:
        import csv

        rows_sorted = sorted(rows, key=metric_value, reverse=not args.ascending)
        rows_sorted = sorted(rows_sorted, key=lambda row: row["film_strength"])
        with open(args.output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_sorted[0].keys()))
            writer.writeheader()
            writer.writerows(rows_sorted)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
