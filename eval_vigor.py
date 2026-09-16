#!/usr/bin/env python3
"""Evaluate UnifyGeo retrieval, localization, and LF-CVGL on VIGOR."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import cv2
import torch
from torch.utils.data import DataLoader, Subset

from unifygeo.datasets import VigorDatasetEval
from unifygeo.evaluation.vigor import (
    apply_reranking,
    compute_all_query_rerank_scores,
    compute_initial_ranking,
    evaluate_oracle_localization,
    extract_global_features,
    gated_lf_cvgl_metrics,
    loader_kwargs,
    localization_metrics,
    official_hit_rate_flags,
    rerank_diagnostics,
    retrieval_metrics,
    run_self_test,
    save_results,
)
from unifygeo.models import build_model
from unifygeo.utils.data import build_evaluation_transforms


PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_SCHEMA = 1
CHECKPOINTS = {
    "same-area": PROJECT_ROOT / "checkpoints" / "unifygeo_vigor_same_area.pth",
    "cross-area": PROJECT_ROOT / "checkpoints" / "unifygeo_vigor_cross_area.pth",
}
EXPECTED_CHECKPOINTS = {
    "same-area": "6443775bd39a84eb4dca04ed5db206be57a8a8fbac6e36a6c045da00a60025ff",
    "cross-area": "71175956671df210c76a830e4962730188bf0148938e597ef2641945d75cd79d",
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the final UnifyGeo model on VIGOR with one CUDA GPU."
    )
    parser.add_argument("--protocol", choices=("same-area", "cross-area"), default="same-area")
    parser.add_argument("--data-root", required=False)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--rerank-query-batch-size", type=int, default=16)
    parser.add_argument("--rerank-reference-batch-size", type=int, default=64)
    parser.add_argument("--localization-batch-size", type=int, default=4)
    parser.add_argument("--similarity-query-chunk", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-queries", type=int, help="Use only the first N queries for debugging.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = str(CHECKPOINTS[args.protocol])
    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / "outputs" / args.protocol)
    positive_fields = (
        "feature_batch_size",
        "rerank_query_batch_size",
        "rerank_reference_batch_size",
        "localization_batch_size",
        "similarity_query_chunk",
        "cpu_threads",
    )
    for field in positive_fields:
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.max_queries is not None and args.max_queries < 1:
        parser.error("--max-queries must be positive")
    if not args.self_test and not args.data_root:
        parser.error("--data-root is required unless --self-test is used")
    return args


def model_config():
    return SimpleNamespace(
        model="convnext_tiny.fb_in22k_ft_in1k_384",
        backbone_norm=None,
        grd_aggregator="soa_gem",
        sat_aggregator="soa_gem",
        aggregator_norm="LN",
        enc_dims=768,
        decoder_norm="BN",
        decoder_match_norm=True,
        train_rerank=False,
        aux_loss=True,
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def signed_cache(path, signature, enabled):
    if not enabled or not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("signature") != signature:
        raise RuntimeError(f"Refusing an unsigned or stale cache: {path}")
    print(f"Loaded validated cache: {path}")
    return payload["data"]


def save_cache(path, signature, data, enabled):
    if enabled:
        torch.save({"signature": signature, "data": data}, path)


def print_metrics(title, metrics):
    print(f"\n{title}")
    print(json.dumps(metrics, indent=2))


def build_datasets(args):
    satellite_transform, ground_transform = build_evaluation_transforms()
    references = VigorDatasetEval(
        args.data_root,
        protocol=args.protocol,
        image_type="reference",
        transform=satellite_transform,
    )
    queries = VigorDatasetEval(
        args.data_root,
        protocol=args.protocol,
        image_type="query",
        transform=ground_transform,
    )
    if args.max_queries is not None:
        queries = Subset(queries, range(min(args.max_queries, len(queries))))
    return references, queries


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if not str(args.device).startswith("cuda"):
        raise ValueError("The public evaluator supports one CUDA GPU only.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable, but this evaluator requires one CUDA GPU.")
    device_index = torch.device(args.device).index
    if device_index is not None and device_index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device is unavailable: {args.device}")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Run scripts/download_checkpoints.py first."
        )
    if not data_root.is_dir():
        raise FileNotFoundError(f"VIGOR data root not found: {data_root}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (output_dir / "run.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_handle)
    cv2.setNumThreads(0)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)

    print("Configuration:")
    print(json.dumps(vars(args), indent=2))
    checkpoint_hash = sha256_file(checkpoint)
    expected_hash = EXPECTED_CHECKPOINTS.get(args.protocol)
    if checkpoint == CHECKPOINTS[args.protocol].resolve() and checkpoint_hash != expected_hash:
        raise RuntimeError(
            f"Default {args.protocol} checkpoint hash mismatch: {checkpoint_hash}"
        )

    model = build_model(model_config())
    state_dict = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state_dict, dict) or not state_dict:
        raise TypeError("Checkpoint must be a non-empty PyTorch state_dict.")
    model.load_state_dict(state_dict, strict=True)
    tensor_count = len(state_dict)
    value_count = sum(tensor.numel() for tensor in state_dict.values())
    if tensor_count != 457 or value_count != 57_700_330:
        raise RuntimeError(
            f"Unexpected checkpoint structure: {tensor_count} tensors, {value_count} values"
        )
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        f"Strict checkpoint load passed: {tensor_count} tensors, "
        f"{value_count:,} values, {trainable_count:,} trainable parameters"
    )
    del state_dict
    gc.collect()
    model = model.to(args.device).eval()

    reference_dataset, query_dataset = build_datasets(args)
    print(f"Dataset: {len(query_dataset):,} queries, {len(reference_dataset):,} references")
    signature = {
        "schema": CACHE_SCHEMA,
        "protocol": args.protocol,
        "checkpoint_sha256": checkpoint_hash,
        "data_root": str(data_root),
        "query_count": len(query_dataset),
        "reference_count": len(reference_dataset),
        "image_size": [384, 384],
        "ground_size": [384, 768],
        "rerank_candidates": 5,
    }
    use_cache = not args.no_cache

    ranking_path = output_dir / "initial_ranking.pt"
    ranking = signed_cache(ranking_path, {**signature, "stage": "ranking"}, use_cache)
    if ranking is None:
        reference_loader = DataLoader(
            reference_dataset,
            shuffle=False,
            **loader_kwargs(args.feature_batch_size, args.num_workers),
        )
        query_loader = DataLoader(
            query_dataset,
            shuffle=False,
            **loader_kwargs(args.feature_batch_size, args.num_workers),
        )
        features = extract_global_features(model, reference_loader, query_loader, args.device)
        ranking = compute_initial_ranking(
            features,
            args.device,
            args.similarity_query_chunk,
            num_candidates=5,
        )
        save_cache(ranking_path, {**signature, "stage": "ranking"}, ranking, use_cache)
        del features, reference_loader, query_loader
        gc.collect()
        torch.cuda.empty_cache()

    audited_hit = official_hit_rate_flags(
        ranking["topk_indices"].numpy(),
        ranking["topk_scores"].numpy(),
        ranking["positive_indices"].numpy(),
        fallback_flags=ranking["paper_hit"].numpy(),
    )
    if not torch.equal(torch.from_numpy(audited_hit), ranking["paper_hit"]):
        raise AssertionError("Cached Hit Rate does not follow the official VIGOR protocol.")

    rerank_path = output_dir / "rerank_local_scores.pt"
    local_scores = signed_cache(rerank_path, {**signature, "stage": "rerank"}, use_cache)
    if local_scores is None:
        local_scores = compute_all_query_rerank_scores(
            model,
            ranking,
            query_dataset,
            reference_dataset,
            args.device,
            args.rerank_query_batch_size,
            args.rerank_reference_batch_size,
            args.num_workers,
        )
        save_cache(rerank_path, {**signature, "stage": "rerank"}, local_scores, use_cache)
    reranked = apply_reranking(ranking, local_scores)

    before = retrieval_metrics(
        ranking["ranks"].numpy(),
        ranking["paper_hit"].numpy(),
        ranking["top1_any_positive"].numpy(),
        len(reference_dataset),
    )
    after = retrieval_metrics(
        reranked["post_ranks"],
        reranked["post_paper_hit"],
        reranked["post_top1_any_positive"],
        len(reference_dataset),
    )
    for key in ("R@5", "R@10", "R@1%"):
        if abs(before[key] - after[key]) > 1e-12:
            raise AssertionError(f"Top-5 reranking changed {key}.")
    diagnostics = rerank_diagnostics(ranking, reranked)
    print_metrics("Retrieval before reranking", before)
    print_metrics("Retrieval after reranking", after)
    print_metrics("Reranking diagnostics", diagnostics)

    summary = {
        "protocol": args.protocol,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "query_count": len(query_dataset),
        "reference_count": len(reference_dataset),
        "model": {
            "backbone": "convnext_tiny.fb_in22k_ft_in1k_384",
            "state_dict_tensor_count": tensor_count,
            "state_dict_value_count": value_count,
            "trainable_parameter_count": trainable_count,
        },
        "retrieval_before_rerank": before,
        "retrieval_after_rerank": after,
        "hit_rate_protocol": {
            "primary_relevant": True,
            "three_semi_positives_masked": True,
            "unrelated_comparison": "strictly greater score (>)",
            "denominator": "all queries",
        },
        "rerank_diagnostics": diagnostics,
    }
    if args.retrieval_only:
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nRetrieval-only evaluation complete: {output_dir}")
        return

    localization_path = output_dir / "oracle_localization.pt"
    localization = signed_cache(
        localization_path,
        {**signature, "stage": "localization"},
        use_cache,
    )
    if localization is None:
        localization = evaluate_oracle_localization(
            model,
            query_dataset,
            reference_dataset,
            ranking["positive_indices"],
            args.device,
            args.localization_batch_size,
            args.num_workers,
        )
        save_cache(
            localization_path,
            {**signature, "stage": "localization"},
            localization,
            use_cache,
        )
    oracle_metrics = localization_metrics(localization["errors_m"])
    lf_before = gated_lf_cvgl_metrics(localization["errors_m"], reranked["pre_exact_top1"])
    lf_after = gated_lf_cvgl_metrics(localization["errors_m"], reranked["post_exact_top1"])
    summary["metric_localization_on_primary_tile"] = oracle_metrics
    summary["lf_cvgl_before_rerank"] = lf_before
    summary["lf_cvgl_after_rerank"] = lf_after
    print_metrics("Metric localization on the primary-positive tile", oracle_metrics)
    print_metrics("LF-CVGL before reranking", lf_before)
    print_metrics("LF-CVGL after reranking", lf_after)
    save_results(output_dir, summary, ranking, reranked, localization)
    print(f"\nEvaluation complete: {output_dir}")


if __name__ == "__main__":
    main()
