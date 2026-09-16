"""Memory-bounded VIGOR retrieval, reranking, Hit Rate and localization evaluation.

Official VIGOR Hit Rate masks the three semi-positive references, then asks
whether the primary positive outranks every unrelated gallery image. It is
intentionally different from "top-1 belongs to any of four tiles".
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from unifygeo.utils.data import PresetIndexSampler


METER_PER_PIXEL_AT_640 = {
    "NewYork": 0.113248,
    "Seattle": 0.100817,
    "SanFrancisco": 0.118141,
    "Chicago": 0.111262,
}
DISTANCE_THRESHOLDS_M = (1, 3, 5, 10, 20)


def autocast_context():
    return torch.amp.autocast("cuda")


def cleanup_cuda(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def loader_kwargs(batch_size: int, workers: int) -> dict:
    result = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": False,
    }
    if workers > 0:
        result["prefetch_factor"] = 1
    return result


def _extract_one_view(model, loader, device: str, view: str):
    """Write one view directly into a contiguous single-GPU feature bank."""
    count = len(loader.dataset)
    feature_bank = None
    label_bank = None
    offset = 0
    feature_keys = (
        ("sat_global_feats", "sat_global_descriptor")
        if view == "reference"
        else ("grd_global_feats", "grd_global_descriptor")
    )
    description = "Reference global features" if view == "reference" else "Query global features"
    with torch.no_grad():
        for batch in tqdm(loader, desc=description):
            images, labels = batch[:2]
            images = images.to(device, non_blocking=True)
            with autocast_context():
                outputs = model(images)
                try:
                    features = next(outputs[key] for key in feature_keys if key in outputs)
                except StopIteration as exc:
                    raise KeyError(
                        f"Model output has none of the expected {view} keys {feature_keys}; "
                        f"available keys: {sorted(outputs)}"
                    ) from exc
                features = F.normalize(features.float(), dim=-1)
            if feature_bank is None:
                feature_bank = torch.empty(
                    (count, features.shape[-1]), dtype=torch.float32, device=device
                )
                label_bank = torch.empty(
                    (count, *labels.shape[1:]), dtype=torch.int64, device="cpu"
                )
            end = offset + len(labels)
            feature_bank[offset:end].copy_(features, non_blocking=False)
            label_bank[offset:end].copy_(labels.long())
            offset = end
            del images, features
    if offset != count or feature_bank is None or label_bank is None:
        raise RuntimeError(f"Incomplete {view} extraction: wrote {offset} of {count} samples")
    return feature_bank, label_bank


def extract_global_features(model, reference_loader, query_loader, device: str) -> dict:
    """Extract both views into preallocated banks on the model GPU."""
    model.eval()
    reference_features, reference_labels = _extract_one_view(
        model, reference_loader, device, "reference"
    )
    query_features, query_labels = _extract_one_view(
        model, query_loader, device, "query"
    )
    return {
        "reference_features": reference_features,
        "reference_labels": reference_labels,
        "query_features": query_features,
        "query_labels": query_labels,
    }


def reference_indices(query_labels: torch.Tensor, reference_labels: torch.Tensor) -> torch.Tensor:
    label_to_index = {int(label): i for i, label in enumerate(reference_labels.tolist())}
    try:
        return torch.tensor(
            [[label_to_index[int(label)] for label in row] for row in query_labels.tolist()],
            dtype=torch.int64,
        )
    except KeyError as exc:
        raise RuntimeError(f"Query label {exc.args[0]} is absent from the gallery") from exc


def official_hit_rate_flags(
    candidate_indices,
    candidate_scores,
    positive_indices,
    fallback_flags=None,
) -> np.ndarray:
    """Apply the official VIGOR Hit Rate rule to a ranked candidate set.

    Primary and three semi-positive references are all excluded from the
    unrelated set. A hit occurs iff no unrelated candidate has a score
    *strictly greater* than the primary score. ``fallback_flags`` is used only
    when the primary is outside the candidate set; for top-5 this preserves
    the full-gallery result for pathological exact-score ties.
    """
    candidates = np.asarray(candidate_indices)
    scores = np.asarray(candidate_scores)
    positives = np.asarray(positive_indices)
    if candidates.shape != scores.shape:
        raise ValueError("Candidate indices and scores must have identical shapes")
    if positives.ndim != 2 or positives.shape[0] != candidates.shape[0] or positives.shape[1] != 4:
        raise ValueError("Each query must provide one primary and three semi-positive indices")
    result = (
        np.zeros(candidates.shape[0], dtype=bool)
        if fallback_flags is None
        else np.asarray(fallback_flags, dtype=bool).copy()
    )
    primary = positives[:, 0]
    membership = candidates == primary[:, None]
    selected = np.flatnonzero(membership.any(axis=1))
    if selected.size == 0:
        return result
    primary_columns = np.argmax(membership[selected], axis=1)
    primary_scores = scores[selected, primary_columns]
    known_positive = (
        candidates[selected, :, None] == positives[selected, None, :]
    ).any(axis=2)
    higher_unrelated = (scores[selected] > primary_scores[:, None]) & ~known_positive
    result[selected] = ~higher_unrelated.any(axis=1)
    return result


def compute_initial_ranking(features, device: str, query_chunk: int, num_candidates: int) -> dict:
    """Compute ranks and top-k without materializing the full Q x R matrix."""
    queries = features["query_features"]
    references = features["reference_features"]
    positive_indices = reference_indices(features["query_labels"], features["reference_labels"])
    primary_indices = positive_indices[:, 0]
    query_count = len(queries)
    ranks = torch.empty(query_count, dtype=torch.int64)
    semi_above_primary = torch.empty(query_count, dtype=torch.int64)
    topk_indices = torch.empty((query_count, num_candidates), dtype=torch.int64)
    topk_scores = torch.empty((query_count, num_candidates), dtype=torch.float32)
    with torch.no_grad():
        for start in tqdm(range(0, query_count, query_chunk), desc="Chunked retrieval ranking"):
            end = min(start + query_chunk, query_count)
            scores = queries[start:end] @ references.T
            local_primary = primary_indices[start:end].to(device)
            rows = torch.arange(end - start, device=device)
            primary_scores = scores[rows, local_primary]
            ranks[start:end] = (1 + (scores > primary_scores[:, None]).sum(dim=1)).cpu()
            local_semi = positive_indices[start:end, 1:].to(device)
            semi_scores = torch.gather(scores, 1, local_semi)
            semi_above_primary[start:end] = (semi_scores > primary_scores[:, None]).sum(dim=1).cpu()
            values, indices = torch.topk(scores, k=num_candidates, dim=1, sorted=True)
            topk_indices[start:end] = indices.cpu()
            topk_scores[start:end] = values.float().cpu()
            del scores
    full_gallery_paper_hit = ranks == (semi_above_primary + 1)
    paper_hit = torch.from_numpy(
        official_hit_rate_flags(
            topk_indices.numpy(),
            topk_scores.numpy(),
            positive_indices.numpy(),
            fallback_flags=full_gallery_paper_hit.numpy(),
        )
    )
    if not torch.equal(paper_hit, full_gallery_paper_hit):
        raise AssertionError(
            "Top-5 and full-gallery official Hit Rate disagree; check score ties or labels"
        )
    top1_any_positive = (topk_indices[:, :1] == positive_indices).any(dim=1)
    return {
        "ranks": ranks,
        "positive_indices": positive_indices,
        "topk_indices": topk_indices,
        "topk_scores": topk_scores,
        "paper_hit": paper_hit,
        "top1_any_positive": top1_any_positive,
    }


def ground_rerank_descriptors(model, images: torch.Tensor) -> torch.Tensor:
    pyramid = model.grd_encoder(images)
    volume = model.grd_local_feats_decoder(pyramid[-1])
    decoder = model.pyramid_decoder
    descriptors = decoder.grd_feature_to_descriptor1(volume)
    if decoder.match_pre_norm:
        descriptors = decoder.grd_descriptor_norm1(descriptors)
    return F.normalize(descriptors, p=2, dim=1)


def satellite_rerank_maps(model, images: torch.Tensor) -> torch.Tensor:
    pyramid = model.sat_encoder(images)
    feature_map = model.sat_local_feats_decoder(pyramid[-1])
    decoder = model.pyramid_decoder
    if decoder.match_pre_norm:
        feature_map = decoder.sat_matching_block_norm1(feature_map)
    return F.normalize(feature_map, p=2, dim=1)


def compute_all_query_rerank_scores(
    model,
    ranking,
    query_dataset,
    reference_dataset,
    device: str,
    query_batch_size: int,
    reference_batch_size: int,
    workers: int,
) -> torch.Tensor:
    """Compute formal local scores for every query top-k, reusing repeated references."""
    model.eval()
    candidates = ranking["topk_indices"]
    num_candidates = candidates.shape[1]
    candidate_flat = candidates.numpy().reshape(-1)
    unique_references, inverse = np.unique(candidate_flat, return_inverse=True)
    query_loader = DataLoader(
        query_dataset, shuffle=False,
        **loader_kwargs(query_batch_size, workers),
    )
    reference_loader = DataLoader(
        reference_dataset,
        sampler=PresetIndexSampler(unique_references.tolist()),
        **loader_kwargs(reference_batch_size, workers),
    )
    print(
        f"Rerank every query: {candidate_flat.size:,} pairs, "
        f"{unique_references.size:,} unique satellite images"
    )
    ground_chunks = []
    with torch.no_grad():
        for images, _, _ in tqdm(query_loader, desc="Ground rerank descriptors"):
            images = images.to(device, non_blocking=True)
            with autocast_context():
                ground_chunks.append(ground_rerank_descriptors(model, images).cpu())
    ground_descriptors = torch.cat(ground_chunks)
    occurrence_order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=len(unique_references))
    offsets = np.concatenate(([0], np.cumsum(counts)))
    scores_flat = torch.empty(candidate_flat.size, dtype=torch.float32)
    unique_start = 0
    with torch.no_grad():
        for images, _ in tqdm(reference_loader, desc="Unique satellite rerank maps"):
            images = images.to(device, non_blocking=True)
            with autocast_context():
                satellite_maps = satellite_rerank_maps(model, images)
            unique_end = unique_start + len(images)
            positions = occurrence_order[offsets[unique_start] : offsets[unique_end]]
            local_reference = torch.from_numpy(inverse[positions] - unique_start).to(device)
            query_ids = torch.from_numpy(positions // num_candidates)
            ground = ground_descriptors[query_ids].to(device, non_blocking=True)
            with autocast_context():
                scores = torch.sum(
                    ground[:, :, None, None] * satellite_maps[local_reference], dim=1
                ).flatten(1).amax(dim=1)
            scores_flat[torch.from_numpy(positions)] = scores.float().cpu()
            unique_start = unique_end
    if unique_start != len(unique_references):
        raise RuntimeError("Rerank reference loader ended prematurely")
    cleanup_cuda(ground_descriptors)
    return scores_flat.reshape_as(ranking["topk_scores"])


def apply_reranking(ranking, local_scores: torch.Tensor) -> dict:
    """Apply raw retrieval + local score fusion as a stable top-k permutation."""
    candidates = ranking["topk_indices"].numpy()
    fused = ranking["topk_scores"].float().numpy() + local_scores.float().numpy()
    order = np.argsort(-fused, axis=1, kind="stable")
    ordered = np.take_along_axis(candidates, order, axis=1)
    positives = ranking["positive_indices"].numpy()
    primary = positives[:, 0]
    base_ranks = ranking["ranks"].numpy()
    in_topk = (candidates == primary[:, None]).any(axis=1)
    post_ranks = base_ranks.copy()
    selected = np.flatnonzero(in_topk)
    primary_columns = np.argmax(candidates[selected] == primary[selected, None], axis=1)
    primary_fused_scores = fused[selected, primary_columns]
    post_ranks[selected] = 1 + np.sum(
        fused[selected] > primary_fused_scores[:, None], axis=1
    )

    post_paper_hit = official_hit_rate_flags(
        candidates,
        fused,
        positives,
        fallback_flags=ranking["paper_hit"].numpy(),
    )
    post_top1_any = (ordered[:, :1] == positives).any(axis=1)
    post_exact_top1 = post_ranks == 1
    pre_exact_top1 = base_ranks == 1
    if not np.array_equal(post_ranks[~in_topk], base_ranks[~in_topk]):
        raise AssertionError("A top-k permutation changed a positive outside top-k")
    transition = np.zeros((candidates.shape[1], candidates.shape[1]), dtype=np.int64)
    np.add.at(transition, (base_ranks[in_topk] - 1, post_ranks[in_topk] - 1), 1)
    return {
        "ordered_candidates": ordered,
        "post_ranks": post_ranks,
        "post_paper_hit": post_paper_hit,
        "post_top1_any_positive": post_top1_any,
        "pre_exact_top1": pre_exact_top1,
        "post_exact_top1": post_exact_top1,
        "rank_transition": transition,
    }


def retrieval_metrics(ranks, paper_hit, top1_any_positive, reference_count: int) -> dict:
    ranks = np.asarray(ranks)
    one_percent = max(1, reference_count // 100)
    metrics = {
        "R@1": float(np.mean(ranks <= 1) * 100),
        "R@5": float(np.mean(ranks <= min(5, reference_count)) * 100),
        "R@10": float(np.mean(ranks <= min(10, reference_count)) * 100),
        "R@1%": float(np.mean(ranks <= one_percent) * 100),
        "Hit_Rate": float(np.mean(paper_hit) * 100),
        "top1_any_of_four_diagnostic": float(np.mean(top1_any_positive) * 100),
        "R@1%_cutoff": int(one_percent),
    }
    if metrics["Hit_Rate"] + 1e-9 < metrics["R@1"]:
        raise AssertionError("Official Hit Rate cannot be below primary-positive R@1")
    return metrics


def rerank_diagnostics(ranking, reranked) -> dict:
    before = ranking["paper_hit"].numpy().astype(bool)
    after = reranked["post_paper_hit"].astype(bool)
    before_any = ranking["top1_any_positive"].numpy().astype(bool)
    after_any = reranked["post_top1_any_positive"].astype(bool)
    base_ranks = ranking["ranks"].numpy()
    return {
        "paper_hit_gained": int(np.sum(~before & after)),
        "paper_hit_lost": int(np.sum(before & ~after)),
        "paper_hit_unchanged": int(np.sum(before == after)),
        "any_positive_top1_gained": int(np.sum(~before_any & after_any)),
        "any_positive_top1_lost": int(np.sum(before_any & ~after_any)),
        "primary_rank_promoted": int(np.sum(reranked["post_ranks"] < base_ranks)),
        "primary_rank_demoted": int(np.sum(reranked["post_ranks"] > base_ranks)),
        "rank_transition_rows_before_cols_after": reranked["rank_transition"].tolist(),
    }


def evaluate_oracle_localization(
    model,
    query_dataset,
    reference_dataset,
    positive_indices,
    device: str,
    batch_size: int,
    workers: int,
) -> dict:
    """Evaluate metric localization on each query's primary-positive tile."""
    query_loader = DataLoader(
        query_dataset, shuffle=False, **loader_kwargs(batch_size, workers)
    )
    ref_loader = DataLoader(
        reference_dataset,
        sampler=PresetIndexSampler(positive_indices[:, 0].tolist()),
        **loader_kwargs(batch_size, workers),
    )
    errors, gt_positions, pred_positions = [], [], []
    model.eval()
    with torch.no_grad():
        iterator = zip(query_loader, ref_loader)
        for (query_images, _, targets), (reference_images, _) in tqdm(
            iterator, total=min(len(query_loader), len(ref_loader)), desc="Oracle-tile localization"
        ):
            query_images = query_images.to(device, non_blocking=True)
            reference_images = reference_images.to(device, non_blocking=True)
            with autocast_context():
                output, _ = model(query_images, reference_images)
            heatmaps = output["heatmap"].float().cpu()
            gt_maps = targets["src_scores"]
            width = heatmaps.shape[-1]
            pred_flat = heatmaps.flatten(1).argmax(dim=1).numpy()
            gt_flat = gt_maps.flatten(1).argmax(dim=1).numpy()
            pred_row, pred_col = pred_flat // width, pred_flat % width
            gt_row, gt_col = gt_flat // width, gt_flat % width
            pixels = np.hypot(pred_row - gt_row, pred_col - gt_col)
            for j, city in enumerate(targets["city"]):
                errors.append(pixels[j] * METER_PER_PIXEL_AT_640[city] * 640 / width)
                gt_positions.append((int(gt_row[j]), int(gt_col[j])))
                pred_positions.append((int(pred_row[j]), int(pred_col[j])))
    return {
        "errors_m": np.asarray(errors, dtype=np.float64),
        "gt_positions_rc": np.asarray(gt_positions, dtype=np.int16),
        "predicted_positions_rc": np.asarray(pred_positions, dtype=np.int16),
    }


def localization_metrics(errors_m) -> dict:
    errors_m = np.asarray(errors_m)
    return {
        "mean_error_m": float(np.mean(errors_m)),
        "median_error_m": float(np.median(errors_m)),
        "accuracy_percent": {
            f"{threshold}m": float(np.mean(errors_m < threshold) * 100)
            for threshold in DISTANCE_THRESHOLDS_M
        },
    }


def gated_lf_cvgl_metrics(errors_m, exact_top1) -> dict:
    errors_m = np.asarray(errors_m)
    exact_top1 = np.asarray(exact_top1, dtype=bool)
    return {
        "accuracy_percent": {
            f"{threshold}m": float(np.mean(exact_top1 & (errors_m < threshold)) * 100)
            for threshold in DISTANCE_THRESHOLDS_M
        },
        "definition": (
            "exact primary-positive Top-1 retrieval AND oracle-tile localization error "
            "below threshold; denominator is every query"
        ),
    }


def save_results(output_dir: Path, summary, ranking, reranked, localization) -> None:
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    np.savez_compressed(
        output_dir / "per_query_results.npz",
        primary_rank_before=ranking["ranks"].numpy(),
        primary_rank_after=reranked["post_ranks"],
        top5_before=ranking["topk_indices"].numpy(),
        top5_after=reranked["ordered_candidates"],
        paper_hit_before=ranking["paper_hit"].numpy(),
        paper_hit_after=reranked["post_paper_hit"],
        top1_any_of_four_before=ranking["top1_any_positive"].numpy(),
        top1_any_of_four_after=reranked["post_top1_any_positive"],
        oracle_localization_error_m=localization["errors_m"],
        gt_positions_rc=localization["gt_positions_rc"],
        predicted_positions_rc=localization["predicted_positions_rc"],
    )


def run_self_test() -> None:
    metrics = retrieval_metrics(
        np.array([1, 2, 10, 11]), np.array([1, 1, 1, 0]),
        np.array([1, 1, 0, 0]), 100
    )
    assert metrics["R@1"] == 25 and metrics["R@5"] == 50
    assert metrics["R@10"] == 75 and metrics["Hit_Rate"] == 75
    ranking = {
        "ranks": torch.tensor([2, 2, 7]),
        "positive_indices": torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]),
        "topk_indices": torch.tensor([[11, 10, 90, 91, 92], [90, 20, 21, 22, 23], [90, 91, 92, 93, 94]]),
        "topk_scores": torch.zeros(3, 5),
        "paper_hit": torch.tensor([True, False, False]),
        "top1_any_positive": torch.tensor([True, False, False]),
    }
    local = torch.tensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float32)
    reranked = apply_reranking(ranking, local)
    assert reranked["post_ranks"].tolist() == [1, 1, 7]
    assert reranked["post_paper_hit"].tolist() == [True, True, False]
    # An unrelated exact tie is not "strictly greater" and must therefore be
    # handled identically before and after reranking.
    tie_candidates = np.array([[90, 10, 11, 12, 13]])
    tie_positives = np.array([[10, 11, 12, 13]])
    assert official_hit_rate_flags(
        tie_candidates, np.array([[1.0, 1.0, 0.9, 0.8, 0.7]]), tie_positives
    ).tolist() == [True]
    assert official_hit_rate_flags(
        tie_candidates, np.array([[1.01, 1.0, 0.9, 0.8, 0.7]]), tie_positives
    ).tolist() == [False]
    print("Self-test passed: retrieval, official Hit Rate and pure top-5 reranking.")
