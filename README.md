# UnifyGeo

Official implementation of **A Unified Hierarchical Framework for Fine-grained Cross-view Geo-localization over Large-scale Scenarios**.

This initial release evaluates the final UnifyGeo models on the VIGOR benchmark. A single command reports image retrieval, metric localization on the primary aerial tile, and large-scale fine-grained cross-view geo-localization (LF-CVGL). Both same-area and cross-area protocols are supported.

## Evaluation pipeline

For each ground-view query, UnifyGeo:

1. extracts global descriptors for the query and the complete aerial gallery;
2. retrieves the five highest-scoring aerial candidates;
3. re-ranks those five candidates using detailed feature matching; and
4. performs metric localization on the top-ranked aerial image.

## Installation

The installation below uses Python 3.8, PyTorch 1.13.1 with CUDA 11.6 wheels, and timm 0.9.16.

```bash
git clone https://github.com/chord-sz/UnifyGeo.git
cd UnifyGeo
conda env create -f environment.yml
conda activate ug
pip install numpy==1.24.4
pip install -r requirements.txt
```

`requirements.txt` pins the PyTorch CUDA 11.6 wheels and the remaining Python packages used by the original environment.

## VIGOR dataset

Download VIGOR from the [official dataset repository](https://github.com/Jeff-Zilence/VIGOR). The evaluator expects the following layout:

```text
VIGOR/
├── Chicago/
│   ├── panorama/
│   └── satellite/
├── NewYork/
│   ├── panorama/
│   └── satellite/
├── SanFrancisco/
│   ├── panorama/
│   └── satellite/
├── Seattle/
│   ├── panorama/
│   └── satellite/
└── splits_new/
    ├── Chicago/
    ├── NewYork/
    ├── SanFrancisco/
    └── Seattle/
```

The dataset is not redistributed by this repository. Please follow the original VIGOR license and terms of use.

## Checkpoints

The final same-area and cross-area checkpoints are published as assets of the `v1.0.0-vigor-eval` GitHub Release:

```bash
python scripts/download_checkpoints.py --protocol all
```

While the repository is private, authenticate with GitHub first:

```bash
gh auth login
GITHUB_TOKEN="$(gh auth token)" python scripts/download_checkpoints.py --protocol all
```

The downloader verifies every file with SHA-256:

| Protocol | File | SHA-256 |
|---|---|---|
| Same-area | `unifygeo_vigor_same_area.pth` | `6443775bd39a84eb4dca04ed5db206be57a8a8fbac6e36a6c045da00a60025ff` |
| Cross-area | `unifygeo_vigor_cross_area.pth` | `71175956671df210c76a830e4962730188bf0148938e597ef2641945d75cd79d` |

Each checkpoint is a plain PyTorch `state_dict` containing 457 tensors and 57,700,330 values. The evaluator requires an exact `strict=True` load.

## Evaluation

Same-area:

```bash
python eval_vigor.py \
  --protocol same-area \
  --data-root /path/to/VIGOR \
  --device cuda:0
```

Cross-area:

```bash
python eval_vigor.py \
  --protocol cross-area \
  --data-root /path/to/VIGOR \
  --device cuda:0
```

The checkpoint path defaults to the matching file under `checkpoints/`. Use `--checkpoint /path/to/weights.pth` to evaluate another structurally compatible checkpoint.

The default settings target a single 24 GB GPU:

```text
--feature-batch-size 32
--rerank-query-batch-size 16
--rerank-reference-batch-size 64
--localization-batch-size 4
--similarity-query-chunk 64
--num-workers 2
```

For a quick pipeline check, add `--max-queries 32`. Results from a subset must not be reported as benchmark results. Add `--retrieval-only` to skip metric localization. Add `--no-cache` to ignore and overwrite no stage caches.

## Metric definitions

Each VIGOR query has one primary positive and three semi-positive aerial references.

- **R@K** measures whether the primary positive is ranked in the first K references.
- **Hit Rate** masks the three semi-positives and tests whether the primary positive outranks every unrelated gallery reference. The same implementation and strict-greater tie rule are used before and after re-ranking.
- **Metric localization** evaluates the predicted camera location on the known primary-positive aerial tile. It reports mean and median errors and recall below 1, 3, 5, 10, and 20 meters.
- **LF-CVGL** counts a query as correct only when the primary positive is ranked first and its metric-localization error is below the distance threshold. The denominator is the complete query set.

Re-ranking always processes the initial top five candidates without consulting ground-truth labels. It adds the detailed matching score to the global retrieval score and only permutes those five candidates. Therefore, R@5 and all larger-cutoff retrieval metrics remain unchanged.

## Outputs

By default, results are written to `outputs/<protocol>/`:

- `run.log`: configuration, checkpoint validation, and metrics;
- `summary.json`: structured aggregate results;
- `per_query_results.npz`: per-query ranks, candidates, Hit Rate flags, and localization errors;
- `initial_ranking.pt`: signed retrieval cache;
- `rerank_local_scores.pt`: signed detailed-matching cache;
- `oracle_localization.pt`: signed metric-localization cache.

Cache signatures include the protocol, checkpoint SHA-256, dataset path, sample counts, image sizes, and candidate count. A stale or unsigned cache is rejected.

## Acknowledgements

We sincerely thank the authors of [Sample4Geo](https://github.com/Skyy93/Sample4Geo) and [CCVPE](https://github.com/tudelft-iv/CCVPE) for making their code publicly available. Our work builds upon these two excellent projects.

## Citation

Please cite the [arXiv paper](https://arxiv.org/abs/2505.07622):

```bibtex
@misc{song2025unifygeo,
  title={A Unified Hierarchical Framework for Fine-grained Cross-view Geo-localization over Large-scale Scenarios},
  author={Zhuo Song and Ye Zhang and Kunhong Li and Longguang Wang and Yulan Guo},
  year={2025},
  eprint={2505.07622},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2505.07622}
}
```

## License

UnifyGeo is released under the Apache License 2.0. See `LICENSE` and `THIRD_PARTY_NOTICES` for details. The VIGOR dataset and released model checkpoints may be subject to their own terms.
