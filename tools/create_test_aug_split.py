#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from dataloader import build_dataset  # noqa: E402

ALL_DATASETS = [
    "dsb2018",
    "monuseg",
    "rus",
    "nuinsseg",
    "isic2017",
    "kvasirseg",
    "clinicdb",
    "drive",
    "promise12",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate augmented test splits for all datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        help="Datasets to process (default: all supported).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Base split to augment (default: test).",
    )
    return parser.parse_args()


def ensure_augmented(dataset: str, split: str) -> None:
    target_split = f"{split}_aug"
    print(f"[create_test_aug_split] Preparing {dataset}:{target_split} ...", end=" ")
    try:
        build_dataset(
            dataset_name=dataset,
            split=target_split,
            image_size=None,
            augment=False,
            root=None,
            ignore_index=255,
        )
    except Exception as exc:  # pragma: no cover - CLI utility
        print(f"FAILED ({exc})")
        return
    print("done")


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        ensure_augmented(dataset.lower(), args.split.lower())


if __name__ == "__main__":
    main()
