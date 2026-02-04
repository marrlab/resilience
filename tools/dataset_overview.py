#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from dataloader import build_dataset

DATASETS = [
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

SUMMARY_PATHS = {
    "dsb2018": "runs/unc_summary_dsb2018_test.json",
    "monuseg": "runs/unc_summary_monuseg_test.json",
    "rus": "runs/unc_summary_rus_test.json",
    "nuinsseg": "runs/unc_summary_nuinsseg_test.json",
    "isic2017": "runs/unc_summary_isic_test.json",
    "kvasirseg": "runs/unc_summary_kvasirseg_test.json",
    "clinicdb": "runs/unc_summary_clinicdb_test.json",
    "drive": "runs/unc_summary_drive_test.json",
    "promise12": "runs/unc_summary_promise12_test.json",
}


def dataset_counts(name: str) -> dict[str, int | str]:
    counts: dict[str, int | str] = {}
    for split in ("train", "val", "test"):
        try:
            dataset, _, _ = build_dataset(
                dataset_name=name,
                split=split,
                image_size=None,
                augment=False,
                root=None,
                ignore_index=255,
            )
            counts[split] = len(dataset)
        except Exception as exc:
            counts[split] = f"ERR: {exc}"
    return counts


def dataset_metrics(name: str) -> dict | str:
    summary_path = SUMMARY_PATHS.get(name)
    if not summary_path:
        return "N/A"
    path = Path(summary_path)
    if not path.exists():
        return "N/A"
    data = json.loads(path.read_text())
    key = next((k for k in data if k.startswith(f"{name}:test")), None)
    if not key:
        return "N/A"
    entry = data[key]
    return {
        "dice_at_80": entry.get("dice_at_80"),
        "dice_at_90": entry.get("dice_at_90"),
        "aurc": entry.get("aurc"),
    }


def main() -> None:
    report = []
    for dataset in DATASETS:
        report.append(
            {
                "dataset": dataset,
                "counts": dataset_counts(dataset),
                "test_metrics": dataset_metrics(dataset),
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
