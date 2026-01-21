from __future__ import annotations

import csv
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms.functional as TF

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

VOC_CLASS_NAMES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def _ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at {path}")


def _resolve_voc_root(root: Optional[Path]) -> Path:
    candidates: Iterable[Path]
    if root:
        candidates = (root,)
    else:
        candidates = (
            Path("datasets/voc"),
            Path("datasets/VOC"),
            Path("datasets/VOCdevkit"),
        )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "JPEGImages").exists():
            return candidate
        if (candidate / "VOC2012" / "JPEGImages").exists():
            return candidate / "VOC2012"
        if (candidate / "VOCdevkit" / "VOC2012" / "JPEGImages").exists():
            return candidate / "VOCdevkit" / "VOC2012"
    raise FileNotFoundError(
        "Unable to locate VOC dataset root. "
        "Expected directories such as datasets/voc/VOCdevkit/VOC2012."
    )


def _resolve_camvid_root(root: Optional[Path]) -> Path:
    candidates: Iterable[Path]
    if root:
        candidates = (root,)
    else:
        candidates = (Path("datasets/camvid"), Path("datasets/CamVid"))
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        required = ["train", "val", "test", "train_labels", "val_labels", "test_labels"]
        if all((candidate / name).exists() for name in required):
            return candidate
    raise FileNotFoundError(
        "Unable to locate CamVid dataset root. "
        "Expected datasets/camvid with train/, val/, test/ and *_labels/ subfolders."
    )


def _resolve_dsb_root(root: Optional[Path]) -> Path:
    candidates: Iterable[Path]
    if root:
        candidates = (root,)
    else:
        candidates = (Path("datasets/dsb2018"), Path("datasets/DSB"))
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "stage1_train").exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate DSB2018 dataset root. Expected stage1_train directory "
        "under datasets/dsb2018."
    )


def _resolve_monuseg_root(root: Optional[Path]) -> Path:
    candidates: Iterable[Path]
    if root:
        candidates = (root,)
    else:
        candidates = (
            Path("datasets/monuseg"),
            Path("datasets/MoNuSeg"),
            Path("datasets/MoNuSeg/MoNuSeg 2018 Training Data"),
        )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "Tissue Images").exists() and (candidate / "Annotations").exists():
            if (candidate / "Tissue Images").is_dir():
                return candidate
            if (candidate.parent / "Tissue Images").exists():
                return candidate.parent
    raise FileNotFoundError(
        "Unable to locate MoNuSeg dataset root. Expected 'Tissue Images' and "
        "'Annotations' directories under datasets/MoNuSeg."
    )


def _split_cases(
    items: Sequence[Path], val_ratio: float, test_ratio: float
) -> Dict[str, List[Path]]:
    if not items:
        raise RuntimeError("No samples available to split.")
    items = sorted(items)
    total = len(items)
    val_count = int(total * val_ratio) if val_ratio > 0 else 0
    test_count = int(total * test_ratio) if test_ratio > 0 else 0
    if val_count + test_count >= total:
        remaining = max(total - 2, 1)
        val_count = min(val_count, remaining)
        test_count = min(test_count, remaining - val_count)
    train_count = total - val_count - test_count
    if train_count <= 0:
        train_count = max(1, total - (val_count + test_count))
    start_val = train_count
    start_test = start_val + val_count
    splits = {
        "train": list(items[:train_count]) or [items[0]],
        "val": list(items[start_val:start_test]) or [items[min(start_val, total - 1)]],
        "test": list(items[start_test:]) or [items[-1]],
    }
    return splits


def _list_image_files(directory: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for path in directory.iterdir():
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
            files[path.stem] = path
    return files


def _match_image_label_paths(image_dir: Path, label_dir: Path) -> List[Tuple[Path, Path]]:
    img_files = _list_image_files(image_dir)
    lbl_files = _list_image_files(label_dir)
    common = sorted(set(img_files).intersection(lbl_files))
    if not common:
        raise RuntimeError(
            f"No overlapping image/label stems between {image_dir} and {label_dir}."
        )
    return [(img_files[stem], lbl_files[stem]) for stem in common]


def _match_camvid_paths(image_dir: Path, label_dir: Path) -> List[Tuple[Path, Path]]:
    img_files = _list_image_files(image_dir)
    lbl_files: Dict[str, Path] = {}
    for path in label_dir.iterdir():
        if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
            continue
        stem = path.stem
        if stem.endswith("_L"):
            stem = stem[:-2]
        lbl_files[stem] = path
    common = sorted(set(img_files).intersection(lbl_files))
    if not common:
        raise RuntimeError(
            f"No overlapping image/label stems between {image_dir} and {label_dir}. "
            "CamVid labels typically use a '_L' suffix; please verify the dataset."
        )
    return [(img_files[stem], lbl_files[stem]) for stem in common]


def _apply_shared_transforms(
    image: Image.Image,
    mask: Image.Image,
    image_size: Optional[Tuple[int, int]],
    augment: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if augment and random.random() < 0.5:
        image = ImageOps.mirror(image)
        mask = ImageOps.mirror(mask)
    if image_size:
        image = image.resize(image_size, Image.BILINEAR)
        mask = mask.resize(image_size, Image.NEAREST)
    image_tensor = TF.to_tensor(image)
    mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))
    return image_tensor, mask_tensor


class _BaseSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[Path, Path]],
        mask_loader: Callable[[Path], Image.Image],
        image_size: Optional[Tuple[int, int]],
        augment: bool,
    ) -> None:
        self.samples = list(samples)
        self.mask_loader = mask_loader
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        mask = self.mask_loader(label_path)
        return _apply_shared_transforms(image, mask, self.image_size, self.augment)


class VOCSegmentationDataset(_BaseSegmentationDataset):
    num_classes: int = 21

    def __init__(
        self,
        root: Optional[Path],
        split: str,
        image_size: Optional[Tuple[int, int]] = None,
        augment: bool = False,
    ) -> None:
        voc_root = _resolve_voc_root(root)
        image_sets = voc_root / "ImageSets" / "Segmentation" / f"{split}.txt"
        _ensure_exists(image_sets, f"VOC split file {split}")
        with open(image_sets, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        if not ids:
            raise RuntimeError(f"No entries found in {image_sets}")
        samples: List[Tuple[Path, Path]] = []
        for sample_id in ids:
            img_path = voc_root / "JPEGImages" / f"{sample_id}.jpg"
            if not img_path.exists():
                img_path = voc_root / "JPEGImages" / f"{sample_id}.png"
            mask_path = voc_root / "SegmentationClass" / f"{sample_id}.png"
            _ensure_exists(img_path, f"VOC image {sample_id}")
            _ensure_exists(mask_path, f"VOC mask {sample_id}")
            samples.append((img_path, mask_path))
        super().__init__(
            samples=samples,
            mask_loader=lambda p: Image.open(p).convert("L"),
            image_size=image_size,
            augment=augment,
        )
        self.class_names = VOC_CLASS_NAMES


@dataclass
class CamVidMetadata:
    class_names: List[str]
    color_to_index: Dict[Tuple[int, int, int], int]


def _load_camvid_metadata(csv_path: Path) -> CamVidMetadata:
    _ensure_exists(csv_path, "CamVid class_dict.csv")
    class_names: List[str] = []
    color_to_index: Dict[Tuple[int, int, int], int] = {}
    with open(csv_path, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or not {"name", "r", "g", "b"}.issubset(
            set(name.strip() for name in reader.fieldnames)
        ):
            raise RuntimeError("CamVid class_dict.csv must contain name,r,g,b headers.")
        for idx, row in enumerate(reader):
            if not row:
                continue
            name = row.get("name", "").strip()
            if not name:
                continue
            class_names.append(name)
            color = (
                int(row.get("r", "0").strip()),
                int(row.get("g", "0").strip()),
                int(row.get("b", "0").strip()),
            )
            color_to_index[color] = idx
    if not class_names:
        raise RuntimeError("No classes defined in CamVid class_dict.csv.")
    return CamVidMetadata(class_names=class_names, color_to_index=color_to_index)


def _camvid_mask_loader_factory(
    mapping: CamVidMetadata, ignore_index: int
) -> Callable[[Path], Image.Image]:
    default_value = ignore_index if ignore_index >= 0 else 0

    def _loader(path: Path) -> Image.Image:
        rgb = Image.open(path).convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)
        codes = (
            arr[..., 0].astype(np.int64) * 256 * 256
            + arr[..., 1].astype(np.int64) * 256
            + arr[..., 2].astype(np.int64)
        )
        encoded = np.full(codes.shape, fill_value=default_value, dtype=np.int64)
        for color, class_idx in mapping.color_to_index.items():
            color_code = color[0] * 256 * 256 + color[1] * 256 + color[2]
            encoded[codes == color_code] = class_idx
        encoded = np.clip(encoded, 0, 255).astype(np.uint8)
        return Image.fromarray(encoded, mode="L")

    return _loader


class CamVidDataset(_BaseSegmentationDataset):
    def __init__(
        self,
        root: Optional[Path],
        split: str,
        image_size: Optional[Tuple[int, int]] = None,
        augment: bool = False,
        ignore_index: int = 255,
    ) -> None:
        camvid_root = _resolve_camvid_root(root)
        metadata = _load_camvid_metadata(camvid_root / "class_dict.csv")
        image_dir = camvid_root / split
        label_dir = camvid_root / f"{split}_labels"
        _ensure_exists(image_dir, f"CamVid image directory for split {split}")
        _ensure_exists(label_dir, f"CamVid label directory for split {split}")
        samples = _match_camvid_paths(image_dir, label_dir)
        mask_loader = _camvid_mask_loader_factory(metadata, ignore_index)
        super().__init__(
            samples=samples,
            mask_loader=mask_loader,
            image_size=image_size,
            augment=augment,
        )
        self.class_names = metadata.class_names
        self.num_classes = len(self.class_names)


class DSB2018Dataset(Dataset):
    class_names = ["background", "nucleus"]
    num_classes = 2

    def __init__(
        self,
        root: Optional[Path],
        split: str,
        image_size: Optional[Tuple[int, int]] = None,
        augment: bool = False,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> None:
        dsb_root = _resolve_dsb_root(root)
        train_root = dsb_root / "stage1_train"
        _ensure_exists(train_root, "DSB2018 stage1_train directory")
        case_dirs = sorted([p for p in train_root.iterdir() if p.is_dir()])
        if not case_dirs:
            raise RuntimeError(f"No case folders found under {train_root}")
        splits = _split_cases(case_dirs, val_ratio=val_ratio, test_ratio=test_ratio)
        split_key = split.lower()
        if split_key not in splits:
            raise ValueError("split must be 'train', 'val', or 'test'")
        selected = splits[split_key]
        if not selected:
            raise RuntimeError(f"No samples for DSB2018 {split} split. Adjust ratios.")
        self.cases = selected
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.cases)

    def _load_image(self, case_dir: Path) -> Image.Image:
        image_dir = case_dir / "images"
        _ensure_exists(image_dir, f"images directory for {case_dir.name}")
        image_files = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        if not image_files:
            raise RuntimeError(f"No image files found in {image_dir}")
        return Image.open(image_files[0]).convert("RGB")

    def _load_mask(self, case_dir: Path, size: Tuple[int, int]) -> Image.Image:
        mask_dir = case_dir / "masks"
        _ensure_exists(mask_dir, f"mask directory for {case_dir.name}")
        mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        has_mask = False
        for mask_path in mask_dir.iterdir():
            if mask_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            has_mask = True
            mask_img = Image.open(mask_path).convert("L").resize(size, Image.NEAREST)
            mask_arr = (np.array(mask_img, dtype=np.uint8) > 0).astype(np.uint8)
            mask = np.maximum(mask, mask_arr)
        if not has_mask:
            raise RuntimeError(f"No mask files found in {mask_dir}")
        return Image.fromarray(mask, mode="L")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        case_dir = self.cases[idx]
        image = self._load_image(case_dir)
        mask = self._load_mask(case_dir, image.size)
        return _apply_shared_transforms(image, mask, self.image_size, self.augment)


class MoNuSegDataset(Dataset):
    class_names = ["background", "nucleus"]
    num_classes = 2

    def __init__(
        self,
        root: Optional[Path],
        split: str,
        image_size: Optional[Tuple[int, int]] = None,
        augment: bool = False,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
    ) -> None:
        monu_root = _resolve_monuseg_root(root)
        images_dir = monu_root / "Tissue Images"
        ann_dir = monu_root / "Annotations"
        _ensure_exists(images_dir, "MoNuSeg tissue images directory")
        _ensure_exists(ann_dir, "MoNuSeg annotations directory")
        samples = [p for p in images_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg"}]
        if not samples:
            raise RuntimeError(f"No MoNuSeg images found in {images_dir}")
        splits = _split_cases(samples, val_ratio=val_ratio, test_ratio=test_ratio)
        split_key = split.lower()
        if split_key not in splits:
            raise ValueError("split must be 'train', 'val', or 'test'")
        selected = splits[split_key]
        if not selected:
            raise RuntimeError(f"No MoNuSeg samples for split {split}. Adjust ratios.")
        self.samples = selected
        self.ann_dir = ann_dir
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def _load_mask(self, image_path: Path, size: Tuple[int, int]) -> Image.Image:
        ann_name = image_path.stem + ".xml"
        ann_path = self.ann_dir / ann_name
        _ensure_exists(ann_path, f"MoNuSeg annotation for {image_path.name}")
        xml_root = ET.parse(ann_path).getroot()
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        for region in xml_root.findall(".//Region"):
            vertices = region.find("Vertices")
            if vertices is None:
                continue
            coords = [
                (float(vertex.attrib["X"]), float(vertex.attrib["Y"]))
                for vertex in vertices
                if "X" in vertex.attrib and "Y" in vertex.attrib
            ]
            if len(coords) >= 3:
                draw.polygon(coords, outline=1, fill=1)
        return mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        mask = self._load_mask(image_path, image.size)
        return _apply_shared_transforms(image, mask, self.image_size, self.augment)


def _maybe_subset(dataset: Dataset, subset_size: Optional[int]) -> Dataset:
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    indices = list(range(subset_size))
    return Subset(dataset, indices)


def build_dataset(
    dataset_name: str,
    split: str,
    image_size: Optional[Tuple[int, int]],
    augment: bool,
    root: Optional[str],
    ignore_index: int,
) -> Tuple[Dataset, int, List[str]]:
    dataset_name = dataset_name.lower()
    split = split.lower()
    root_path = Path(root) if root else None
    if dataset_name == "voc":
        if split not in {"train", "val"}:
            raise ValueError("VOC supports 'train' and 'val' splits.")
        dataset = VOCSegmentationDataset(
            root=root_path, split=split, image_size=image_size, augment=augment
        )
    elif dataset_name == "camvid":
        if split not in {"train", "val", "test"}:
            raise ValueError("CamVid supports 'train', 'val', or 'test' splits.")
        dataset = CamVidDataset(
            root=root_path,
            split=split,
            image_size=image_size,
            augment=augment,
            ignore_index=ignore_index,
        )
    elif dataset_name == "dsb2018":
        if split not in {"train", "val", "test"}:
            raise ValueError("DSB2018 supports 'train', 'val', or 'test' splits.")
        dataset = DSB2018Dataset(
            root=root_path,
            split=split,
            image_size=image_size,
            augment=augment,
        )
    elif dataset_name == "monuseg":
        if split not in {"train", "val", "test"}:
            raise ValueError("MoNuSeg supports 'train', 'val', or 'test' splits.")
        dataset = MoNuSegDataset(
            root=root_path,
            split=split,
            image_size=image_size,
            augment=augment,
        )
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")
    num_classes = getattr(dataset, "num_classes", None)
    class_names = getattr(dataset, "class_names", None)
    if num_classes is None or class_names is None:
        raise RuntimeError(f"Dataset {dataset_name} must define num_classes and class_names.")
    return dataset, num_classes, class_names


def build_dataloaders(
    dataset_name: str,
    batch_size: int,
    image_size: Optional[Tuple[int, int]],
    num_workers: int,
    pin_memory: bool,
    root: Optional[str],
    ignore_index: int,
    train_subset: Optional[int] = None,
    val_subset: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, int, List[str]]:
    train_dataset, num_classes, class_names = build_dataset(
        dataset_name=dataset_name,
        split="train",
        image_size=image_size,
        augment=True,
        root=root,
        ignore_index=ignore_index,
    )
    val_dataset, _, _ = build_dataset(
        dataset_name=dataset_name,
        split="val",
        image_size=image_size,
        augment=False,
        root=root,
        ignore_index=ignore_index,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(f"{dataset_name} dataset is empty. Please verify the data root.")

    train_loader = DataLoader(
        _maybe_subset(train_dataset, train_subset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        _maybe_subset(val_dataset, val_subset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, num_classes, class_names


def build_split_dataloader(
    dataset_name: str,
    split: str,
    batch_size: int,
    image_size: Optional[Tuple[int, int]],
    num_workers: int,
    pin_memory: bool,
    root: Optional[str],
    ignore_index: int,
    subset: Optional[int] = None,
    shuffle: bool = False,
) -> Tuple[DataLoader, int, List[str]]:
    dataset, num_classes, class_names = build_dataset(
        dataset_name=dataset_name,
        split=split,
        image_size=image_size,
        augment=shuffle,
        root=root,
        ignore_index=ignore_index,
    )
    loader = DataLoader(
        _maybe_subset(dataset, subset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return loader, num_classes, class_names
