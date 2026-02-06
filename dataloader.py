from __future__ import annotations

import csv
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageEnhance, ImageChops, ImageFilter
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms.functional as TF

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_AUGMENT_SEED = 1337


def _apply_image_translation(image: Image.Image, dx: int, dy: int, fill) -> Image.Image:
    canvas = Image.new(image.mode, image.size, fill)
    canvas.paste(image, (dx, dy))
    return canvas


def _augment_image_and_mask(image: Image.Image, mask: Image.Image, rng: random.Random) -> Tuple[Image.Image, Image.Image]:
    angle = rng.uniform(-12.0, 12.0)
    image = image.rotate(angle, resample=Image.BILINEAR)
    mask = mask.rotate(angle, resample=Image.NEAREST)
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
        mask = ImageOps.mirror(mask)
    dx = rng.randint(-16, 16)
    dy = rng.randint(-16, 16)
    image = _apply_image_translation(image, dx, dy, fill=(0, 0, 0))
    mask = _apply_image_translation(mask, dx, dy, fill=0)

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.8, 1.2))

    if rng.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(radius=1.2))

    arr = np.array(image).astype(np.float32) / 255.0
    noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0.0, 0.03, arr.shape)
    arr = np.clip(arr + noise, 0.0, 1.0)
    image = Image.fromarray((arr * 255).astype(np.uint8))

    mask = (np.array(mask) > 0).astype(np.uint8)
    mask_img = Image.fromarray(mask * 255, mode="L")
    return image, mask_img

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
        possible = [candidate]
        nested = candidate / "MoNuSeg 2018 Training Data"
        if nested.exists():
            possible.append(nested)
        for root_candidate in possible:
            splits_dir = root_candidate / "splits"
            if (splits_dir / "train" / "images").exists() and (
                splits_dir / "train" / "annotations"
            ).exists():
                return root_candidate
    raise FileNotFoundError(
        "Unable to locate MoNuSeg dataset root with precomputed splits. "
        "Expected directories like datasets/MoNuSeg/.../splits/train/images."
    )


def _resolve_rus_root(root: Optional[Path]) -> Path:
    candidates: Iterable[Path]
    if root:
        candidates = (root,)
    else:
        candidates = (
            Path("datasets/US/RUS"),
            Path("datasets/us/RUS"),
            Path("datasets/US/abdominal_US/abdominal_US/RUS"),
        )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        images_dir = candidate / "images"
        ann_dir = candidate / "annotations"
        if images_dir.exists() and ann_dir.exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate the RUS ultrasound dataset root. "
        "Expected directories like datasets/US/RUS with images/ and annotations/."
    )


def _resolve_nuinsseg_root(root: Optional[Path]) -> Path:
    candidates: Iterable[Path]
    if root:
        candidates = (root,)
    else:
        candidates = (
            Path("datasets/NuInsSeg"),
            Path("datasets/nuinsseg"),
        )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        splits_dir = candidate / "splits"
        if (splits_dir / "train" / "images").exists() and (
            splits_dir / "train" / "annotations"
        ).exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate NuInsSeg splits. "
        "Run datasets/NuInsSeg/split_nuinsseg.py to create train/val/test folders."
    )


def _resolve_isic_root(root: Optional[Path]) -> Path:
    candidates: Iterable[Path]
    if root:
        candidates = (root,)
    else:
        candidates = (
            Path("datasets/isic/isic2017_task1"),
            Path("datasets/isic2017_task1"),
            Path("datasets/isic"),
        )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        train_dir = candidate / "ISIC-2017_Training_Data"
        if train_dir.exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate ISIC 2017 dataset root. "
        "Expected directories like datasets/isic/isic2017_task1 with ISIC-2017_Training_Data."
    )


def _resolve_kvasirseg_root(root: Optional[Path]) -> Path:
    candidates: Iterable[Path]
    if root:
        candidates = (root,)
    else:
        candidates = (
            Path("datasets/kvasir-seg/Kvasir-SEG"),
            Path("datasets/kvasir/Kvasir-SEG"),
        )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "splits" / "train" / "images").exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate Kvasir-SEG splits. Run datasets/kvasir-seg/split_polyps.py --dataset kvasir."
    )


def _resolve_clinicdb_root(root: Optional[Path]) -> Path:
    candidates: Iterable[Path]
    if root:
        candidates = (root,)
    else:
        candidates = (
            Path("datasets/kvasir-seg/CVC-ClinicDB"),
            Path("datasets/kvasir/CVC-ClinicDB"),
        )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "splits" / "train" / "images").exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate CVC-ClinicDB splits. Run datasets/kvasir-seg/split_polyps.py --dataset clinicdb."
    )


def _resolve_drive_root(root: Optional[Path]) -> Path:
    candidates: Iterable[Path]
    if root:
        candidates = (root,)
    else:
        candidates = (
            Path("datasets/DRIVE/DRIVE"),
            Path("datasets/drive/DRIVE"),
        )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "training" / "images").exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate DRIVE dataset root. Expected directories like datasets/DRIVE/DRIVE/training."
    )


def _resolve_promise12_root(root: Optional[Path]) -> Path:
    candidates: Iterable[Path]
    if root:
        candidates = (root,)
    else:
        candidates = (
            Path("datasets/PROMISE12"),
            Path("datasets/promise12"),
        )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        train_dir = candidate / "trainning"
        if train_dir.exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate PROMISE12 dataset root. Expected directories like datasets/PROMISE12/trainning."
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


def _load_dsb_case_mask(case_dir: Path, size: Tuple[int, int]) -> Image.Image:
    mask_dir = case_dir / "masks"
    mask = Image.new("L", size, 0)
    for mask_path in mask_dir.iterdir():
        if mask_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_img = Image.open(mask_path).convert("L").resize(size, Image.NEAREST)
        mask_arr = (np.array(mask_img) > 0).astype(np.uint8)
        mask_base = np.array(mask)
        mask = Image.fromarray(np.maximum(mask_base, mask_arr * 255).astype(np.uint8))
    return mask


def _ensure_dsb_augmented_split(
    train_root: Path, split: str, selected_cases: List[Path]
) -> List[Path]:
    aug_root = train_root / "augmented" / split
    aug_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(_AUGMENT_SEED)
    aug_cases: List[Path] = []
    for case_dir in selected_cases:
        dest_dir = aug_root / case_dir.name
        image_dest = dest_dir / "images"
        mask_dest = dest_dir / "masks"
        if not (image_dest / f"{case_dir.name}.png").exists():
            image_dest.mkdir(parents=True, exist_ok=True)
            mask_dest.mkdir(parents=True, exist_ok=True)
            image_files = sorted((case_dir / "images").glob("*"))
            if not image_files:
                continue
            image = Image.open(image_files[0]).convert("RGB")
            mask = _load_dsb_case_mask(case_dir, image.size)
            case_rng = random.Random(rng.randint(0, 2**32 - 1))
            image_aug, mask_aug = _augment_image_and_mask(image, mask, case_rng)
            image_aug.save(image_dest / f"{case_dir.name}.png")
            mask_aug.save(mask_dest / "mask.png")
        aug_cases.append(dest_dir)
    return aug_cases


def _rasterize_monuseg_mask(xml_path: Path, size: Tuple[int, int]) -> Image.Image:
    xml_root = ET.parse(xml_path).getroot()
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for region in xml_root.findall(".//Region"):
        vertices = region.find("Vertices")
        if vertices is None:
            continue
        points = [
            (float(vertex.attrib["X"]), float(vertex.attrib["Y"]))
            for vertex in vertices
            if "X" in vertex.attrib and "Y" in vertex.attrib
        ]
        if len(points) >= 3:
            draw.polygon(points, outline=1, fill=1)
    return mask


def _ensure_monuseg_augmented_split(root: Path, split: str) -> Tuple[Path, Path]:
    base_dir = root / "splits" / split
    images_dir = base_dir / "images"
    ann_dir = base_dir / "annotations"
    if not images_dir.exists() or not ann_dir.exists():
        raise FileNotFoundError(f"MoNuSeg split '{split}' not found at {base_dir}")
    aug_dir = root / "splits_aug" / split
    aug_img_dir = aug_dir / "images"
    aug_ann_dir = aug_dir / "annotations"
    aug_img_dir.mkdir(parents=True, exist_ok=True)
    aug_ann_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(_AUGMENT_SEED)
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = image_path.stem
        dest_img = aug_img_dir / f"{stem}.png"
        dest_mask = aug_ann_dir / f"{stem}.png"
        if dest_img.exists() and dest_mask.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        mask_path = ann_dir / f"{stem}.xml"
        if not mask_path.exists():
            continue
        mask = _rasterize_monuseg_mask(mask_path, image.size)
        case_rng = random.Random(rng.randint(0, 2**32 - 1))
        image_aug, mask_aug = _augment_image_and_mask(image, mask, case_rng)
        image_aug.save(dest_img)
        mask_aug.save(dest_mask)
    return aug_img_dir, aug_ann_dir


def _ensure_nuinsseg_augmented_split(root: Path, split: str) -> Tuple[Path, Path]:
    base_dir = root / "splits" / split
    images_dir = base_dir / "images"
    ann_dir = base_dir / "annotations"
    _ensure_exists(images_dir, f"NuInsSeg split images directory for {split}")
    _ensure_exists(ann_dir, f"NuInsSeg split annotations directory for {split}")
    aug_dir = root / "splits_aug" / split
    aug_img_dir = aug_dir / "images"
    aug_ann_dir = aug_dir / "annotations"
    aug_img_dir.mkdir(parents=True, exist_ok=True)
    aug_ann_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(_AUGMENT_SEED)
    pairs = _match_image_label_paths(images_dir, ann_dir)
    for image_path, mask_path in pairs:
        stem = image_path.stem
        dest_img = aug_img_dir / f"{stem}.png"
        dest_mask = aug_ann_dir / f"{stem}.png"
        if dest_img.exists() and dest_mask.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        bin_mask = (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)
        mask_img = Image.fromarray(bin_mask * 255, mode="L")
        case_rng = random.Random(rng.randint(0, 2**32 - 1))
        image_aug, mask_aug = _augment_image_and_mask(image, mask_img, case_rng)
        image_aug.save(dest_img)
        mask_aug.save(dest_mask)
    return aug_img_dir, aug_ann_dir


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
        case_dirs = sorted([p for p in train_root.iterdir() if p.is_dir() and p.name != "augmented"])
        if not case_dirs:
            raise RuntimeError(f"No case folders found under {train_root}")
        splits = _split_cases(case_dirs, val_ratio=val_ratio, test_ratio=test_ratio)
        split_key = split.lower()
        augmented = False
        if split_key.endswith("_aug"):
            augmented = True
            split_key = split_key[:-4]
        if split_key not in splits:
            raise ValueError("split must be 'train', 'val', 'test', or *_aug")
        selected = splits[split_key]
        if not selected:
            raise RuntimeError(f"No samples for DSB2018 {split_key} split. Adjust ratios.")
        if augmented:
            self.cases = _ensure_dsb_augmented_split(train_root, split_key, selected)
        else:
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
    ) -> None:
        monu_root = _resolve_monuseg_root(root)
        split_key = split.lower()
        augmented = False
        if split_key.endswith("_aug"):
            augmented = True
            split_key = split_key[:-4]
        if augmented:
            images_dir, ann_dir = _ensure_monuseg_augmented_split(monu_root, split_key)
        else:
            split_dir = monu_root / "splits" / split_key
            images_dir = split_dir / "images"
            ann_dir = split_dir / "annotations"
            _ensure_exists(images_dir, f"MoNuSeg split images directory for {split_key}")
            _ensure_exists(ann_dir, f"MoNuSeg split annotations directory for {split_key}")
        samples = sorted(
            [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        )
        if not samples:
            raise RuntimeError(
                f"No MoNuSeg samples found in {images_dir}. "
                "Run split_monuseg.py to generate splits."
            )
        self.samples = samples
        self.ann_dir = ann_dir
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def _load_mask(self, image_path: Path, size: Tuple[int, int]) -> Image.Image:
        base_name = image_path.stem
        ann_xml = self.ann_dir / f"{base_name}.xml"
        if ann_xml.exists():
            return _rasterize_monuseg_mask(ann_xml, size)
        ann_png = self.ann_dir / f"{base_name}.png"
        if ann_png.exists():
            return Image.open(ann_png).convert("L").resize(size, Image.NEAREST)
        raise FileNotFoundError(f"Annotation for {image_path.name} not found in {self.ann_dir}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        mask = self._load_mask(image_path, image.size)
        return _apply_shared_transforms(image, mask, self.image_size, self.augment)


class RUSDataset(Dataset):
    class_names = ["background", "organ"]
    num_classes = 2

    def __init__(
        self,
        root: Optional[Path],
        split: str,
        image_size: Optional[Tuple[int, int]] = None,
        augment: bool = False,
    ) -> None:
        rus_root = _resolve_rus_root(root)
        split_key = split.lower()
        if split_key not in {"train", "val", "test"}:
            raise ValueError("RUS supports 'train', 'val', or 'test' splits.")
        images_dir = rus_root / "images" / split_key
        ann_dir = rus_root / "annotations" / split_key
        _ensure_exists(images_dir, f"RUS images directory for split '{split_key}'")
        _ensure_exists(ann_dir, f"RUS annotations directory for split '{split_key}'")
        samples = _match_image_label_paths(images_dir, ann_dir)
        if not samples:
            raise RuntimeError(
                f"No overlapping image/mask files found for RUS split '{split_key}'."
            )
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask_arr = (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)
        mask_img = Image.fromarray(mask_arr, mode="L")
        return _apply_shared_transforms(image, mask_img, self.image_size, self.augment)


class NuInsSegDataset(Dataset):
    class_names = ["background", "nucleus"]
    num_classes = 2

    def __init__(
        self,
        root: Optional[Path],
        split: str,
        image_size: Optional[Tuple[int, int]] = None,
        augment: bool = False,
    ) -> None:
        nuinsseg_root = _resolve_nuinsseg_root(root)
        split_key = split.lower()
        augmented = False
        if split_key.endswith("_aug"):
            augmented = True
            split_key = split_key[:-4]
        if split_key not in {"train", "val", "test"}:
            raise ValueError("NuInsSeg supports 'train', 'val', 'test', or *_aug splits.")
        if augmented:
            images_dir, ann_dir = _ensure_nuinsseg_augmented_split(nuinsseg_root, split_key)
        else:
            split_dir = nuinsseg_root / "splits" / split_key
            images_dir = split_dir / "images"
            ann_dir = split_dir / "annotations"
            _ensure_exists(images_dir, f"NuInsSeg images directory for split '{split_key}'")
            _ensure_exists(ann_dir, f"NuInsSeg annotations directory for split '{split_key}'")
        samples = _match_image_label_paths(images_dir, ann_dir)
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask_arr = (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)
        mask_img = Image.fromarray(mask_arr, mode="L")
        return _apply_shared_transforms(image, mask_img, self.image_size, self.augment)


class ISIC2017Dataset(Dataset):
    class_names = ["background", "lesion"]
    num_classes = 2
    _SPLIT_MAP = {"train": "Training", "val": "Validation", "test": "Test_v2"}

    def __init__(
        self,
        root: Optional[Path],
        split: str,
        image_size: Optional[Tuple[int, int]] = None,
        augment: bool = False,
    ) -> None:
        isic_root = _resolve_isic_root(root)
        split_key = split.lower()
        if split_key not in self._SPLIT_MAP:
            raise ValueError("ISIC2017 supports 'train', 'val', or 'test' splits.")
        split_name = self._SPLIT_MAP[split_key]
        images_dir = isic_root / f"ISIC-2017_{split_name}_Data"
        masks_dir = isic_root / f"ISIC-2017_{split_name}_Part1_GroundTruth"
        _ensure_exists(images_dir, f"ISIC2017 images directory for {split_name}")
        _ensure_exists(masks_dir, f"ISIC2017 masks directory for {split_name}")
        multi_dirs = [
            d
            for d in sorted(isic_root.glob(f"ISIC-2017_{split_name}_Part1_GroundTruth*"))
            if d.is_dir() and d != masks_dir and "Part2" not in d.name
        ]
        samples: List[Tuple[Path, Path]] = []
        extra_masks: List[List[Path]] = []
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if image_path.stem.endswith("_superpixels"):
                continue
            mask_path = masks_dir / f"{image_path.stem}_segmentation.png"
            if not mask_path.exists():
                continue
            annot_masks: List[Path] = []
            for multi_dir in multi_dirs:
                candidate = multi_dir / f"{image_path.stem}_segmentation.png"
                if candidate.exists():
                    annot_masks.append(candidate)
            samples.append((image_path, mask_path))
            extra_masks.append(annot_masks)
        if not samples:
            raise RuntimeError(f"No ISIC2017 samples found in {images_dir}")
        self.samples = samples
        self.multi_mask_paths = extra_masks
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask_arr = (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)
        mask_img = Image.fromarray(mask_arr, mode="L")
        return _apply_shared_transforms(image, mask_img, self.image_size, self.augment)


class DriveDataset(Dataset):
    class_names = ["background", "vessel"]
    num_classes = 2

    def __init__(
        self,
        root: Optional[Path],
        split: str,
        image_size: Optional[Tuple[int, int]] = None,
        augment: bool = False,
    ) -> None:
        drive_root = _resolve_drive_root(root)
        split_key = split.lower()
        if split_key not in {"train", "val", "test"}:
            raise ValueError("DRIVE supports 'train', 'val', or 'test' splits.")
        if split_key == "val":
            split_dir = drive_root / "training"
            self.samples = self._build_samples(split_dir, offset=10, limit=10)
        else:
            if split_key == "train":
                split_dir = drive_root / "training"
                self.samples = self._build_samples(split_dir, offset=0, limit=10)
            else:
                split_dir = drive_root / "test"
                self.samples = self._build_samples(split_dir)
        self.image_size = image_size
        self.augment = augment

    def _build_samples(
        self, split_dir: Path, offset: int = 0, limit: Optional[int] = None
    ) -> List[Tuple[Path, Path]]:
        images_dir = split_dir / "images"
        manual_dir = None
        mask_style = False
        if images_dir.exists():
            manual_dir = split_dir / "1st_manual"
        else:
            alt_images = split_dir / "Images"
            alt_manual = split_dir / "manual"
            if alt_images.exists():
                images_dir = alt_images
                manual_dir = alt_manual
        if manual_dir is None or not manual_dir.exists():
            mask_dir = split_dir / "mask"
            if mask_dir.exists():
                manual_dir = mask_dir
                mask_style = True
        _ensure_exists(images_dir, f"DRIVE images directory at {split_dir}")
        _ensure_exists(manual_dir, f"DRIVE manual directory at {split_dir}")
        image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
        if limit is not None:
            image_files = image_files[offset : offset + limit]
        samples: List[Tuple[Path, Path]] = []
        for image_path in image_files:
            if mask_style:
                mask_path = manual_dir / f"{image_path.stem}_mask.gif"
                if not mask_path.exists():
                    mask_path = manual_dir / f"{image_path.stem}_mask.png"
                stem = image_path.stem
            else:
                stem = image_path.stem.split("_")[0]
                mask_path = manual_dir / f"{stem}_manual1.gif"
                if not mask_path.exists():
                    mask_path = manual_dir / f"{stem}_manual1.png"
            if not mask_path.exists():
                continue
            samples.append((image_path, mask_path))
        if not samples:
            raise RuntimeError(f"No DRIVE samples found under {split_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask_arr = (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)
        mask_img = Image.fromarray(mask_arr, mode="L")
        return _apply_shared_transforms(image, mask_img, self.image_size, self.augment)


class Promise12Dataset(Dataset):
    class_names = ["background", "prostate"]
    num_classes = 2

    def __init__(
        self,
        root: Optional[Path],
        split: str,
        image_size: Optional[Tuple[int, int]] = None,
        augment: bool = False,
    ) -> None:
        try:
            import SimpleITK as sitk  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Please install SimpleITK to use the PROMISE12 dataset.") from exc
        promise_root = _resolve_promise12_root(root)
        split_key = split.lower()
        if split_key not in {"train", "val", "test"}:
            raise ValueError("PROMISE12 supports 'train', 'val', or 'test' splits.")
        if split_key == "test":
            case_dir = promise_root / "test"
            case_paths = self._list_cases(case_dir)
        else:
            case_dir = promise_root / "trainning"
            all_cases = self._list_cases(case_dir)
            if split_key == "train":
                case_paths = all_cases[:40]
            else:
                case_paths = all_cases[40:]
        self.slices: List[Tuple[np.ndarray, np.ndarray]] = []
        for case_path in case_paths:
            seg_path = case_path.with_name(f"{case_path.stem}_segmentation.mhd")
            if not seg_path.exists():
                continue
            image = sitk.ReadImage(str(case_path))
            mask = sitk.ReadImage(str(seg_path))
            img_arr = sitk.GetArrayFromImage(image).astype(np.float32)
            mask_arr = sitk.GetArrayFromImage(mask).astype(np.uint8)
            for z in range(img_arr.shape[0]):
                img_slice = img_arr[z]
                slice_min = float(img_slice.min())
                slice_max = float(img_slice.max())
                if slice_max > slice_min:
                    norm = (img_slice - slice_min) / (slice_max - slice_min)
                else:
                    norm = np.zeros_like(img_slice, dtype=np.float32)
                img_uint8 = (norm * 255.0).clip(0, 255).astype(np.uint8)
                mask_slice = (mask_arr[z] > 0).astype(np.uint8)
                self.slices.append((img_uint8, mask_slice))
        if not self.slices:
            raise RuntimeError(f"No PROMISE12 slices were extracted from {case_dir}")
        self.image_size = image_size
        self.augment = augment

    def _list_cases(self, directory: Path) -> List[Path]:
        cases = sorted(directory.glob("Case??.mhd"))
        if not cases:
            raise RuntimeError(f"No PROMISE12 cases found in {directory}")
        return cases

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_slice, mask_slice = self.slices[idx]
        image = Image.fromarray(image_slice, mode="L").convert("RGB")
        mask = Image.fromarray(mask_slice * 255, mode="L")
        return _apply_shared_transforms(image, mask, self.image_size, self.augment)


class _PolypDataset(Dataset):
    class_names = ["background", "polyp"]
    num_classes = 2

    def __init__(self, samples: List[Tuple[Path, Path]], image_size: Optional[Tuple[int, int]], augment: bool):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask_arr = (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)
        mask_img = Image.fromarray(mask_arr, mode="L")
        return _apply_shared_transforms(image, mask_img, self.image_size, self.augment)


class KvasirSegDataset(_PolypDataset):
    def __init__(
        self,
        root: Optional[Path],
        split: str,
        image_size: Optional[Tuple[int, int]] = None,
        augment: bool = False,
    ) -> None:
        dataset_root = _resolve_kvasirseg_root(root)
        split_key = split.lower()
        if split_key not in {"train", "val", "test"}:
            raise ValueError("Kvasir-SEG supports 'train', 'val', or 'test' splits.")
        split_dir = dataset_root / "splits" / split_key
        images_dir = split_dir / "images"
        ann_dir = split_dir / "annotations"
        _ensure_exists(images_dir, f"Kvasir-SEG images for split '{split_key}'")
        _ensure_exists(ann_dir, f"Kvasir-SEG annotations for split '{split_key}'")
        samples = _match_image_label_paths(images_dir, ann_dir)
        if not samples:
            raise RuntimeError(f"No samples found in {images_dir}")
        super().__init__(samples, image_size, augment)


class ClinicDBDataset(_PolypDataset):
    def __init__(
        self,
        root: Optional[Path],
        split: str,
        image_size: Optional[Tuple[int, int]] = None,
        augment: bool = False,
    ) -> None:
        dataset_root = _resolve_clinicdb_root(root)
        split_key = split.lower()
        if split_key not in {"train", "val", "test"}:
            raise ValueError("ClinicDB supports 'train', 'val', or 'test' splits.")
        split_dir = dataset_root / "splits" / split_key
        images_dir = split_dir / "images"
        ann_dir = split_dir / "annotations"
        _ensure_exists(images_dir, f"CVC-ClinicDB images for split '{split_key}'")
        _ensure_exists(ann_dir, f"CVC-ClinicDB annotations for split '{split_key}'")
        samples = _match_image_label_paths(images_dir, ann_dir)
        if not samples:
            raise RuntimeError(f"No samples found in {images_dir}")
        super().__init__(samples, image_size, augment)


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
        base_split = split[:-4] if split.endswith("_aug") else split
        if base_split not in {"train", "val", "test"}:
            raise ValueError("DSB2018 supports 'train', 'val', 'test', or *_aug splits.")
        dataset = DSB2018Dataset(
            root=root_path,
            split=split,
            image_size=image_size,
            augment=augment,
        )
    elif dataset_name == "monuseg":
        base_split = split[:-4] if split.endswith("_aug") else split
        if base_split not in {"train", "val", "test"}:
            raise ValueError("MoNuSeg supports 'train', 'val', 'test', or *_aug splits.")
        dataset = MoNuSegDataset(
            root=root_path,
            split=split,
            image_size=image_size,
            augment=augment,
        )
    elif dataset_name == "rus":
        if split not in {"train", "val", "test"}:
            raise ValueError("RUS supports 'train', 'val', or 'test' splits.")
        dataset = RUSDataset(
            root=root_path,
            split=split,
            image_size=image_size,
            augment=augment,
        )
    elif dataset_name == "nuinsseg":
        base_split = split[:-4] if split.endswith("_aug") else split
        if base_split not in {"train", "val", "test"}:
            raise ValueError("NuInsSeg supports 'train', 'val', 'test', or *_aug splits.")
        dataset = NuInsSegDataset(
            root=root_path,
            split=split,
            image_size=image_size,
            augment=augment,
        )
    elif dataset_name == "isic2017":
        if split not in {"train", "val", "test"}:
            raise ValueError("ISIC2017 supports 'train', 'val', or 'test' splits.")
        dataset = ISIC2017Dataset(
            root=root_path,
            split=split,
            image_size=image_size,
            augment=augment,
        )
    elif dataset_name == "kvasirseg":
        if split not in {"train", "val", "test"}:
            raise ValueError("Kvasir-SEG supports 'train', 'val', or 'test' splits.")
        dataset = KvasirSegDataset(
            root=root_path,
            split=split,
            image_size=image_size,
            augment=augment,
        )
    elif dataset_name == "clinicdb":
        if split not in {"train", "val", "test"}:
            raise ValueError("ClinicDB supports 'train', 'val', or 'test' splits.")
        dataset = ClinicDBDataset(
            root=root_path,
            split=split,
            image_size=image_size,
            augment=augment,
        )
    elif dataset_name == "drive":
        if split not in {"train", "val", "test"}:
            raise ValueError("DRIVE supports 'train', 'val', or 'test' splits.")
        dataset = DriveDataset(
            root=root_path,
            split=split,
            image_size=image_size,
            augment=augment,
        )
    elif dataset_name == "promise12":
        if split not in {"train", "val", "test"}:
            raise ValueError("PROMISE12 supports 'train', 'val', or 'test' splits.")
        dataset = Promise12Dataset(
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
