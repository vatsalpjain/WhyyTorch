"""Generate augmented copies of labeled images.

Saves new PNGs into data/images/ as sequential image_N.png files
and appends entries to data/labels.json.

Naming:
    image_7.png  ->  image_61.png, image_62.png, ...

Usage:
    python 05-football-cnn/data/augment_images.py --source image_7.png
    python 05-football-cnn/data/augment_images.py --all
    python 05-football-cnn/data/augment_images.py --all-ball
    python 05-football-cnn/data/augment_images.py --rename-existing
    python 05-football-cnn/data/augment_images.py --source image_7.png --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "images"
LABELS_PATH = Path(__file__).resolve().parent / "labels.json"

IMAGE_NAME_RE = re.compile(r"^image_(\d+)\.png$")
AUG_NAME_RE = re.compile(r"^(?P<base>image_\d+)_aug_(?P<aug>.+)\.png$")


@dataclass
class AugmentResult:
    suffix: str
    image_bgr: np.ndarray
    update_label: Callable[[dict], dict] | None = None


def load_labels() -> list[dict]:
    if not LABELS_PATH.exists():
        return []
    return json.loads(LABELS_PATH.read_text())


def save_labels(labels: list[dict]) -> None:
    LABELS_PATH.write_text(json.dumps(labels, indent=2))


def labels_by_image(labels: list[dict]) -> dict[str, dict]:
    return {item["image"]: item for item in labels}


def max_image_number(labels: list[dict]) -> int:
    highest = 0
    for item in labels:
        match = IMAGE_NAME_RE.match(item["image"])
        if match:
            highest = max(highest, int(match.group(1)))
    for path in DATA_DIR.glob("image_*.png"):
        match = IMAGE_NAME_RE.match(path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest


def clip_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def flip_label_h(label: dict) -> dict:
    out = deepcopy(label)
    if label.get("has_ball") and label.get("cx") is not None:
        out["cx"] = round(1.0 - float(label["cx"]), 6)
    return out


def aug_hflip(image: np.ndarray, label: dict) -> AugmentResult:
    return AugmentResult("hflip", cv2.flip(image, 1), flip_label_h)


def aug_bright(image: np.ndarray, label: dict) -> AugmentResult:
    out = clip_uint8(image.astype(np.float32) * 1.20)
    return AugmentResult("bright", out)


def aug_dark(image: np.ndarray, label: dict) -> AugmentResult:
    out = clip_uint8(image.astype(np.float32) * 0.80)
    return AugmentResult("dark", out)


def aug_contrast_up(image: np.ndarray, label: dict) -> AugmentResult:
    out = clip_uint8(image.astype(np.float32) * 1.15 + 10)
    return AugmentResult("contrast_up", out)


def aug_contrast_down(image: np.ndarray, label: dict) -> AugmentResult:
    out = clip_uint8(image.astype(np.float32) * 0.90 + 5)
    return AugmentResult("contrast_down", out)


def aug_noise(image: np.ndarray, label: dict) -> AugmentResult:
    noise = np.random.normal(0, 8, image.shape).astype(np.float32)
    out = clip_uint8(image.astype(np.float32) + noise)
    return AugmentResult("noise", out)


def aug_gamma_light(image: np.ndarray, label: dict) -> AugmentResult:
    table = ((np.arange(256) / 255.0) ** 0.85 * 255).astype(np.uint8)
    return AugmentResult("gamma_light", cv2.LUT(image, table))


def aug_hflip_bright(image: np.ndarray, label: dict) -> AugmentResult:
    flipped = cv2.flip(image, 1)
    out = clip_uint8(flipped.astype(np.float32) * 1.15)
    return AugmentResult("hflip_bright", out, flip_label_h)


DEFAULT_AUGMENTATIONS: list[Callable[[np.ndarray, dict], AugmentResult]] = [
    aug_hflip,
    aug_bright,
    aug_dark,
    aug_contrast_up,
    aug_contrast_down,
    aug_noise,
    aug_gamma_light,
    aug_hflip_bright,
]


def build_label(
    source_label: dict,
    output_name: str,
    updated: dict | None = None,
    aug_type: str | None = None,
) -> dict:
    base = updated if updated is not None else deepcopy(source_label)
    base["image"] = output_name
    base["source"] = source_label["image"]
    if aug_type is not None:
        base["aug"] = aug_type
    return base


def already_augmented_sources(labels: list[dict]) -> set[str]:
    sources: set[str] = set()
    for item in labels:
        source = item.get("source")
        if source:
            sources.add(source)
            continue
        match = AUG_NAME_RE.match(item["image"])
        if match:
            sources.add(f"{match.group('base')}.png")
    return sources


def rename_existing_aug_files(labels: list[dict], dry_run: bool = False) -> int:
    """Rename image_*_aug_*.png files to sequential image_N.png and update labels."""
    next_num = max_image_number(labels) + 1
    # Prefer disk files so we don't miss unlabeled augs.
    aug_paths = sorted(
        DATA_DIR.glob("image_*_aug_*.png"),
        key=lambda p: (
            int(AUG_NAME_RE.match(p.name).group("base").split("_")[1]),
            AUG_NAME_RE.match(p.name).group("aug"),
        )
        if AUG_NAME_RE.match(p.name)
        else (0, p.name),
    )
    label_map = labels_by_image(labels)
    renamed = 0

    for path in aug_paths:
        match = AUG_NAME_RE.match(path.name)
        if not match:
            continue
        old_name = path.name
        new_name = f"image_{next_num}.png"
        new_path = DATA_DIR / new_name
        source_name = f"{match.group('base')}.png"
        aug_type = match.group("aug")

        if dry_run:
            print(f"would rename: {old_name} -> {new_name}")
        else:
            if new_path.exists():
                raise FileExistsError(f"Target already exists: {new_path}")
            path.rename(new_path)
            if old_name in label_map:
                item = label_map[old_name]
                item["image"] = new_name
                item["source"] = source_name
                item["aug"] = aug_type
            else:
                # Recover label from source if possible.
                source_label = label_map.get(source_name)
                if source_label is None:
                    raise KeyError(f"No label for renamed file {old_name} or source {source_name}")
                updated = flip_label_h(source_label) if "hflip" in aug_type else None
                labels.append(build_label(source_label, new_name, updated, aug_type))
            print(f"renamed: {old_name} -> {new_name}")
        next_num += 1
        renamed += 1

    return renamed


def augment_one(
    source_name: str,
    labels: list[dict],
    augmentations: list[Callable[[np.ndarray, dict], AugmentResult]],
    next_num: int,
    dry_run: bool = False,
    force: bool = False,
) -> tuple[int, int]:
    source_path = DATA_DIR / source_name
    if not source_path.exists():
        raise FileNotFoundError(f"Missing image: {source_path}")

    label_map = labels_by_image(labels)
    if source_name not in label_map:
        raise KeyError(f"No label found for {source_name}")

    done = already_augmented_sources(labels)
    if source_name in done and not force:
        print(f"skip (already augmented): {source_name}")
        return 0, next_num

    source_label = label_map[source_name]
    image = cv2.imread(str(source_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {source_path}")

    created = 0
    for aug_fn in augmentations:
        result = aug_fn(image, source_label)
        out_name = f"image_{next_num}.png"
        out_path = DATA_DIR / out_name

        if out_path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")

        new_label = build_label(
            source_label,
            out_name,
            result.update_label(source_label) if result.update_label else None,
            result.suffix,
        )

        if dry_run:
            print(f"would create: {out_name}  <- {source_name}  [{result.suffix}]")
        else:
            cv2.imwrite(str(out_path), result.image_bgr)
            labels.append(new_label)
            print(f"created: {out_name}  <- {source_name}  [{result.suffix}]")
        next_num += 1
        created += 1

    return created, next_num


def pick_sources(args: argparse.Namespace, labels: list[dict]) -> list[str]:
    originals = [
        item["image"]
        for item in labels
        if IMAGE_NAME_RE.match(item["image"]) and not item.get("source") and "_aug_" not in item["image"]
    ]
    # Keep original numeric order.
    originals = sorted(originals, key=lambda name: int(IMAGE_NAME_RE.match(name).group(1)))

    if args.all:
        return originals
    if args.all_ball:
        label_map = labels_by_image(labels)
        return [name for name in originals if label_map[name].get("has_ball")]
    if not args.source:
        raise SystemExit("Pass --source image_7.png, or use --all / --all-ball")
    return args.source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment labeled football images.")
    parser.add_argument(
        "--source",
        nargs="+",
        help="Original image file names, e.g. image_7.png",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Augment every original labeled image (ball and no-ball)",
    )
    parser.add_argument(
        "--all-ball",
        action="store_true",
        help="Augment every image labeled with has_ball=true",
    )
    parser.add_argument(
        "--rename-existing",
        action="store_true",
        help="Rename existing *_aug_*.png files to sequential image_N.png",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be created without writing files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create augs even if this source was already augmented",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noise augmentation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    labels = load_labels()
    total = 0

    if args.rename_existing:
        total += rename_existing_aug_files(labels, dry_run=args.dry_run)
        if not args.dry_run:
            save_labels(labels)
            print(f"Done renaming. Renamed {total} files.")
            print(f"Updated labels: {LABELS_PATH}")
        else:
            print(f"Dry run complete. Would rename {total} files.")
        if not (args.all or args.all_ball or args.source):
            return

    sources = pick_sources(args, labels)
    next_num = max_image_number(labels) + 1

    for source_name in sources:
        created, next_num = augment_one(
            source_name,
            labels,
            DEFAULT_AUGMENTATIONS,
            next_num,
            dry_run=args.dry_run,
            force=args.force,
        )
        total += created

    if not args.dry_run:
        save_labels(labels)
        print(f"Done. Created/renamed {total} images.")
        print(f"Updated labels: {LABELS_PATH}")
    else:
        print(f"Dry run complete. Would create/rename up to {total} images.")


if __name__ == "__main__":
    main()
