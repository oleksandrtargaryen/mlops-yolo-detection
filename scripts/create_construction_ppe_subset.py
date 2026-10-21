import argparse
import random
import shutil
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_split(src_root: Path, dst_root: Path, split: str, count: int, seed: int):
    src_images = src_root / "images" / split
    src_labels = src_root / "labels" / split
    dst_images = dst_root / "images" / split
    dst_labels = dst_root / "labels" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMG_EXT)
    # беремо тільки ті картинки, для яких є розмітка
    images = [p for p in images if (src_labels / f"{p.stem}.txt").exists()]

    if count > len(images):
        print(f"{split}: requested {count}, but only {len(images)} available")
        count = len(images)

    random.Random(seed).shuffle(images)
    for img in images[:count]:
        shutil.copy2(img, dst_images / img.name)
        shutil.copy2(src_labels / f"{img.stem}.txt", dst_labels / f"{img.stem}.txt")

    print(f"{split}: {count} images")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="datasets/construction-ppe")
    parser.add_argument("--dst", default="datasets/construction-ppe-small")
    parser.add_argument("--train", type=int, default=100)
    parser.add_argument("--val", type=int, default=30)
    parser.add_argument("--test", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not (src / "images").exists():
        raise SystemExit(f"Dataset not found: {src}")

    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} already exists, use --overwrite")
        shutil.rmtree(dst)

    for split, count in [("train", args.train), ("val", args.val), ("test", args.test)]:
        copy_split(src, dst, split, count, args.seed)

    with open(src / "data.yaml", encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]

    data = {
        "path": str(dst),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    with open(dst / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"Done: {dst}")


if __name__ == "__main__":
    main()
