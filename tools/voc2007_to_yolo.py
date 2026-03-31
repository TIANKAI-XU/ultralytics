from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


VOC_CLASSES = [
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
CLASS_TO_ID = {name: idx for idx, name in enumerate(VOC_CLASSES)}
# Map each split to the official VOC2007 extracted package directory.
# 将每个数据划分映射到官方 VOC2007 解压后的目录。
SPLIT_PACKAGES = {
    "train": "VOCtrainval_06-Nov-2007",
    "val": "VOCtrainval_06-Nov-2007",
    "trainval": "VOCtrainval_06-Nov-2007",
    "test": "VOCtest_06-Nov-2007",
}


def parse_args() -> argparse.Namespace:
    # Keep the CLI minimal: point to the extracted VOC2007 root and choose output/splits.
    # 命令行尽量保持简单：指定 VOC2007 解压根目录，以及输出目录和需要转换的 split。
    parser = argparse.ArgumentParser(description="Convert Pascal VOC2007 to YOLO detection format.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("VOC2007"),
        help="VOC2007 extracted root directory containing VOCtrainval_06-Nov-2007 and VOCtest_06-Nov-2007.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets") / "VOC2007_YOLO",
        help="Output YOLO dataset directory.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "trainval", "test"],
        help="VOC splits to convert.",
    )
    parser.add_argument(
        "--include-difficult",
        action="store_true",
        help="Include VOC objects with difficult=1.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only convert the first N images of each split.",
    )
    return parser.parse_args()


def get_voc_root(source: Path, split: str) -> Path:
    # train/val come from the trainval package, while test comes from the test package.
    # train/val 来自 trainval 压缩包，test 来自 test 压缩包。
    package = SPLIT_PACKAGES[split]
    voc_root = source / package / "VOCdevkit" / "VOC2007"
    if not voc_root.exists():
        raise FileNotFoundError(f"VOC root not found for split '{split}': {voc_root}")
    return voc_root


def get_split_file(source: Path, split: str) -> Path:
    voc_root = get_voc_root(source, split)
    split_file = voc_root / "ImageSets" / "Main" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    return split_file


def read_split_ids(source: Path, split: str, limit: int | None) -> list[str]:
    image_ids = [x.strip() for x in get_split_file(source, split).read_text(encoding="utf-8").splitlines() if x.strip()]
    return image_ids[:limit] if limit is not None else image_ids


def yolo_box(width: int, height: int, xmin: float, xmax: float, ymin: float, ymax: float) -> tuple[float, float, float, float]:
    # VOC boxes are absolute corner coordinates; YOLO expects normalized center-width-height.
    # VOC 标注框是绝对坐标的左上/右下点，YOLO 需要归一化后的中心点和宽高。
    x_center = ((xmin + xmax) / 2.0 - 1.0) / width
    y_center = ((ymin + ymax) / 2.0 - 1.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def convert_annotation(xml_path: Path, label_path: Path, include_difficult: bool) -> None:
    # Convert one VOC XML file into one YOLO txt label file.
    # 将单个 VOC XML 标注文件转换为对应的 YOLO txt 标签文件。
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")

    width = int(size.findtext("width", default="0"))
    height = int(size.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}: {width}x{height}")

    rows: list[str] = []
    for obj in root.findall("object"):
        cls_name = obj.findtext("name")
        if cls_name not in CLASS_TO_ID:
            continue

        difficult = int(obj.findtext("difficult", default="0"))
        if difficult and not include_difficult:
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.findtext("xmin", default="0"))
        xmax = float(bndbox.findtext("xmax", default="0"))
        ymin = float(bndbox.findtext("ymin", default="0"))
        ymax = float(bndbox.findtext("ymax", default="0"))
        if xmax <= xmin or ymax <= ymin:
            continue

        x_center, y_center, box_width, box_height = yolo_box(width, height, xmin, xmax, ymin, ymax)
        rows.append(
            f"{CLASS_TO_ID[cls_name]} "
            f"{x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )

    label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def find_image(images_dir: Path, image_id: str) -> Path:
    # Most VOC2007 images are .jpg, but a small fallback set of suffixes costs almost nothing.
    # VOC2007 图片通常是 .jpg，这里顺手兼容几个常见后缀，代价很低。
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        image_path = images_dir / f"{image_id}{suffix}"
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"Image not found for '{image_id}' under {images_dir}")


def convert_split(source: Path, output: Path, split: str, include_difficult: bool, limit: int | None) -> int:
    # Export one split into YOLO-style images/ and labels/ folders.
    # 将单个 split 导出为 YOLO 所需的 images/ 和 labels/ 目录结构。
    voc_root = get_voc_root(source, split)
    annotations_dir = voc_root / "Annotations"
    images_dir = voc_root / "JPEGImages"
    image_ids = read_split_ids(source, split, limit)

    out_images = output / "images" / split
    out_labels = output / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        xml_path = annotations_dir / f"{image_id}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"Annotation not found: {xml_path}")

        image_path = find_image(images_dir, image_id)
        label_path = out_labels / f"{image_id}.txt"
        convert_annotation(xml_path, label_path, include_difficult)
        shutil.copy2(image_path, out_images / image_path.name)

    return len(image_ids)


def write_yaml(output: Path, splits: list[str]) -> Path:
    # Generate the Ultralytics dataset yaml so the converted dataset can be used directly.
    # 生成 Ultralytics 数据集 yaml，方便转换后直接训练。
    yaml_path = output / "VOC2007.yaml"
    lines = [f"path: {output.resolve().as_posix()}"]
    for split in splits:
        lines.append(f"{split}: images/{split}")
    lines.append("names:")
    for idx, name in enumerate(VOC_CLASSES):
        lines.append(f"  {idx}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def main() -> None:
    # End-to-end conversion: resolve paths, convert each split, then write dataset yaml.
    # 整体流程：解析路径，逐个 split 转换，最后写出数据集 yaml。
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    counts: dict[str, int] = {}
    for split in args.splits:
        counts[split] = convert_split(source, output, split, args.include_difficult, args.limit)

    yaml_path = write_yaml(output, args.splits)

    print(f"Source: {source}")
    print(f"Output: {output}")
    for split, count in counts.items():
        print(f"{split}: converted {count} images")
    print(f"data yaml: {yaml_path}")


if __name__ == "__main__":
    main()
