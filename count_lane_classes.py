#!/usr/bin/env python3

from pathlib import Path
from collections import Counter

LABEL_ROOT = Path(
    "/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/datasets/labels_corrected"
)

CLASSES = [0, 1, 2, 3]


def get_classes(label_path):
    """读取一个标签文件中实际出现的 class。"""
    classes = set()

    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"[WARN] 无法读取: {label_path} | {e}")
        return classes

    if not text:
        return classes

    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        try:
            cls_id = int(float(parts[0]))
        except (ValueError, IndexError):
            print(
                f"[WARN] 无法解析: {label_path} "
                f"第 {line_no} 行: {line}"
            )
            continue

        classes.add(cls_id)

    return classes


def count_split(split):
    folder = LABEL_ROOT / split
    counter = Counter()

    txt_files = sorted(folder.glob("*.txt"))

    empty_count = 0

    for txt in txt_files:
        classes = get_classes(txt)

        if not classes:
            empty_count += 1
            continue

        # 一个文件中同一个 class 即使意外出现多行，
        # 这里也只算 1 条
        for cls_id in classes:
            counter[cls_id] += 1

    return counter, len(txt_files), empty_count


def main():
    print("=" * 65)
    print("Lane Robot 标签类别统计")
    print(f"目录: {LABEL_ROOT}")
    print("=" * 65)

    total = Counter()
    total_files = 0
    total_empty = 0

    for split in ["train", "valid"]:
        counter, file_count, empty_count = count_split(split)

        total.update(counter)
        total_files += file_count
        total_empty += empty_count

        print(f"\n[{split}]")
        print(f"标签文件总数 : {file_count}")
        print(f"空标签文件   : {empty_count}")

        for cls_id in CLASSES:
            print(f"class {cls_id}: {counter[cls_id]} 条")

    print("\n" + "=" * 65)
    print("[TOTAL]")
    print(f"标签文件总数 : {total_files}")
    print(f"空标签文件   : {total_empty}")

    for cls_id in CLASSES:
        print(f"class {cls_id}: {total[cls_id]} 条")

    print("=" * 65)


if __name__ == "__main__":
    main()
