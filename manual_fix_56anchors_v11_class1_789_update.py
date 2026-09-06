#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


SCRIPT_VERSION = "2026-08-10-v11-base-class1-789"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASETS_ROOT = PROJECT_ROOT / "datasets"

NUM_Y_ANCHORS = 56
NUM_X_BINS = 640
CROP_RATIO = 2.0 / 3.0

IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

# OpenCV BGR: five visually distinct colors.
CLASS_COLORS = {
    0: (255, 255, 0),   # cyan
    1: (0, 255, 255),   # yellow
    2: (0, 255, 0),     # green
    3: (0, 0, 255),     # red
    4: (255, 0, 255),   # magenta
}


# ============================================================
# Files
# ============================================================

def resolve_project_path(path: Path) -> Path:
    """Resolve relative paths against the project root, not the shell cwd.

    Expected layout:
        project/
        ├── scripts/
        │   └── this_script.py
        └── datasets/
            ├── images/
            └── labels/
    """
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片: {path}")
    return image


def find_images(image_dir: Path) -> list[Path]:
    return sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def load_label_file(path: Path) -> dict[int, list[tuple[float, float]]]:
    """
    Strict parser.

    Every non-empty line is:
        class_id x1 y1 x2 y2 ...

    Only classes explicitly present in the file are returned. No class 0 is
    created implicitly. Duplicate lines of the same class are concatenated.
    """
    result: dict[int, list[tuple[float, float]]] = {}
    if not path.exists():
        return result

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            tokens = line.split()
            try:
                class_id = int(tokens[0])
            except (ValueError, IndexError):
                print(f"[警告] 类别解析失败: {path}:{line_no}: {line}")
                continue

            if class_id not in CLASS_COLORS:
                print(f"[警告] 不支持的类别 {class_id}: {path}:{line_no}")
                continue

            coords = tokens[1:]
            if len(coords) % 2 != 0:
                print(f"[警告] 奇数个坐标，忽略最后一个: {path}:{line_no}")
                coords = coords[:-1]

            points: list[tuple[float, float]] = []
            for i in range(0, len(coords), 2):
                try:
                    x = float(coords[i])
                    y = float(coords[i + 1])
                except ValueError:
                    print(f"[警告] 点解析失败: {path}:{line_no}, point={i // 2}")
                    continue

                if not math.isfinite(x) or not math.isfinite(y):
                    continue

                if not 0.0 <= y <= 1.0:
                    print(
                        f"[警告] y 坐标越界，点已跳过: {path}:{line_no}, "
                        f"point={i // 2}, x={x}, y={y}"
                    )
                    continue

                # Corrected labels always contain all 56 anchor positions.
                # x=-1 means this fixed y row has no valid line point. Keep
                # missing rows out of the in-memory annotation so all existing
                # drawing/editing logic continues to operate on valid points.
                if math.isclose(x, -1.0, rel_tol=0.0, abs_tol=1e-9):
                    continue

                if not 0.0 <= x <= 1.0:
                    print(
                        f"[警告] x 坐标越界，点已跳过: {path}:{line_no}, "
                        f"point={i // 2}, x={x}, y={y}"
                    )
                    continue

                points.append((x, y))

            result.setdefault(class_id, []).extend(points)

    return result


def save_label_file_atomic(
    path: Path,
    annotations: dict[int, list[tuple[float, float]]],
) -> None:
    """Atomically save every existing class as exactly 56 anchor positions.

    Each output line keeps the fixed bottom-to-top y-anchor order:
        class_id x0 y0 x1 y1 ... x55 y55

    A row intersected by the line stores its normalized x coordinate. A row
    without an intersection stores x=-1. Internal annotations remain sparse,
    so the editor's drawing and editing behavior is unchanged.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    y_anchors = make_y_anchor_norms()
    lines: list[str] = []

    for class_id in sorted(annotations):
        points = annotations[class_id]
        if not points:
            continue

        # Start with all 56 rows marked as no-lane, then place each valid point
        # into its nearest fixed y anchor. If several points map to one row, the
        # last point wins, matching the editor's replace-on-same-row behavior.
        row_x = np.full(NUM_Y_ANCHORS, -1.0, dtype=np.float64)
        for x, y in points:
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                continue
            yi = nearest_y_anchor(float(y), y_anchors)
            row_x[yi] = float(np.clip(x, 0.0, 1.0))

        values = [str(class_id)]
        for yi, y in enumerate(y_anchors):
            x = float(row_x[yi])
            x_text = "-1" if x < 0.0 else f"{x:.6f}"
            values.extend((x_text, f"{float(y):.6f}"))
        lines.append(" ".join(values))

    content = "\n".join(lines)
    if content:
        content += "\n"

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


# ============================================================
# Coordinate system
# ============================================================

def make_y_anchor_norms() -> np.ndarray:
    # 0 = bottom, 55 = top edge of lower 2/3.
    return np.linspace(
        1.0,
        1.0 - CROP_RATIO,
        NUM_Y_ANCHORS,
        dtype=np.float64,
    )


def normalized_to_pixel(
    x: float,
    y: float,
    width: int,
    height: int,
) -> tuple[int, int]:
    px = int(round(x * max(width - 1, 1)))
    py = int(round(y * max(height - 1, 1)))
    return (
        int(np.clip(px, 0, max(width - 1, 0))),
        int(np.clip(py, 0, max(height - 1, 0))),
    )


def pixel_x_to_bin(x_pixel: float, width: int) -> int:
    x_norm = x_pixel / max(width - 1, 1)
    x_bin = int(round(x_norm * NUM_X_BINS))
    return int(np.clip(x_bin, 0, NUM_X_BINS - 1))


def x_bin_to_norm(x_bin: float) -> float:
    return float(x_bin) / float(NUM_X_BINS)


def x_bin_to_pixel(x_bin: float, width: int) -> float:
    return x_bin_to_norm(x_bin) * max(width - 1, 1)


def nearest_y_anchor(y_norm: float, anchors: np.ndarray) -> int:
    return int(np.argmin(np.abs(anchors - y_norm)))


def fit_image_rect(
    image_width: int,
    image_height: int,
    canvas_width: int,
    canvas_height: int,
) -> tuple[int, int, int, int]:
    """Return an aspect-ratio-preserving image rectangle inside a canvas.

    The result is ``(left, top, width, height)``. Any unused canvas area is
    treated as non-interactive letterbox padding.
    """
    image_width = max(int(image_width), 1)
    image_height = max(int(image_height), 1)
    canvas_width = max(int(canvas_width), 1)
    canvas_height = max(int(canvas_height), 1)

    scale = min(canvas_width / image_width, canvas_height / image_height)
    view_width = max(1, min(canvas_width, int(round(image_width * scale))))
    view_height = max(1, min(canvas_height, int(round(image_height * scale))))
    left = (canvas_width - view_width) // 2
    top = (canvas_height - view_height) // 2
    return left, top, view_width, view_height


def intersections_from_segmented_controls(
    controls: dict[int, int],
    control_segments: dict[int, int],
    y_anchors: np.ndarray,
) -> list[tuple[float, float]]:
    """
    Interpolate each visible segment independently.

    Pressing K ends the current visible segment. Pressing L creates a new
    visible segment. No interpolation is performed between different segment
    IDs, so the fixed y rows between them remain absent and are saved as x=-1.
    """
    if len(controls) < 2:
        raise ValueError("至少需要两个位于不同 y 行的手动点")

    grouped: dict[int, list[tuple[int, int]]] = {}
    for yi, xb in controls.items():
        segment_id = int(control_segments.get(yi, 0))
        grouped.setdefault(segment_id, []).append((yi, xb))

    # Different visible segments must occupy non-overlapping y ranges.
    # Otherwise one fixed row would receive two x values for the same class.
    ranges = sorted(
        (
            min(yi for yi, _xb in items),
            max(yi for yi, _xb in items),
            segment_id,
        )
        for segment_id, items in grouped.items()
    )
    for (_prev_min, prev_max, prev_id), (curr_min, _curr_max, curr_id) in zip(
        ranges[:-1],
        ranges[1:],
    ):
        if curr_min <= prev_max:
            raise ValueError(
                f"可见段 y 范围重叠：segment {prev_id} 与 segment {curr_id}"
            )

    generated: dict[int, float] = {}
    for segment_id in sorted(grouped):
        ordered = sorted(grouped[segment_id])

        # A one-row visible fragment is still a valid sparse annotation.
        if len(ordered) == 1:
            yi, xb = ordered[0]
            generated[yi] = float(np.clip(xb, 0.0, NUM_X_BINS - 1.0))
            continue

        for (y0, x0), (y1, x1) in zip(ordered[:-1], ordered[1:]):
            if y1 <= y0:
                continue
            for yi in range(y0, y1 + 1):
                ratio = (yi - y0) / float(y1 - y0)
                x_bin = x0 + ratio * (x1 - x0)
                generated[yi] = float(
                    np.clip(x_bin, 0.0, NUM_X_BINS - 1.0)
                )

    return [
        (generated[yi] / float(NUM_X_BINS), float(y_anchors[yi]))
        for yi in sorted(generated)
    ]


def split_points_on_anchor_gaps(
    points: list[tuple[float, float]],
    y_anchors: np.ndarray,
) -> list[list[tuple[float, float]]]:
    """Split sparse saved points wherever one or more fixed rows are missing."""
    rows: dict[int, tuple[float, float]] = {}
    for x_norm, y_norm in points:
        yi = nearest_y_anchor(float(y_norm), y_anchors)
        rows[yi] = (float(x_norm), float(y_norm))

    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    previous_yi: Optional[int] = None

    for yi in sorted(rows):
        if previous_yi is not None and yi - previous_yi > 1:
            if current:
                segments.append(current)
            current = []
        current.append(rows[yi])
        previous_yi = yi

    if current:
        segments.append(current)
    return segments


# ============================================================
# Editor
# ============================================================

class LabelEditor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.split = args.split

        self.datasets_root = resolve_project_path(args.datasets_root)

        # Default layout requested by the project:
        # project/scripts/<script>.py
        # project/datasets/images/...
        # project/datasets/labels/...
        #
        # The image layout alone determines the dataset scope. Original labels
        # are optional. Existing labels_corrected files are resumed first,
        # otherwise the matching datasets/labels file is used.
        #
        # This keeps both supported layouts:
        #   datasets/images/<split>/...  -> labels[_corrected]/<split>/...
        #   datasets/images/...          -> labels[_corrected]/...
        direct_image_dir = self.datasets_root / "images"
        direct_label_dir = self.datasets_root / "labels"
        split_image_dir = direct_image_dir / self.split
        split_label_dir = direct_label_dir / self.split

        if split_image_dir.is_dir():
            self.dataset_scope = self.split
            self.image_dir = split_image_dir
            self.original_label_dir = split_label_dir
            self.corrected_label_dir = self.datasets_root / "labels_corrected" / self.split
            progress_name = self.split
        else:
            self.dataset_scope = "root"
            self.image_dir = direct_image_dir
            self.original_label_dir = direct_label_dir
            self.corrected_label_dir = self.datasets_root / "labels_corrected"
            progress_name = "dataset"

        self.progress_dir = self.datasets_root / "labels_corrected" / ".progress"
        self.progress_path = self.progress_dir / f"{progress_name}.json"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"图片目录不存在: {self.image_dir}\n"
                f"期望项目结构: {PROJECT_ROOT / 'datasets' / 'images'}"
            )
        # Do not create or require the original labels directory. It is only a
        # read-only optional source. New annotations are always written to
        # labels_corrected, preserving the V11 save behavior.
        if not self.original_label_dir.is_dir():
            print(
                f"[提示] 原始标签目录不存在，将从空标签开始标注: "
                f"{self.original_label_dir}"
            )

        self.corrected_label_dir.mkdir(parents=True, exist_ok=True)
        self.progress_dir.mkdir(parents=True, exist_ok=True)

        self.image_paths = find_images(self.image_dir)
        if not self.image_paths:
            raise RuntimeError(f"没有找到图片: {self.image_dir}")

        self.image_index = self.determine_start_index()
        self.image: Optional[np.ndarray] = None
        self.relative_image_path: Optional[Path] = None
        self.original_label_path: Optional[Path] = None
        self.corrected_label_path: Optional[Path] = None
        self.loaded_label_path: Optional[Path] = None
        self.loaded_source = "empty"

        # Important: no default class. User must press 0-4 before drawing.
        self.current_class: Optional[int] = None

        # Whole-line reclassification mode.
        # Press source class number, then M, then target class number.
        self.reclass_source: Optional[int] = None

        # Only actual classes loaded from the label file are stored here.
        self.annotations: dict[int, list[tuple[float, float]]] = {}

        # Editable control points keyed by class and fixed y-row index.
        # Existing annotation points are copied here on the first edit so that
        # newly clicked points and the old line participate in one polyline.
        self.controls: dict[int, dict[int, int]] = {}

        # Per-row visible-segment ID. Rows in different segments are never
        # interpolated together. This is how K/L creates an x=-1 occlusion gap.
        self.control_segments: dict[int, dict[int, int]] = {}
        self.active_segment: dict[int, int] = {}
        self.occlusion_mode: set[int] = set()

        # History entry: (y_row, existed, old_x, old_segment_id).
        self.history: dict[int, list[tuple[int, bool, int, int]]] = {}
        self.controls_initialized: set[int] = set()

        # Dedicated class-1 correction helper. It is independent of the
        # current manual-edit class and only replaces class 1.
        #
        # Key 7 chooses the START construction:
        #   one point  -> midpoint(point, image-left-edge on the same row)
        #   two points -> midpoint(P1, P2)
        #
        # Then choose how to construct the END point:
        #   key 8 + one click -> use that click directly as the end point
        #   key 9 + one click -> midpoint(click, image-left-edge on same row)
        #
        # Finally the two constructed points are snapped to the fixed 56 y
        # anchors / 640 x bins and interpolated exactly as the existing editor
        # line-generation logic does, replacing only class 1.
        self.midpoint_mode = "off"  # off | pick7 | ready | pick8_target | pick9_target
        self.midpoint_reference_points: list[tuple[float, float]] = []
        self.midpoint_point: Optional[tuple[float, float]] = None
        self.midpoint_class: Optional[int] = 1
        self.midpoint_used_left_edge = False
        self.midpoint_end_preview: Optional[tuple[float, float]] = None

        self.y_anchors = make_y_anchor_norms()
        self.crop_top_norm = 1.0 - CROP_RATIO

        # Resizable-window display state. The image is rendered into an
        # aspect-ratio-preserving viewport inside the current window canvas.
        # Mouse coordinates are mapped through this viewport, so resizing the
        # window never changes annotation coordinates.
        self.canvas_width = 0
        self.canvas_height = 0
        self.viewport_x = 0
        self.viewport_y = 0
        self.viewport_width = 1
        self.viewport_height = 1
        self.window_initialized = False
        self.dirty = False
        self.status = ""
        self.status_until = 0.0

        self.window = f"final label editor {SCRIPT_VERSION} [{self.split}]"
        # WINDOW_NORMAL enables manual resize. WINDOW_FREERATIO lets the frame
        # fill the client area; render() performs its own proportional fitting
        # and letterboxing so mouse mapping remains exact.
        window_flags = cv2.WINDOW_NORMAL
        if hasattr(cv2, "WINDOW_FREERATIO"):
            window_flags |= cv2.WINDOW_FREERATIO
        cv2.namedWindow(self.window, window_flags)
        cv2.setMouseCallback(self.window, self.mouse_callback)

        self.load_current_image()
        cv2.resizeWindow(self.window, self.canvas_width, self.canvas_height)
        self.window_initialized = True
        self.write_progress(self.image_index)

    # ---------------- progress ----------------

    def determine_start_index(self) -> int:
        if self.args.start_index is not None:
            return max(0, min(self.args.start_index, len(self.image_paths) - 1))
        if self.args.restart:
            return 0

        if self.progress_path.exists():
            try:
                data = json.loads(self.progress_path.read_text(encoding="utf-8"))
                resume_image = data.get("resume_image")
                if resume_image:
                    target = (self.image_dir / resume_image).resolve()
                    for i, p in enumerate(self.image_paths):
                        if p.resolve() == target:
                            print(f"[断点恢复] {resume_image}")
                            return i
            except Exception as exc:
                print(f"[警告] 断点读取失败: {exc}")

        # This correction pass deliberately starts from the first image when
        # there is no v13-specific progress file. Existing labels_corrected
        # files may come from older tools and must not cause source labels to
        # be skipped.
        return 0

    def write_progress(self, index: int, completed: bool = False) -> None:
        if index >= len(self.image_paths):
            safe_index = len(self.image_paths)
            resume_image = None
            completed = True
        else:
            safe_index = max(index, 0)
            resume_image = str(self.image_paths[safe_index].relative_to(self.image_dir))

        data = {
            "split": self.split,
            "dataset_scope": self.dataset_scope,
            "resume_index": safe_index,
            "resume_image": resume_image,
            "completed": completed,
            "total_images": len(self.image_paths),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "script_version": SCRIPT_VERSION,
        }
        tmp = self.progress_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.progress_path)

    # ---------------- loading ----------------

    def set_status(self, text: str, duration: float = 2.0) -> None:
        self.status = text
        self.status_until = time.monotonic() + duration
        print(text)

    def update_initial_window_size(self) -> None:
        """Choose only the startup size; users may resize freely afterward."""
        assert self.image is not None
        h, w = self.image.shape[:2]
        scale = min(1.0, self.args.max_width / w, self.args.max_height / h)
        self.canvas_width = max(1, int(round(w * scale)))
        self.canvas_height = max(1, int(round(h * scale)))
        self.update_viewport(self.canvas_width, self.canvas_height)

    def current_window_canvas_size(self) -> tuple[int, int]:
        """Read the current drawable window size with a safe fallback."""
        if self.window_initialized and hasattr(cv2, "getWindowImageRect"):
            try:
                _x, _y, width, height = cv2.getWindowImageRect(self.window)
                if width > 1 and height > 1:
                    return int(width), int(height)
            except cv2.error:
                pass
        return max(self.canvas_width, 1), max(self.canvas_height, 1)

    def update_viewport(self, canvas_width: int, canvas_height: int) -> None:
        assert self.image is not None
        h, w = self.image.shape[:2]
        self.canvas_width = max(int(canvas_width), 1)
        self.canvas_height = max(int(canvas_height), 1)
        (
            self.viewport_x,
            self.viewport_y,
            self.viewport_width,
            self.viewport_height,
        ) = fit_image_rect(
            w,
            h,
            self.canvas_width,
            self.canvas_height,
        )

    def window_to_image(self, dx: int, dy: int) -> Optional[tuple[float, float]]:
        """Map current window coordinates to original-image coordinates.

        Returns ``None`` when the pointer lies in the letterbox padding.
        """
        assert self.image is not None

        # Refresh geometry here as well as in render() so a click immediately
        # after dragging a window edge still uses the newest client size.
        canvas_width, canvas_height = self.current_window_canvas_size()
        self.update_viewport(canvas_width, canvas_height)

        vx = self.viewport_x
        vy = self.viewport_y
        vw = self.viewport_width
        vh = self.viewport_height
        if not (vx <= dx < vx + vw and vy <= dy < vy + vh):
            return None

        h, w = self.image.shape[:2]
        x = (dx - vx) * max(w - 1, 1) / max(vw - 1, 1)
        y = (dy - vy) * max(h - 1, 1) / max(vh - 1, 1)
        return (
            float(np.clip(x, 0, max(w - 1, 0))),
            float(np.clip(y, 0, max(h - 1, 0))),
        )

    def choose_label_source(self) -> None:
        """Choose the annotation source requested for the correction workflow.

        Priority:
          1. If labels_corrected already exists, continue from it.
          2. Otherwise, if the original datasets/labels file exists, load it.
          3. If neither exists, start from an empty annotation.

        This also covers the case where the original labels file is missing but
        a corrected file already exists. Key O still forces the original label.
        """
        assert self.original_label_path is not None
        assert self.corrected_label_path is not None

        if self.corrected_label_path.exists():
            self.loaded_label_path = self.corrected_label_path
            self.loaded_source = "corrected"
        elif self.original_label_path.exists():
            self.loaded_label_path = self.original_label_path
            self.loaded_source = "original"
        else:
            self.loaded_label_path = None
            self.loaded_source = "empty"

    def load_current_image(self, force_original: bool = False) -> None:
        image_path = self.image_paths[self.image_index]
        self.relative_image_path = image_path.relative_to(self.image_dir)
        rel_label = self.relative_image_path.with_suffix(".txt")
        self.original_label_path = self.original_label_dir / rel_label
        self.corrected_label_path = self.corrected_label_dir / rel_label

        self.image = read_image(image_path)
        self.update_initial_window_size()

        if force_original:
            if self.original_label_path.exists():
                self.loaded_label_path = self.original_label_path
                self.loaded_source = "original-forced"
            else:
                self.loaded_label_path = None
                self.loaded_source = "empty"
        else:
            self.choose_label_source()

        self.annotations = (
            load_label_file(self.loaded_label_path)
            if self.loaded_label_path is not None
            else {}
        )

        # Never preselect 0 or any other class after loading.
        self.current_class = None
        self.reclass_source = None
        self.controls = {}
        self.control_segments = {}
        self.active_segment = {}
        self.occlusion_mode = set()
        self.history = {}
        self.controls_initialized = set()
        self.reset_midpoint_helper()
        self.dirty = False

        counts = {cid: len(points) for cid, points in self.annotations.items()}
        first_line = "<empty>"
        if self.loaded_label_path is not None and self.loaded_label_path.exists():
            lines = self.loaded_label_path.read_text(encoding="utf-8").splitlines()
            first_line = lines[0][:180] if lines else "<empty file>"

        print("\n" + "=" * 90)
        print(f"[RUNNING SCRIPT] {Path(__file__).resolve()}")
        print(f"[SCRIPT VERSION] {SCRIPT_VERSION}")
        print(f"[IMAGE]          {image_path}")
        print(f"[LABEL SOURCE]   {self.loaded_source}")
        print(f"[LABEL PATH]     {self.loaded_label_path}")
        print(f"[FIRST LINE]     {first_line}")
        print(f"[PARSED]         {counts}")
        print("=" * 90)

        self.set_status(
            f"source={self.loaded_source} | classes={counts} | selected=NONE",
            duration=3.0,
        )

    # ---------------- mouse ----------------

    def require_class(self) -> Optional[int]:
        if self.current_class is None:
            self.set_status("未选择编辑类别：请先按数字键 0-4")
            return None
        return self.current_class

    def reset_midpoint_helper(self) -> None:
        """Reset the dedicated class-1 7/8/9 correction helper."""
        self.midpoint_mode = "off"
        self.midpoint_reference_points = []
        self.midpoint_point = None
        self.midpoint_class = 1
        self.midpoint_used_left_edge = False
        self.midpoint_end_preview = None

    @staticmethod
    def midpoint_with_left_edge(point: tuple[float, float]) -> tuple[float, float]:
        """Return midpoint between a normalized point and x=0 on the same row."""
        return (float(point[0]) * 0.5, float(point[1]))

    @staticmethod
    def midpoint_between_points(
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> tuple[float, float]:
        return (
            (float(p1[0]) + float(p2[0])) * 0.5,
            (float(p1[1]) + float(p2[1])) * 0.5,
        )

    def start_midpoint_helper(self) -> None:
        """Press 7 and select one or two points for the class-1 START midpoint.

        One selected point:
            start = midpoint(P1, left image edge on P1's row)

        Two selected points:
            start = midpoint(P1, P2)

        After at least one point, press 8 or 9 to choose how the END point is
        constructed.
        """
        if self.reclass_source is not None:
            self.set_status("当前处于整线改类模式，请先按 M 取消后再使用 7/8/9")
            return

        self.midpoint_mode = "pick7"
        self.midpoint_reference_points = []
        self.midpoint_point = None
        self.midpoint_class = 1
        self.midpoint_used_left_edge = False
        self.midpoint_end_preview = None
        self.set_status(
            "CLASS 1 修正：按 7 后左键选 1 或 2 个点。"
            "1 点=该点与左边界中点；2 点=两点中点。然后按 8 或 9",
            duration=6.0,
        )

    def arm_midpoint_target(self, mode: int = 8) -> None:
        """Press 8 or 9 after key-7 construction.

        8: the next clicked point is used directly as the end point.
        9: the next clicked point is first averaged with the image left edge;
           that new midpoint is used as the end point.
        """
        if mode not in (8, 9):
            raise ValueError(f"unsupported class1 target mode: {mode}")

        if not self.midpoint_reference_points or self.midpoint_point is None:
            self.set_status(f"请先按 7，并至少左键选择 1 个点后再按 {mode}")
            return

        if self.midpoint_point[1] < self.crop_top_norm:
            self.set_status(
                "按 7 得到的中点位于图片上方 1/3，不能生成 class 1；请重新按 7",
                duration=4.0,
            )
            self.reset_midpoint_helper()
            return

        self.midpoint_end_preview = None
        if mode == 8:
            self.midpoint_mode = "pick8_target"
            self.set_status(
                "8 模式：左键选择 1 个目标点；将直接连接 7 得到的中点与该点，"
                "然后吸附插值并覆盖 class 1",
                duration=6.0,
            )
        else:
            self.midpoint_mode = "pick9_target"
            self.set_status(
                "9 模式：左键选择 1 个目标点；先取该点与同一行左边界的中点，"
                "再连接两个中点并覆盖 class 1",
                duration=6.0,
            )

    def replace_class1_with_midpoint_line(
        self,
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        construction_text: str,
    ) -> bool:
        """Snap/interpolate the constructed segment and replace class 1.

        Both endpoints are snapped to the existing 56 fixed y anchors and
        640 x bins. Only rows between the endpoints are generated. Other
        class-1 rows remain absent in memory and are emitted as x=-1 at save.
        Classes 0/2/3/4 and the normal v11 editing/save/clear logic are not
        modified by this helper.
        """
        if self.image is None:
            return False

        start_x_norm, start_y_norm = start_point
        end_x_norm, end_y_norm = end_point
        if start_y_norm < self.crop_top_norm or end_y_norm < self.crop_top_norm:
            self.set_status("7/8/9 构造的两个端点都必须位于图片下方 2/3")
            return False

        h, w = self.image.shape[:2]
        start_yi = nearest_y_anchor(start_y_norm, self.y_anchors)
        end_yi = nearest_y_anchor(end_y_norm, self.y_anchors)
        if start_yi == end_yi:
            self.set_status(
                "生成失败：两个端点吸附到同一个 y 锚点；请重新按 7 后再选 8/9",
                duration=4.0,
            )
            return False

        start_xb = pixel_x_to_bin(start_x_norm * max(w - 1, 1), w)
        end_xb = pixel_x_to_bin(end_x_norm * max(w - 1, 1), w)

        y0, x0 = start_yi, start_xb
        y1, x1 = end_yi, end_xb
        if y1 < y0:
            y0, y1 = y1, y0
            x0, x1 = x1, x0

        generated: list[tuple[float, float]] = []
        for yi in range(y0, y1 + 1):
            ratio = (yi - y0) / float(y1 - y0)
            xb = x0 + ratio * (x1 - x0)
            xb = float(np.clip(xb, 0.0, NUM_X_BINS - 1.0))
            generated.append((xb / float(NUM_X_BINS), float(self.y_anchors[yi])))

        # New class-1 correction replaces only class 1. Clear pending class-1
        # manual controls so a later save cannot regenerate the old class-1 line.
        self.annotations[1] = generated
        self.controls.pop(1, None)
        self.control_segments.pop(1, None)
        self.active_segment.pop(1, None)
        self.occlusion_mode.discard(1)
        self.history.pop(1, None)
        self.controls_initialized.discard(1)

        self.dirty = True
        self.current_class = 1
        self.set_status(
            f"class 1 已按 {construction_text} 替换：y_row {y0}..{y1}，"
            f"有效点 {len(generated)}；其余 class 1 行保存为 -1。按 S 保存",
            duration=6.0,
        )
        return True

    def handle_midpoint_left_click(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
    ) -> bool:
        """Consume left clicks for the dedicated class-1 7/8/9 helper."""
        if self.midpoint_mode == "off":
            return False

        x_norm = float(np.clip(x / max(width - 1, 1), 0.0, 1.0))
        y_norm = float(np.clip(y / max(height - 1, 1), 0.0, 1.0))

        if y_norm < self.crop_top_norm:
            self.set_status("7/8/9 修正点无效：只能选择图片下方 2/3")
            return True

        point = (x_norm, y_norm)

        if self.midpoint_mode == "pick7":
            if len(self.midpoint_reference_points) >= 2:
                self.set_status("7 已经选满两个点：请按 8 或 9")
                return True

            self.midpoint_reference_points.append(point)
            count = len(self.midpoint_reference_points)

            if count == 1:
                # Important new rule: one point under key 7 immediately means
                # midpoint(P1, left edge), rather than waiting for key 8.
                self.midpoint_point = self.midpoint_with_left_edge(point)
                self.midpoint_used_left_edge = True
                self.set_status(
                    "7：已选 1 点，起点中点=该点与同一行左边界的中点。"
                    "现在可直接按 8/9，或再点第 2 点改为两点中点",
                    duration=6.0,
                )
                return True

            p1, p2 = self.midpoint_reference_points[:2]
            self.midpoint_point = self.midpoint_between_points(p1, p2)
            self.midpoint_used_left_edge = False
            if self.midpoint_point[1] < self.crop_top_norm:
                self.set_status(
                    "7 的两点中点位于图片上方 1/3；请重新按 7",
                    duration=4.0,
                )
                self.reset_midpoint_helper()
                return True

            self.midpoint_mode = "ready"
            self.set_status(
                f"7：两点中点已确定 x={self.midpoint_point[0]:.6f}, "
                f"y={self.midpoint_point[1]:.6f}。现在按 8 或 9",
                duration=5.0,
            )
            return True

        if self.midpoint_mode == "ready":
            self.set_status("7 的中点已经确定：请按 8 或 9")
            return True

        if self.midpoint_mode == "pick8_target":
            assert self.midpoint_point is not None
            self.midpoint_end_preview = point
            success = self.replace_class1_with_midpoint_line(
                self.midpoint_point,
                point,
                "7中点 -> 8目标点",
            )
            if success:
                self.reset_midpoint_helper()
            return True

        if self.midpoint_mode == "pick9_target":
            assert self.midpoint_point is not None
            end_midpoint = self.midpoint_with_left_edge(point)
            self.midpoint_end_preview = end_midpoint
            success = self.replace_class1_with_midpoint_line(
                self.midpoint_point,
                end_midpoint,
                "7中点 -> 9目标点/左边界中点",
            )
            if success:
                self.reset_midpoint_helper()
            return True

        return False

    def ensure_controls_from_annotation(self, class_id: int) -> dict[int, int]:
        """
        Copy an existing sparse line into editable controls on first edit.

        Missing anchor rows in a corrected label are preserved as segment
        breaks. Therefore an existing x=-1 interval is not accidentally filled
        by interpolation when the class is edited again.
        """
        if class_id in self.controls_initialized:
            return self.controls.setdefault(class_id, {})

        controls = self.controls.setdefault(class_id, {})
        segments = self.control_segments.setdefault(class_id, {})
        points = self.annotations.get(class_id, [])

        rows: dict[int, int] = {}
        for x_norm, y_norm in points:
            if y_norm < self.crop_top_norm - 1e-9 or y_norm > 1.0 + 1e-9:
                continue

            yi = nearest_y_anchor(float(y_norm), self.y_anchors)
            xb = int(round(float(x_norm) * NUM_X_BINS))
            rows[yi] = int(np.clip(xb, 0, NUM_X_BINS - 1))

        segment_id = 0
        previous_yi: Optional[int] = None
        for yi in sorted(rows):
            if previous_yi is not None and yi - previous_yi > 1:
                segment_id += 1
            controls[yi] = rows[yi]
            segments[yi] = segment_id
            previous_yi = yi

        self.active_segment[class_id] = segment_id if controls else 0
        self.controls_initialized.add(class_id)
        self.history.setdefault(class_id, [])

        if controls:
            visible_count = len(set(segments.values()))
            self.set_status(
                f"class {class_id}：已载入 {len(controls)} 个有效行，"
                f"识别为 {visible_count} 段可见线",
                duration=2.5,
            )

        return controls


    def mouse_callback(self, event, dx, dy, flags, userdata) -> None:
        if self.image is None:
            return
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return

        mapped = self.window_to_image(dx, dy)
        if mapped is None:
            self.set_status("点击无效：黑边区域不属于图片")
            return

        h, w = self.image.shape[:2]
        x, y = mapped
        y_norm = y / max(h - 1, 1)

        # The 7/8/9 helper is dedicated to class 1 and therefore intentionally
        # does NOT require selecting class 1 (or any class) first.
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.handle_midpoint_left_click(x, y, w, h):
                return

        class_id = self.require_class()
        if class_id is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if y_norm < self.crop_top_norm:
                self.set_status("点击无效：只能标注图片下方 2/3")
                return

            self.ensure_controls_from_annotation(class_id)
            if class_id in self.occlusion_mode:
                self.set_status(
                    f"class {class_id} 当前处于遮挡模式：该区间保存为 -1；"
                    "按 L 后再继续点击可见线",
                    duration=2.5,
                )
                return

            yi = nearest_y_anchor(y_norm, self.y_anchors)
            xb = pixel_x_to_bin(x, w)

            class_controls = self.controls.setdefault(class_id, {})
            class_segments = self.control_segments.setdefault(class_id, {})
            class_history = self.history.setdefault(class_id, [])

            old_exists = yi in class_controls
            old_x = int(class_controls.get(yi, 0))
            old_segment = int(class_segments.get(yi, -1))
            class_history.append(
                (yi, old_exists, old_x, old_segment)
            )

            if old_exists:
                # Replacing an existing row keeps it in its original visible
                # segment. This matters when reopening a label that already
                # contains one or more x=-1 gaps.
                segment_id = old_segment
            else:
                segment_id = int(
                    self.active_segment.setdefault(class_id, 0)
                )
            class_controls[yi] = xb
            class_segments[yi] = segment_id
            self.dirty = True

            self.set_status(
                f"class={class_id}, segment={segment_id}, "
                f"x_bin={xb}, x={x_bin_to_norm(xb):.6f}, "
                f"y_row={yi}, y={self.y_anchors[yi]:.6f}",
                duration=1.0,
            )

        elif event == cv2.EVENT_RBUTTONDOWN:
            class_controls = self.ensure_controls_from_annotation(class_id)
            if not class_controls:
                self.set_status(f"class {class_id} 没有可删除的点")
                return

            class_segments = self.control_segments.setdefault(class_id, {})
            nearest = min(
                class_controls,
                key=lambda yi: (
                    x_bin_to_pixel(class_controls[yi], w) - x
                ) ** 2 + (
                    self.y_anchors[yi] * max(h - 1, 1) - y
                ) ** 2,
            )
            old_x = int(class_controls[nearest])
            old_segment = int(class_segments.get(nearest, 0))
            self.history.setdefault(class_id, []).append(
                (nearest, True, old_x, old_segment)
            )
            del class_controls[nearest]
            class_segments.pop(nearest, None)
            self.dirty = True
            self.set_status(
                f"删除 class {class_id} 手动点 y_row={nearest}"
            )


    # ---------------- editing ----------------

    def undo(self) -> None:
        class_id = self.require_class()
        if class_id is None:
            return
        hist = self.history.get(class_id, [])
        if not hist:
            self.set_status("没有可以撤销的手动操作")
            return

        yi, existed, old_x, old_segment = hist.pop()
        controls = self.controls.setdefault(class_id, {})
        segments = self.control_segments.setdefault(class_id, {})
        if existed:
            controls[yi] = old_x
            segments[yi] = old_segment
        else:
            controls.pop(yi, None)
            segments.pop(yi, None)
        self.dirty = True

    def start_occlusion(self) -> None:
        """Finish the current visible segment and enter x=-1 mode."""
        class_id = self.require_class()
        if class_id is None:
            return

        controls = self.ensure_controls_from_annotation(class_id)
        current_segment = int(self.active_segment.setdefault(class_id, 0))
        current_count = sum(
            1
            for yi in controls
            if self.control_segments.get(class_id, {}).get(yi, 0)
            == current_segment
        )
        if current_count == 0:
            self.set_status(
                f"class {class_id} 当前可见段还没有点，不能按 K 开始遮挡"
            )
            return

        if class_id in self.occlusion_mode:
            self.set_status(
                f"class {class_id} 已处于遮挡模式；按 L 恢复正常标注"
            )
            return

        self.occlusion_mode.add(class_id)
        self.set_status(
            f"class {class_id}：遮挡开始。K 到 L 之间不记录坐标，"
            "最终对应锚点保存为 -1；按 L 新建下一段可见线",
            duration=4.0,
        )

    def resume_visible(self) -> None:
        """Leave x=-1 mode and start a separate visible segment."""
        class_id = self.require_class()
        if class_id is None:
            return

        if class_id not in self.occlusion_mode:
            self.set_status(
                f"class {class_id} 当前不是遮挡模式，不需要按 L"
            )
            return

        self.occlusion_mode.discard(class_id)
        existing = self.control_segments.get(class_id, {})
        next_segment = max(existing.values(), default=-1) + 1
        self.active_segment[class_id] = next_segment
        self.set_status(
            f"class {class_id}：已恢复正常标注，当前为可见段 "
            f"{next_segment}；下一次左键开始记录坐标",
            duration=3.0,
        )

    def clear_controls(self) -> None:
        class_id = self.require_class()
        if class_id is None:
            return
        self.controls.pop(class_id, None)
        self.control_segments.pop(class_id, None)
        self.active_segment.pop(class_id, None)
        self.occlusion_mode.discard(class_id)
        self.history.pop(class_id, None)
        self.controls_initialized.discard(class_id)
        self.dirty = True
        self.set_status(f"已清空 class {class_id} 的手动点")

    def delete_class(self) -> None:
        class_id = self.require_class()
        if class_id is None:
            return
        self.annotations.pop(class_id, None)
        self.controls.pop(class_id, None)
        self.control_segments.pop(class_id, None)
        self.active_segment.pop(class_id, None)
        self.occlusion_mode.discard(class_id)
        self.history.pop(class_id, None)
        self.controls_initialized.discard(class_id)
        self.dirty = True
        self.set_status(f"已删除 class {class_id} 标签")

    def finalize_class(self, class_id: Optional[int] = None) -> bool:
        if class_id is None:
            class_id = self.require_class()
        if class_id is None:
            return False

        controls = self.controls.get(class_id, {})
        if len(controls) < 2:
            self.set_status(f"class {class_id} 至少需要两个不同 y 行的手动点")
            return False

        segment_map = self.control_segments.get(class_id, {})
        try:
            points = intersections_from_segmented_controls(
                controls,
                segment_map,
                self.y_anchors,
            )
        except ValueError as exc:
            self.set_status(
                f"class {class_id} 无法生成标签：{exc}",
                duration=4.0,
            )
            return False

        visible_segments = len(
            {segment_map.get(yi, 0) for yi in controls}
        )

        self.annotations[class_id] = points
        self.controls.pop(class_id, None)
        self.control_segments.pop(class_id, None)
        self.active_segment.pop(class_id, None)
        self.occlusion_mode.discard(class_id)
        self.history.pop(class_id, None)
        self.controls_initialized.discard(class_id)
        self.dirty = True
        self.set_status(
            f"class {class_id} 已生成 {len(points)} 个有效交点，"
            f"共 {visible_segments} 段；段间锚点保存为 -1"
        )
        return True

    def finalize_pending(self) -> bool:
        pending = [cid for cid, pts in self.controls.items() if pts]
        for cid in pending:
            if len(self.controls[cid]) < 2:
                self.set_status(f"无法保存：class {cid} 只有一个手动点")
                return False
        for cid in pending:
            if not self.finalize_class(cid):
                return False
        return True


    def start_reclassify(self) -> None:
        """Enter whole-line class-change mode for the selected class."""
        source = self.require_class()
        if source is None:
            return

        has_annotation = bool(self.annotations.get(source))
        has_controls = bool(self.controls.get(source))
        if not has_annotation and not has_controls:
            self.set_status(f"class {source} 没有可改类别的整条线")
            return

        self.reclass_source = source
        self.set_status(
            f"整线改类：源类别={source}，现在按目标类别数字 0-4；再次按 M 取消",
            duration=4.0,
        )

    def cancel_reclassify(self) -> None:
        if self.reclass_source is None:
            return
        source = self.reclass_source
        self.reclass_source = None
        self.set_status(f"已取消整线改类：源类别={source}")

    def apply_reclassify(self, target: int) -> bool:
        """Move the entire selected line from source class to target class.

        Coordinates, point order, and pending control points remain unchanged.
        Existing target data is never overwritten.
        """
        source = self.reclass_source
        if source is None:
            return False

        if target == source:
            self.reclass_source = None
            self.set_status(f"源类别和目标类别都是 {source}，未做修改")
            return False

        target_has_annotation = bool(self.annotations.get(target))
        target_has_controls = bool(self.controls.get(target))
        if target_has_annotation or target_has_controls:
            self.reclass_source = None
            self.set_status(
                f"改类失败：目标 class {target} 已有线，未覆盖任何数据",
                duration=4.0,
            )
            return False

        moved = False
        if source in self.annotations:
            self.annotations[target] = self.annotations.pop(source)
            moved = True
        if source in self.controls:
            self.controls[target] = self.controls.pop(source)
            moved = True
        if source in self.control_segments:
            self.control_segments[target] = self.control_segments.pop(source)
        if source in self.active_segment:
            self.active_segment[target] = self.active_segment.pop(source)
        if source in self.occlusion_mode:
            self.occlusion_mode.discard(source)
            self.occlusion_mode.add(target)
        if source in self.history:
            self.history[target] = self.history.pop(source)
        if source in self.controls_initialized:
            self.controls_initialized.discard(source)
            self.controls_initialized.add(target)

        self.reclass_source = None
        self.current_class = target

        if not moved:
            self.set_status(f"class {source} 没有可移动的线")
            return False

        self.dirty = True
        point_count = len(self.annotations.get(target, []))
        control_count = len(self.controls.get(target, {}))
        self.set_status(
            f"整条线已从 class {source} 改为 class {target}；"
            f"标签点={point_count}，手动点={control_count}",
            duration=4.0,
        )
        return True

    # ---------------- saving/navigation ----------------

    def save_current(self) -> bool:
        assert self.corrected_label_path is not None
        selected = self.current_class
        if not self.finalize_pending():
            self.current_class = selected
            return False
        self.current_class = selected

        save_label_file_atomic(self.corrected_label_path, self.annotations)
        self.write_progress(self.image_index)
        self.dirty = False
        self.set_status(f"已保存: {self.corrected_label_path}")
        return True

    def move_to(self, index: int) -> None:
        if index < 0:
            self.set_status("已经是第一张")
            return
        if index >= len(self.image_paths):
            self.write_progress(len(self.image_paths), completed=True)
            self.set_status("全部完成")
            return
        self.image_index = index
        self.load_current_image()
        self.write_progress(index)

    # ---------------- drawing ----------------

    def draw_background(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        crop_top = int(round(self.crop_top_norm * max(h - 1, 1)))
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w - 1, max(crop_top - 1, 0)), (0, 0, 0), -1)
        canvas[:] = cv2.addWeighted(canvas, 0.72, overlay, 0.28, 0)
        cv2.line(canvas, (0, crop_top), (w - 1, crop_top), (0, 255, 255), 2, cv2.LINE_AA)

    def draw_annotations(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        for class_id, points in self.annotations.items():
            if not points:
                continue
            color = CLASS_COLORS[class_id]

            visible_segments = split_points_on_anchor_gaps(
                points,
                self.y_anchors,
            )
            all_pixels: list[tuple[int, int]] = []

            for segment_points in visible_segments:
                pixels = [
                    normalized_to_pixel(x, y, w, h)
                    for x, y in segment_points
                ]
                all_pixels.extend(pixels)
                if len(pixels) >= 2:
                    poly = np.asarray(
                        pixels,
                        dtype=np.int32,
                    ).reshape(-1, 1, 2)
                    cv2.polylines(
                        canvas,
                        [poly],
                        False,
                        (0, 0, 0),
                        7,
                        cv2.LINE_AA,
                    )
                    cv2.polylines(
                        canvas,
                        [poly],
                        False,
                        color,
                        3,
                        cv2.LINE_AA,
                    )
                for point in pixels:
                    cv2.circle(
                        canvas,
                        point,
                        3,
                        (0, 0, 0),
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.circle(
                        canvas,
                        point,
                        2,
                        color,
                        -1,
                        cv2.LINE_AA,
                    )

            # Permanent class marker: impossible to confuse class 3 with class 0.
            if all_pixels:
                lx = min(
                    max(all_pixels[0][0] + 8, 5),
                    max(w - 120, 5),
                )
                ly = min(max(all_pixels[0][1] - 8, 22), h - 5)
                label = (
                    f"CLASS {class_id} points={len(points)} "
                    f"segments={len(visible_segments)}"
                )
                cv2.putText(
                    canvas,
                    label,
                    (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    label,
                    (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )


    def draw_controls(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        for class_id, controls in self.controls.items():
            if not controls:
                continue
            color = CLASS_COLORS[class_id]
            segment_map = self.control_segments.get(class_id, {})

            grouped: dict[int, list[tuple[int, int]]] = {}
            for yi, xb in controls.items():
                segment_id = int(segment_map.get(yi, 0))
                grouped.setdefault(segment_id, []).append((yi, xb))

            for segment_id in sorted(grouped):
                ordered = sorted(grouped[segment_id])
                pixels = [
                    (
                        int(round(x_bin_to_pixel(xb, w))),
                        int(
                            round(
                                self.y_anchors[yi]
                                * max(h - 1, 1)
                            )
                        ),
                    )
                    for yi, xb in ordered
                ]

                # Draw each visible segment independently. Never connect K/L
                # separated segments across the x=-1 occlusion interval.
                if len(pixels) >= 2:
                    poly = np.asarray(
                        pixels,
                        dtype=np.int32,
                    ).reshape(-1, 1, 2)
                    cv2.polylines(
                        canvas,
                        [poly],
                        False,
                        (255, 255, 255),
                        7,
                        cv2.LINE_AA,
                    )
                    cv2.polylines(
                        canvas,
                        [poly],
                        False,
                        color,
                        3,
                        cv2.LINE_AA,
                    )
                for point in pixels:
                    cv2.circle(
                        canvas,
                        point,
                        6,
                        (255, 255, 255),
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.circle(
                        canvas,
                        point,
                        3,
                        color,
                        -1,
                        cv2.LINE_AA,
                    )

    def draw_midpoint_helper(self, canvas: np.ndarray) -> None:
        """Visualize temporary class-1 7/8/9 construction points."""
        if self.midpoint_mode == "off":
            return

        h, w = canvas.shape[:2]

        def to_pixel(point: tuple[float, float]) -> tuple[int, int]:
            return normalized_to_pixel(point[0], point[1], w, h)

        ref_pixels = [to_pixel(p) for p in self.midpoint_reference_points]
        for index, point in enumerate(ref_pixels, start=1):
            cv2.circle(canvas, point, 7, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, point, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"P7-{index}",
                (point[0] + 7, point[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if len(ref_pixels) >= 2:
            cv2.line(canvas, ref_pixels[0], ref_pixels[1], (255, 255, 255), 1, cv2.LINE_AA)

        # When key 7 has only one point, show the left-edge construction used
        # to obtain the start midpoint.
        if self.midpoint_used_left_edge and self.midpoint_reference_points:
            p1 = self.midpoint_reference_points[0]
            left_px = to_pixel((0.0, p1[1]))
            cv2.circle(canvas, left_px, 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, left_px, 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.line(canvas, ref_pixels[0], left_px, (255, 255, 255), 1, cv2.LINE_AA)

        if self.midpoint_point is not None:
            midpoint_px = to_pixel(self.midpoint_point)
            cv2.circle(canvas, midpoint_px, 8, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, midpoint_px, 5, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                "MID7",
                (midpoint_px[0] + 8, midpoint_px[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

    def draw_ui(self, canvas: np.ndarray) -> None:
        """Draw a compact text overlay without changing v10 image geometry.

        This remains part of the original-image render pass, exactly as in v10.
        Only text sizing and overflow handling are changed. Annotation points,
        line widths, window sizing, viewport fitting, and mouse mapping are not
        affected.
        """
        h, w = canvas.shape[:2]
        counts = {cid: len(pts) for cid, pts in self.annotations.items()}
        selected = "NONE" if self.current_class is None else str(self.current_class)
        reclass = "OFF" if self.reclass_source is None else f"FROM {self.reclass_source}: PRESS TARGET 0-4"
        if self.current_class is None:
            occlusion = "NO CLASS"
        elif self.current_class in self.occlusion_mode:
            occlusion = "MASKING (-1), PRESS L TO RESUME"
        else:
            segment_id = self.active_segment.get(self.current_class, 0)
            occlusion = f"VISIBLE SEGMENT {segment_id}"
        midpoint_state = self.midpoint_mode.upper()
        lines = [
            f"VERSION={SCRIPT_VERSION}",
            f"{self.split} {self.image_index + 1}/{len(self.image_paths)}  {self.relative_image_path}",
            f"SOURCE={self.loaded_source}  VISIBLE_CLASSES={counts}  SELECTED={selected}",
            f"OCCLUSION={occlusion}",
            f"RECLASS={reclass}",
            f"CLASS1_FIX={midpoint_state}",
            f"LABEL={self.loaded_label_path}",
            "Press 0-4 to select class; no class is selected automatically",
            "Left:add/replace; Right:remove; K:begin x=-1 occlusion",
            "L:resume as a new visible segment; no line crosses the gap",
            "7:pick 1/2 pts -> MID7; 8:direct target; 9:target+left-edge midpoint -> REPLACE class1",
            "Enter:interpolate each visible segment on the fixed 56 rows",
            "M:move whole line to another class (select source, M, target digit)",
            "Z:undo C:clear manual X:delete class S:save D:next A:prev O:load original R:reload Q:quit",
        ]

        # Keep v10's render order and coordinate system. Only the text itself is
        # smaller. Long text is clipped with an ellipsis instead of extending
        # beyond the image and covering unrelated regions.
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.34
        thickness = 1
        line_step = 18
        text_x = 8
        max_text_width = max(w - text_x - 8, 1)

        def fit_line(line: str) -> str:
            width = cv2.getTextSize(
                line,
                font,
                font_scale,
                thickness,
            )[0][0]
            if width <= max_text_width:
                return line

            suffix = "..."
            suffix_width = cv2.getTextSize(
                suffix,
                font,
                font_scale,
                thickness,
            )[0][0]
            if suffix_width >= max_text_width:
                return ""

            low, high = 0, len(line)
            while low < high:
                mid = (low + high + 1) // 2
                candidate = line[:mid] + suffix
                candidate_width = cv2.getTextSize(
                    candidate,
                    font,
                    font_scale,
                    thickness,
                )[0][0]
                if candidate_width <= max_text_width:
                    low = mid
                else:
                    high = mid - 1
            return line[:low] + suffix

        y = 19
        for raw_line in lines:
            line = fit_line(raw_line)
            if not line:
                continue
            size, base = cv2.getTextSize(
                line,
                font,
                font_scale,
                thickness,
            )
            cv2.rectangle(
                canvas,
                (5, y - size[1] - 4),
                (min(w - 1, text_x + size[0] + 4), y + base + 3),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                canvas,
                line,
                (text_x, y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            y += line_step

        # Color legend. Its smaller UI dimensions do not alter annotation point
        # dimensions; draw_controls() still uses the original v10 radii 8 and 5.
        x = 8
        legend_y = y + 2
        box_size = 14
        for cid in sorted(CLASS_COLORS):
            color = CLASS_COLORS[cid]
            cv2.rectangle(
                canvas,
                (x, legend_y),
                (x + box_size, legend_y + box_size),
                color,
                -1,
            )
            cv2.putText(
                canvas,
                str(cid),
                (x + box_size + 4, legend_y + box_size),
                font,
                0.36,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            x += 44

        if self.status and time.monotonic() < self.status_until:
            status_scale = 0.40
            old_scale = font_scale
            font_scale = status_scale
            status_text = fit_line(self.status)
            font_scale = old_scale
            cv2.putText(
                canvas,
                status_text,
                (8, h - 14),
                font,
                status_scale,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                status_text,
                (8, h - 14),
                font,
                status_scale,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def render(self) -> np.ndarray:
        assert self.image is not None

        image_canvas = self.image.copy()
        self.draw_background(image_canvas)
        self.draw_annotations(image_canvas)
        self.draw_controls(image_canvas)
        self.draw_midpoint_helper(image_canvas)
        self.draw_ui(image_canvas)

        canvas_width, canvas_height = self.current_window_canvas_size()
        self.update_viewport(canvas_width, canvas_height)

        interpolation = (
            cv2.INTER_AREA
            if self.viewport_width < image_canvas.shape[1]
            or self.viewport_height < image_canvas.shape[0]
            else cv2.INTER_LINEAR
        )
        fitted = cv2.resize(
            image_canvas,
            (self.viewport_width, self.viewport_height),
            interpolation=interpolation,
        )

        # Neutral padding marks non-image areas. Mouse clicks in this padding
        # are rejected by window_to_image().
        frame = np.full(
            (self.canvas_height, self.canvas_width, 3),
            32,
            dtype=np.uint8,
        )
        x0 = self.viewport_x
        y0 = self.viewport_y
        frame[
            y0:y0 + self.viewport_height,
            x0:x0 + self.viewport_width,
        ] = fitted
        return frame

    # ---------------- run ----------------

    def run(self) -> None:
        print("=" * 90)
        print(f"RUNNING SCRIPT: {Path(__file__).resolve()}")
        print(f"SCRIPT VERSION: {SCRIPT_VERSION}")
        print(f"SCRIPT DIR:     {SCRIPT_DIR}")
        print(f"PROJECT ROOT:   {PROJECT_ROOT}")
        print(f"DATASETS ROOT:  {self.datasets_root}")
        print(f"IMAGE DIR:      {self.image_dir}")
        print(f"LABEL DIR:      {self.original_label_dir}")
        print(f"OUTPUT DIR:     {self.corrected_label_dir}")
        print(f"DATASET SCOPE:  {self.dataset_scope}")
        print("标签读取：已有 labels_corrected 时优先读取；否则读取 datasets/labels；两者都没有则空标签。")
        print("没有默认类别；只有新增普通控制点时才需要先按 0-4 选择类别。")
        print("遮挡标注：完成一段可见线后按 K，遮挡区保存为 -1。")
        print("按 L 恢复正常标注，并开始同一类别的下一段可见线。")
        print("CLASS 1 修正无需先选类别。")
        print("按 7 后选 1 点：自动取该点与同一行图片左边界的中点。")
        print("按 7 后选 2 点：自动取两个点的中点。")
        print("之后按 8，再选 1 点：直接连接 7 中点与该点。")
        print("或者不按 8 而按 9，再选 1 点：先取该点与左边界的中点，再连接两个中点。")
        print("最终按固定 56 行/640 x-bin 吸附插值，并直接覆盖 class 1。")
        print("class 0/2/3/4 保持不变；class 1 线段之外的行保存为 -1。")
        print("整线改类：先按源类别数字，再按 M，再按目标类别数字。")
        print("若目标类别已有线，操作会被拒绝，不会覆盖数据。")
        print("窗口支持拖动边缘自由缩放；图片保持比例，黑边区域不可标注。")
        print("=" * 90)

        while True:
            cv2.imshow(self.window, self.render())
            key = cv2.waitKeyEx(20)
            if key == -1:
                continue
            k = key & 0xFF

            if k in tuple(ord(str(i)) for i in range(5)):
                digit_class = int(chr(k))
                if self.reclass_source is not None:
                    self.apply_reclassify(digit_class)
                else:
                    self.current_class = digit_class
                    exists = self.current_class in self.annotations
                    self.set_status(
                        f"selected class={self.current_class}; "
                        f"loaded={'yes' if exists else 'no'}"
                    )
            elif k == ord("7"):
                self.start_midpoint_helper()
            elif k == ord("8"):
                self.arm_midpoint_target(8)
            elif k == ord("9"):
                self.arm_midpoint_target(9)
            elif k in (ord("m"), ord("M")):
                if self.reclass_source is None:
                    self.start_reclassify()
                else:
                    self.cancel_reclassify()
            elif k in (ord("k"), ord("K")):
                self.start_occlusion()
            elif k in (ord("l"), ord("L")):
                self.resume_visible()
            elif k in (10, 13):
                self.finalize_class()
            elif k in (ord("z"), ord("Z")):
                self.undo()
            elif k in (ord("c"), ord("C")):
                self.clear_controls()
            elif k in (ord("x"), ord("X")):
                self.delete_class()
            elif k in (ord("s"), ord("S")):
                self.save_current()
            elif k in (ord("d"), ord("D")):
                if self.save_current():
                    self.move_to(self.image_index + 1)
            elif k in (ord("a"), ord("A")):
                if self.save_current():
                    self.move_to(self.image_index - 1)
            elif k in (ord("o"), ord("O")):
                self.load_current_image(force_original=True)
                self.write_progress(self.image_index)
            elif k in (ord("r"), ord("R")):
                self.load_current_image()
                self.write_progress(self.image_index)
            elif k in (ord("q"), ord("Q")):
                if self.save_current():
                    self.write_progress(
                        self.image_index + 1,
                        completed=self.image_index + 1 >= len(self.image_paths),
                    )
                    break
            elif k == 27:
                self.write_progress(self.image_index)
                print("当前修改未保存，下次仍从当前图片继续。")
                break

        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "valid"], default="train")
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("datasets"),
        help=(
            "数据集根目录。相对路径按 scripts 的上一级项目目录解析；"
            "默认使用与 scripts 同级的 datasets"
        ),
    )
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--max-width",
        type=int,
        default=1500,
        help="窗口启动时的最大宽度；启动后可自由拖动调整",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=900,
        help="窗口启动时的最大高度；启动后可自由拖动调整",
    )
    return parser.parse_args()


def main() -> None:
    editor = LabelEditor(parse_args())
    editor.run()


if __name__ == "__main__":
    main()
