#!/usr/bin/env python3
"""Generate LaneRobotV3B pre-label txt files for an external image dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import infer_onnx_v3b as v3b  # noqa: E402


# 直接复用当前 infer_onnx_v3b.py 所使用的基础推理/解码函数。
base = v3b.base


# ----------------------------------------------------------------------
# 默认路径
# ----------------------------------------------------------------------

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "runs/lane/lane_v3b-4/weights/best.onnx"
)

DEFAULT_SOURCE = Path(
    "/home/xhm/Desktop/fixing/datasets/images"
)

DEFAULT_LABELS = Path(
    "/home/xhm/Desktop/fixing/datasets/labels"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use the LaneRobotV3B ONNX model to generate "
            "pre-label txt files."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"ONNX model path. Default: {DEFAULT_MODEL}",
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Image directory. Default: {DEFAULT_SOURCE}",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help=f"Output label directory. Default: {DEFAULT_LABELS}",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="ONNX Runtime device. Default: auto",
    )

    parser.add_argument(
        "--exist-thr",
        type=float,
        default=0.5,
        help="Lane existence threshold. Default: 0.5",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Top-K used by soft-argmax. Default: 5",
    )

    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable polynomial smoothing.",
    )

    parser.add_argument(
        "--poly-degree",
        type=int,
        default=2,
        help="Polynomial smoothing degree. Default: 2",
    )

    parser.add_argument(
        "--poly-blend",
        type=float,
        default=0.5,
        help="Polynomial smoothing blend. Default: 0.5",
    )

    parser.add_argument(
        "--letterbox",
        action="store_true",
        help=(
            "Use LaneRobot letterbox preprocessing. "
            "DO NOT enable unless the checkpoint was trained "
            "with the same letterbox policy."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing txt labels. "
            "Default behavior is to skip existing labels."
        ),
    )

    return parser.parse_args()


def label_path_for(
    image_path: Path,
    source: Path,
    labels_root: Path,
) -> Path:
    """
    保持 images 下的目录结构。

    例如：

        images/train/001.jpg
        ->
        labels/train/001.txt

        images/valid/002.png
        ->
        labels/valid/002.txt
    """

    if source.is_file():
        relative = Path(image_path.name)
    else:
        relative = image_path.relative_to(source)

    return (labels_root / relative).with_suffix(".txt")


def main() -> None:
    args = parse_args()

    model_path = args.model.expanduser().resolve()
    source = args.source.expanduser().resolve()
    labels_root = args.labels.expanduser().resolve()

    # ------------------------------------------------------------------
    # 参数检查
    # ------------------------------------------------------------------

    if not model_path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}"
        )

    if not 0.0 <= args.exist_thr <= 1.0:
        raise ValueError(
            "--exist-thr must be in [0, 1]"
        )

    if args.topk < 1:
        raise ValueError(
            "--topk must be >= 1"
        )

    if args.poly_degree < 0:
        raise ValueError(
            "--poly-degree must be >= 0"
        )

    if not 0.0 <= args.poly_blend <= 1.0:
        raise ValueError(
            "--poly-blend must be in [0, 1]"
        )

    # ------------------------------------------------------------------
    # ONNX Runtime
    # ------------------------------------------------------------------

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is not installed.\n"
            "CPU:\n"
            "  python -m pip install onnxruntime\n"
            "GPU:\n"
            "  python -m pip install onnxruntime-gpu"
        ) from exc

    # ------------------------------------------------------------------
    # 找图片
    # ------------------------------------------------------------------

    image_paths = base.scan_images(source)

    providers = base.choose_providers(
        ort,
        args.device,
    )

    # ------------------------------------------------------------------
    # 创建 ONNX Session
    # ------------------------------------------------------------------

    session = ort.InferenceSession(
        str(model_path),
        providers=providers,
    )

    # ------------------------------------------------------------------
    # 严格检查 V3B ONNX
    #
    # input:
    #   images [1,3,256,448]
    #
    # output:
    #   lane_output [1,322,56,4]
    # ------------------------------------------------------------------

    input_info = session.get_inputs()[0]
    input_name = input_info.name

    input_hw = v3b.get_v3b_input_hw(
        session
    )

    output_infos = session.get_outputs()

    output_names = [
        item.name
        for item in output_infos
    ]

    x_grids, output_layout = (
        v3b.inspect_v3b_output_layout(
            output_infos,
            v3b.V3B_X_GRIDS,
        )
    )

    labels_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # 打印配置
    # ------------------------------------------------------------------

    print("=" * 72)
    print(
        "LaneRobotV3B external-dataset "
        "pre-label inference"
    )
    print(f"model       : {model_path}")
    print(f"source      : {source}")
    print(f"labels      : {labels_root}")
    print(f"images      : {len(image_paths)}")
    print(
        f"input       : {input_name} "
        f"{input_hw[0]}x{input_hw[1]}"
    )
    print(f"outputs     : {output_names}")
    print(f"layout      : {output_layout}")
    print(f"x_grids     : {x_grids}")
    print(
        f"providers   : "
        f"{session.get_providers()}"
    )
    print(f"exist_thr   : {args.exist_thr}")
    print(f"topk        : {args.topk}")
    print(f"letterbox   : {args.letterbox}")
    print(
        f"smoothing   : "
        f"{not args.no_smooth}"
    )
    print(f"overwrite   : {args.overwrite}")
    print("=" * 72)

    completed = 0
    skipped = 0
    failed = 0

    # ------------------------------------------------------------------
    # 开始逐张推理
    # ------------------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        txt_path = label_path_for(
            image_path,
            source,
            labels_root,
        )

        # --------------------------------------------------------------
        # 默认不覆盖已有标签。
        #
        # 这样如果 labels 中已经有人工标注，
        # 不会被模型预测直接覆盖。
        # --------------------------------------------------------------

        if (
            txt_path.exists()
            and not args.overwrite
        ):
            print(
                f"[{index}/{len(image_paths)}] "
                f"SKIP {image_path.name} "
                f"-> {txt_path} "
                f"(label exists)"
            )

            skipped += 1
            continue

        try:
            # ----------------------------------------------------------
            # 读取 RGB 图片
            # ----------------------------------------------------------

            rgb_image = base.load_image_rgb(
                image_path
            )

            # ----------------------------------------------------------
            # 和 infer_onnx_v3b 一样的预处理
            #
            # 默认：
            #
            # RGB
            # -> direct resize 448x256
            # -> float32
            # -> /255
            # -> CHW
            # -> batch
            #
            # 不传 --letterbox 时就是直接 resize。
            # ----------------------------------------------------------

            (
                input_tensor,
                letterbox_meta,
            ) = base.preprocess_with_policy(
                rgb_image,
                input_hw,
                letterbox=args.letterbox,
            )

            # ----------------------------------------------------------
            # ONNX 推理
            # ----------------------------------------------------------

            outputs = session.run(
                output_names,
                {
                    input_name: input_tensor
                },
            )

            # ----------------------------------------------------------
            # V3B 输出：
            #
            # lane_output:
            # [1,322,56,4]
            #
            # ->
            #
            # cls_logits:
            # [1,321,56,4]
            #
            # offset:
            # [1,1,56,4]
            # ----------------------------------------------------------

            (
                cls_logits,
                offset,
            ) = v3b.split_v3b_output(
                outputs,
                expected_x_grids=(
                    v3b.V3B_X_GRIDS
                ),
            )

            # ----------------------------------------------------------
            # 与原推理脚本完全相同的 lane decode
            # ----------------------------------------------------------

            (
                lanes_batch,
                decoded_x_grids,
            ) = base.decode_lanes(
                cls_logits=cls_logits,
                offset=offset,
                topk=args.topk,
                exist_thr=args.exist_thr,
                smooth=not args.no_smooth,
                poly_degree=args.poly_degree,
                poly_blend=args.poly_blend,
            )

            lanes = lanes_batch[0]

            # 默认 direct resize 时这里保持 None。
            result_row_y = None

            # ----------------------------------------------------------
            # 如果手动启用了 letterbox，
            # 将预测坐标恢复到原图坐标体系。
            # ----------------------------------------------------------

            if letterbox_meta is not None:
                model_row_y = np.linspace(
                    1.0,
                    0.333333,
                    lanes.shape[0],
                    dtype=np.float32,
                )

                (
                    lanes,
                    result_row_y,
                ) = (
                    base.restore_lanes_from_letterbox(
                        lanes,
                        model_row_y,
                        decoded_x_grids,
                        letterbox_meta,
                    )
                )

            # ----------------------------------------------------------
            # 保存为训练标签格式
            #
            # lane_id x1 y1 x2 y2 ... x56 y56
            #
            # 不存在的点：
            #
            # x = -1
            #
            # 如果整张图片一条线都没有，
            # 会生成一个空 txt。
            # ----------------------------------------------------------

            base.save_prediction_txt(
                txt_path=txt_path,
                lanes=lanes,
                x_grids=decoded_x_grids,
                row_y=result_row_y,
            )

            # ----------------------------------------------------------
            # 仅用于终端显示预测到了哪些 lane。
            # 不生成可视化图片。
            # ----------------------------------------------------------

            active_lane_ids = [
                lane_id
                for lane_id
                in range(lanes.shape[1])
                if np.any(
                    lanes[:, lane_id] >= 0
                )
            ]

            if active_lane_ids:
                lane_text = ", ".join(
                    base.LANE_NAMES.get(
                        lane_id,
                        f"lane_{lane_id}",
                    )
                    for lane_id
                    in active_lane_ids
                )
            else:
                lane_text = "no lanes"

            print(
                f"[{index}/{len(image_paths)}] "
                f"OK   "
                f"{image_path.name} "
                f"-> {txt_path} "
                f"({lane_text})"
            )

            completed += 1

        except Exception as exc:
            print(
                f"[{index}/{len(image_paths)}] "
                f"FAIL {image_path}: {exc}",
                file=sys.stderr,
            )

            failed += 1

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "Pre-label inference complete"
    )
    print(f"completed : {completed}")
    print(f"skipped   : {skipped}")
    print(f"failed    : {failed}")
    print(f"labels    : {labels_root}")
    print("=" * 72)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
