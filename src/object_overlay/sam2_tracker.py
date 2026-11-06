"""SAM 2.1 视频目标跟踪后端。

torch / sam2 只在真正需要模型推理时导入，因此默认安装和 motion 后端
不依赖这些可选包。人工确认的 Mask 会作为 SAM 2.1 的 conditioning frame
重新注入跟踪状态，并从该帧向后（以及必要时向前）重新传播。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .masking import (
    TrackingHooks,
    _edit_manual_frame,
    _load_saved_mask,
    _select_target_roi,
)
from .video import find_ffmpeg, read_image, write_image


DEFAULT_MODEL = "sam2.1_hiera_small"
DEFAULT_TRACK_FPS = 15.0
MAX_TRACK_FRAMES = 480
TRACKING_MAX_SIDE = 1280
OBJECT_ID = 1
CACHE_VERSION = 1

SAM2_MODELS = {
    "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}

CACHE_DIR_NAME = "masks_sam2"
AUTO_DIR_NAME = "auto"
MANUAL_DIR_NAME = "manual"
TRACKING_FRAMES_DIR_NAME = "sam2_frames"
ANCHORS_FILE_NAME = "anchors.json"
MANIFEST_FILE_NAME = "manifest.json"
TRACKING_META_FILE_NAME = "tracking_meta.json"


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _sha1_array(array: np.ndarray) -> str:
    return hashlib.sha1(array.tobytes()).hexdigest()


def resolve_checkpoint(
    checkpoint: Path | None,
    model_name: str = DEFAULT_MODEL,
) -> Path:
    """按参数、SAM2_CHECKPOINT、项目 checkpoints 目录的顺序找权重。"""
    if checkpoint is not None:
        return checkpoint.expanduser().resolve()

    environment = os.environ.get("SAM2_CHECKPOINT")

    if environment:
        return Path(environment).expanduser().resolve()

    project_root = Path(__file__).resolve().parents[2]
    return (project_root / "checkpoints" / f"{model_name}.pt").resolve()


def _checkpoint_identity(path: Path) -> dict[str, object]:
    identity: dict[str, object] = {"path": str(path)}

    try:
        stat = path.stat()
    except OSError:
        return identity

    identity["size"] = stat.st_size
    identity["mtime_ns"] = stat.st_mtime_ns
    return identity


def _video_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask.astype(np.uint8, copy=False)

    return cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )


class Sam2Tracker:
    """SAM 2.1 视频跟踪状态与 Mask 缓存。

    predictor 参数仅用于测试：传入一个实现官方同名方法的假对象时，不会
    导入 torch / sam2，也不会抽取密集跟踪帧。
    """

    def __init__(
        self,
        video_path: Path,
        frame_files: list[Path],
        images: list[np.ndarray],
        output_dir: Path,
        base_index: int,
        interval: float,
        *,
        model_name: str = DEFAULT_MODEL,
        checkpoint: Path | None = None,
        force: bool = False,
        target_track_fps: float = DEFAULT_TRACK_FPS,
        max_side: int = TRACKING_MAX_SIDE,
        predictor: Any | None = None,
        initial_box: tuple[int, int, int, int] | None = None,
    ) -> None:
        if model_name not in SAM2_MODELS:
            raise ValueError(
                f"不支持的 SAM 2.1 模型：{model_name}，"
                f"可选 {'、'.join(SAM2_MODELS)}"
            )

        if len(frame_files) != len(images):
            raise ValueError("SAM 2.1 后端收到的抽帧与图像数量不一致。")

        self.video_path = video_path.expanduser().resolve()
        self.frame_files = frame_files
        self.images = images
        self.output_dir = output_dir
        self.base_index = base_index
        self.interval = interval
        self.model_name = model_name
        self.checkpoint = resolve_checkpoint(checkpoint, model_name)
        self.force = force
        self.target_track_fps = target_track_fps
        self.max_side = max_side
        self._provided_predictor = predictor
        self.initial_box = initial_box

        self.height, self.width = images[0].shape[:2]
        self.stride = 1
        self.track_fps = 1.0 / interval
        self.track_width = self.width
        self.track_height = self.height
        self.masks = [
            np.zeros((self.height, self.width), dtype=np.uint8)
            for _ in images
        ]
        self.anchor_masks: dict[int, np.ndarray] = {}
        self.initial_mask: np.ndarray | None = None
        self.background_mask: np.ndarray | None = None
        self._predictor: Any | None = None
        self._state: dict[str, Any] | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._runtime_anchor_frames: set[int] = set()
        self._track_frames_ready = False
        self._cache_loaded = False
        self.propagation_log: list[tuple[int, bool, list[int]]] = []

        self.cache_dir = output_dir / CACHE_DIR_NAME
        self.auto_dir = self.cache_dir / AUTO_DIR_NAME
        self.manual_dir = self.cache_dir / MANUAL_DIR_NAME
        self.anchors_path = self.cache_dir / ANCHORS_FILE_NAME
        self.manifest_path = self.cache_dir / MANIFEST_FILE_NAME
        self.track_dir = output_dir / TRACKING_FRAMES_DIR_NAME
        self.track_meta_path = self.track_dir / TRACKING_META_FILE_NAME

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def prepare(self) -> None:
        """读取缓存；缓存无效时初始化目标并完整传播。"""
        self._compute_tracking_plan()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.auto_dir.mkdir(parents=True, exist_ok=True)
        self.manual_dir.mkdir(parents=True, exist_ok=True)

        if self._load_anchors():
            self._override_anchors()

        if (
            not self.force
            and self.anchor_masks
            and self._load_cached_masks()
        ):
            self._set_background_mask()
            print("复用 SAM 2.1 跟踪缓存与人工锚点。")
            return

        self._ensure_tracking_frames()
        self._ensure_runtime()

        if not self.anchor_masks:
            self._initialize_target()
        else:
            self._inject_all_anchors()
            self._propagate_initial()

        self._set_background_mask()
        self._save_cache()

    def hooks(self) -> TrackingHooks:
        return TrackingHooks(
            name="sam2",
            manual_indices=set(self.anchor_masks) - {self.base_index},
            correct=self.correct,
            restore=self.restore,
            keeps_base_mask=True,
            background_manual=self.base_index in self.anchor_masks,
            manual_dir=self.manual_dir,
            background_mask_path=self._manual_path(self.base_index),
        )

    def correct(self, frame_index: int, mask: np.ndarray) -> list[np.ndarray]:
        """注入人工 Mask，并重传播受影响的后继（或前驱）帧。"""
        self._ensure_runtime_for_edit()

        if frame_index < 0 or frame_index >= len(self.images):
            raise IndexError(f"帧下标超出范围：{frame_index}")

        self._add_mask_anchor(frame_index, mask)
        self.anchor_masks[frame_index] = mask.copy()
        self.masks[frame_index] = mask.copy()
        self._propagate_after_change(frame_index)
        self._set_background_mask()
        self._save_cache()
        return [item.copy() for item in self.masks]

    def restore(self, frame_index: int) -> list[np.ndarray]:
        """删除该帧的人工锚点，恢复自动预测。

        初始帧没有可回退的自动传播，恢复为初始化时确认的基准 Mask。
        """
        self._ensure_runtime_for_edit()

        if frame_index < 0 or frame_index >= len(self.images):
            raise IndexError(f"帧下标超出范围：{frame_index}")

        if frame_index not in self.anchor_masks:
            return [item.copy() for item in self.masks]

        if frame_index == self.base_index:
            if self.initial_mask is None:
                return [item.copy() for item in self.masks]

            if np.array_equal(self.anchor_masks[frame_index], self.initial_mask):
                return [item.copy() for item in self.masks]

            self._clear_anchor_prompt(frame_index)
            self._add_mask_anchor(self.base_index, self.initial_mask)
            self.anchor_masks[self.base_index] = self.initial_mask.copy()
            self.masks[self.base_index] = self.initial_mask.copy()
            self._run_propagation(
                self._track_index(self.base_index),
                forward=True,
                reverse=self._track_index(self.base_index) > 0,
            )
        else:
            self._clear_anchor_prompt(frame_index)
            self.anchor_masks.pop(frame_index, None)
            self._delete_manual_mask(frame_index)
            self._propagate_after_change(frame_index)

        self._override_anchors()
        self._set_background_mask()
        self._save_cache()
        return [item.copy() for item in self.masks]

    # ------------------------------------------------------------------
    # 跟踪帧、模型和传播
    # ------------------------------------------------------------------
    def _compute_tracking_plan(self) -> None:
        output_fps = 1.0 / self.interval
        self.stride = max(1, int(round(self.target_track_fps / output_fps)))

        if len(self.images) * self.stride > MAX_TRACK_FRAMES:
            self.stride = max(1, MAX_TRACK_FRAMES // len(self.images))

        self.track_fps = output_fps * self.stride
        print(
            "SAM 2.1 跟踪采样："
            f"{self.track_fps:.2f} FPS（每 {self.stride} 个跟踪帧取 1 个输出帧）"
        )

    def _track_index(self, frame_index: int) -> int:
        return frame_index * self.stride

    def _output_index(self, track_index: int) -> int | None:
        if track_index % self.stride != 0:
            return None

        frame_index = track_index // self.stride

        if 0 <= frame_index < len(self.images):
            return frame_index

        return None

    def _ensure_tracking_frames(self) -> None:
        if self._track_frames_ready:
            return

        if self._provided_predictor is not None:
            self.track_width = self.width
            self.track_height = self.height
            self._track_frames_ready = True
            return

        metadata = self._read_tracking_meta()
        required = len(self.images) * self.stride
        existing = sorted(self.track_dir.glob("*.jpg"))

        if (
            metadata is not None
            and metadata.get("track_fps") == self.track_fps
            and metadata.get("max_side") == self.max_side
            and metadata.get("frames") == len(existing)
            and len(existing) >= required
        ):
            first = read_image(existing[0], cv2.IMREAD_COLOR)

            if first is not None:
                self.track_height, self.track_width = first.shape[:2]
                self._track_frames_ready = True
                return

        if self.track_dir.exists():
            shutil.rmtree(self.track_dir)

        self.track_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg = find_ffmpeg()
        scale_filter = (
            f"fps={self.track_fps:g},"
            f"scale=w='min({self.max_side},iw)':h=-2"
        )
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(self.video_path),
            "-vf",
            scale_filter,
            "-q:v",
            "2",
            "-start_number",
            "0",
            str(self.track_dir / "%05d.jpg"),
        ]

        try:
            import subprocess

            subprocess.run(command, check=True)
        except Exception as exc:
            raise RuntimeError(
                f"SAM 2.1 跟踪帧抽取失败：{exc}"
            ) from exc

        frame_paths = sorted(
            self.track_dir.glob("*.jpg"),
            key=lambda path: int(path.stem),
        )

        if len(frame_paths) < required:
            raise RuntimeError(
                "SAM 2.1 跟踪帧数量不足："
                f"需要至少 {required} 帧，实际 {len(frame_paths)} 帧。"
            )

        first = read_image(frame_paths[0], cv2.IMREAD_COLOR)

        if first is None:
            raise RuntimeError("无法读取 SAM 2.1 跟踪帧。")

        self.track_height, self.track_width = first.shape[:2]
        self.track_meta_path.write_text(
            json.dumps(
                {
                    "video": _video_identity(self.video_path),
                    "track_fps": self.track_fps,
                    "max_side": self.max_side,
                    "frames": len(frame_paths),
                    "size": [self.track_width, self.track_height],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._track_frames_ready = True

    def _read_tracking_meta(self) -> dict[str, Any] | None:
        if not self.track_meta_path.exists():
            return None

        try:
            data = json.loads(
                self.track_meta_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return None

        if data.get("video") != _video_identity(self.video_path):
            return None

        return data

    def _ensure_runtime(self) -> None:
        if self._predictor is not None and self._state is not None:
            return

        self._ensure_tracking_frames()

        if self._provided_predictor is not None:
            predictor = self._provided_predictor
            self._torch = None
            self._device = None
        else:
            try:
                import torch
                from sam2.build_sam import build_sam2_video_predictor
            except ImportError as exc:
                raise RuntimeError(
                    "SAM 2.1 可选依赖未安装。请先运行：\n"
                    "  $env:SAM2_BUILD_CUDA=\"0\"\n"
                    "  uv sync --extra sam2\n"
                    "然后在 NVIDIA GPU 上重试。"
                ) from exc

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "SAM 2.1 需要可用的 CUDA GPU，"
                    "当前 torch.cuda.is_available() 为 False。"
                )

            if not self.checkpoint.is_file():
                raise RuntimeError(
                    "找不到 SAM 2.1 权重："
                    f"{self.checkpoint}\n"
                    "请从 https://dl.fbaipublicfiles.com/"
                    "segment_anything_2/092824/"
                    f"{self.model_name}.pt 下载，"
                    "或用 --sam2-checkpoint 指定路径。"
                )

            print(f"正在加载 SAM 2.1 模型：{self.model_name}")
            predictor = build_sam2_video_predictor(
                SAM2_MODELS[self.model_name],
                str(self.checkpoint),
                device="cuda",
                apply_postprocessing=True,
                vos_optimized=False,
            )
            # 人工修正必须成为 conditioning frame，才能影响后续传播。
            predictor.add_all_frames_to_correct_as_cond = True
            predictor.clear_non_cond_mem_around_input = True
            self._torch = torch
            self._device = torch.device("cuda")

        self._predictor = predictor
        self._state = predictor.init_state(
            video_path=str(self.track_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        self._runtime_anchor_frames = set()
        print(
            "SAM 2.1 状态已初始化："
            f"{self._state.get('num_frames', '?')} 个跟踪帧"
        )

    def _ensure_runtime_for_edit(self) -> None:
        self._ensure_runtime()

        if self._state is None:
            raise RuntimeError("SAM 2.1 跟踪状态未初始化。")

        for frame_index in sorted(self.anchor_masks):
            if frame_index in self._runtime_anchor_frames:
                continue

            self._add_mask_anchor(frame_index, self.anchor_masks[frame_index])

    def _initialize_target(self) -> None:
        if self._state is None:
            raise RuntimeError("SAM 2.1 跟踪状态未初始化。")

        box = self.initial_box
        mask: np.ndarray | None = None
        mask_source = "AUTO"

        while True:
            if mask is None:
                if box is None:
                    box = _select_target_roi(
                        self.images[self.base_index],
                        self.base_index + 1,
                        window_prefix="1 Select SAM2 Target",
                    )

                mask = self._prompt_box(box)
                mask_source = "AUTO"

            action, reviewed = self._review_initial_mask(
                mask,
                source=mask_source,
            )

            if action == "accept":
                self.initial_mask = reviewed.copy()
                self.anchor_masks[self.base_index] = reviewed.copy()
                self.masks[self.base_index] = reviewed.copy()
                self._add_mask_anchor(self.base_index, reviewed)
                self._run_propagation(
                    self._track_index(self.base_index),
                    forward=True,
                    reverse=self._track_index(self.base_index) > 0,
                )
                self._override_anchors()
                return

            if action == "reselect":
                box = None
                mask = None
                continue

            if action == "edit":
                mask = _edit_manual_frame(
                    self.base_index,
                    self.images,
                    self.manual_dir,
                    True,
                    self.width,
                    self.height,
                    mask_path=self._manual_path(self.base_index),
                )
                mask_source = "MANUAL"
                # 修正后回到预览，由用户确认。
                continue

            raise RuntimeError("已取消 SAM 2.1 目标初始化。")

    def _prompt_box(self, box: tuple[int, int, int, int]) -> np.ndarray:
        if self._state is None or self._predictor is None:
            raise RuntimeError("SAM 2.1 跟踪状态未初始化。")

        x, y, object_width, object_height = box
        x1 = x
        y1 = y
        x2 = x + object_width
        y2 = y + object_height
        scale_x = self.track_width / self.width
        scale_y = self.track_height / self.height
        box_input = np.array(
            [
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y,
            ],
            dtype=np.float32,
        )

        _, _, mask_logits = self._predictor.add_new_points_or_box(
            self._state,
            frame_idx=self._track_index(self.base_index),
            obj_id=OBJECT_ID,
            box=box_input,
        )
        return self._mask_from_logits(mask_logits)

    def _review_initial_mask(
        self,
        mask: np.ndarray,
        *,
        source: str = "AUTO",
    ) -> tuple[str, np.ndarray]:
        frame_name = f"frame_{self.base_index + 1:04d}"
        window_name = f"SAM2 Initial Target - {frame_name}"
        display_scale = min(1500 / self.width, 850 / self.height, 1.0)

        print()
        print("=" * 60)
        print(f"SAM 2.1 初始目标：{frame_name}")
        print("Enter 或空格：接受当前 SAM 2.1 Mask")
        print("M：用现有编辑器修正 Mask")
        print("R：重新框选目标")
        print("Q 或 Esc：取消")
        print("=" * 60)

        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                preview = self.images[self.base_index].copy()
                selected = mask > 0

                if np.any(selected):
                    preview[selected] = self.images[self.base_index][selected]
                    pixels = preview[selected].astype(np.float32)
                    green = np.zeros_like(pixels)
                    green[:, 1] = 255
                    preview[selected] = (
                        pixels * 0.68 + green * 0.32
                    ).clip(0, 255).astype(np.uint8)

                cv2.putText(
                    preview,
                    f"SAM2 INITIAL TARGET  {frame_name}  "
                    f"Mask: {source if np.any(mask) else 'EMPTY'}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    preview,
                    "Enter/Space=accept  M=edit  R=reselect  Q=cancel",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                preview = cv2.resize(
                    preview,
                    None,
                    fx=display_scale,
                    fy=display_scale,
                    interpolation=cv2.INTER_AREA,
                )
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(0) & 0xFF

                if key in (10, 13, 32):
                    return "accept", mask

                if key in (ord("m"), ord("M")):
                    return "edit", mask

                if key in (ord("r"), ord("R")):
                    return "reselect", mask

                if key in (27, ord("q"), ord("Q")):
                    return "cancel", mask
        finally:
            cv2.destroyWindow(window_name)

    def _add_mask_anchor(self, frame_index: int, mask: np.ndarray) -> None:
        if self._state is None or self._predictor is None:
            raise RuntimeError("SAM 2.1 跟踪状态未初始化。")

        if self._torch is None:
            # 测试用 predictor 只记录调用，不需要真实 Tensor。
            mask_input = (mask > 0).astype(np.uint8)
        else:
            mask_input = self._torch.from_numpy(mask > 0)

        self._predictor.add_new_mask(
            self._state,
            frame_idx=self._track_index(frame_index),
            obj_id=OBJECT_ID,
            mask=mask_input,
        )
        self._runtime_anchor_frames.add(frame_index)

    def _clear_anchor_prompt(self, frame_index: int) -> None:
        if self._state is None or self._predictor is None:
            raise RuntimeError("SAM 2.1 跟踪状态未初始化。")

        self._predictor.clear_all_prompts_in_frame(
            self._state,
            self._track_index(frame_index),
            OBJECT_ID,
        )
        self._runtime_anchor_frames.discard(frame_index)

    def _inject_all_anchors(self) -> None:
        for frame_index in sorted(self.anchor_masks):
            self._add_mask_anchor(
                frame_index,
                self.anchor_masks[frame_index],
            )

    def _propagate_initial(self) -> None:
        init_track = self._track_index(self.base_index)
        self._run_propagation(
            init_track,
            forward=True,
            reverse=init_track > 0,
        )
        self._override_anchors()

    def _propagate_after_change(self, frame_index: int) -> None:
        track_index = self._track_index(frame_index)
        init_track = self._track_index(self.base_index)

        if track_index < init_track:
            self._run_propagation(track_index, forward=False, reverse=True)
            self._run_propagation(track_index, forward=True, reverse=False)
        else:
            self._run_propagation(track_index, forward=True, reverse=False)

        self._override_anchors()

    def _run_propagation(
        self,
        start_track_index: int,
        *,
        forward: bool,
        reverse: bool,
    ) -> None:
        if self._predictor is None or self._state is None:
            raise RuntimeError("SAM 2.1 跟踪状态未初始化。")

        directions: list[bool] = []

        if reverse:
            directions.append(True)

        if forward:
            directions.append(False)

        for is_reverse in directions:
            seen: list[int] = []

            for track_index, _obj_ids, mask_logits in (
                self._predictor.propagate_in_video(
                    self._state,
                    start_frame_idx=start_track_index,
                    reverse=is_reverse,
                )
            ):
                seen.append(track_index)
                output_index = self._output_index(track_index)

                if output_index is None:
                    continue

                self.masks[output_index] = self._mask_from_logits(
                    mask_logits
                )

            self.propagation_log.append(
                (start_track_index, is_reverse, seen)
            )

    def _mask_from_logits(self, mask_logits: Any) -> np.ndarray:
        array = mask_logits

        if hasattr(array, "detach"):
            array = array.detach().float().cpu().numpy()

        array = np.asarray(array)

        if array.ndim == 4:
            array = array[0, 0]
        elif array.ndim == 3:
            array = array[0]

        mask = (array > 0.0).astype(np.uint8) * 255
        return _resize_mask(mask, self.width, self.height)

    def _override_anchors(self) -> None:
        for frame_index, mask in self.anchor_masks.items():
            if 0 <= frame_index < len(self.masks):
                self.masks[frame_index] = mask.copy()

    def _set_background_mask(self) -> None:
        self.background_mask = self.masks[self.base_index].copy()

    # ------------------------------------------------------------------
    # 缓存
    # ------------------------------------------------------------------
    def _manual_path(self, frame_index: int) -> Path:
        return self.manual_dir / f"frame_{frame_index + 1:04d}.png"

    def _auto_path(self, frame_index: int) -> Path:
        return self.auto_dir / f"frame_{frame_index + 1:04d}.png"

    def _delete_manual_mask(self, frame_index: int) -> None:
        path = self._manual_path(frame_index)

        if path.exists():
            path.unlink()

    def _load_anchors(self) -> bool:
        if not self.anchors_path.exists():
            return False

        try:
            data = json.loads(
                self.anchors_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"SAM 2.1 人工锚点索引无法读取：{self.anchors_path}"
            ) from exc

        if data.get("init_frame") != self.base_index:
            print("背景帧已改变，忽略旧的 SAM 2.1 人工锚点。")
            return False

        frames = sorted(int(value) for value in data.get("frames", []))
        loaded: dict[int, np.ndarray] = {}

        for frame_index in frames:
            path = self._manual_path(frame_index)

            if not path.exists():
                raise RuntimeError(
                    f"SAM 2.1 人工锚点 Mask 缺失：{path}"
                )

            loaded[frame_index] = _load_saved_mask(
                path,
                self.width,
                self.height,
            )

        if self.base_index not in loaded:
            print("SAM 2.1 人工锚点缺少初始帧，需要重新初始化。")
            return False

        self.anchor_masks = loaded
        self.initial_mask = loaded[self.base_index].copy()
        return True

    def _load_cached_masks(self) -> bool:
        if not self.manifest_path.exists():
            return False

        try:
            manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return False

        if not self._manifest_matches(manifest):
            return False

        masks: list[np.ndarray] = []
        mask_hashes = manifest.get("masks", {})

        for frame_index in range(len(self.images)):
            path = self._auto_path(frame_index)

            if not path.exists():
                return False

            if mask_hashes.get(str(frame_index)) != _sha1_file(path):
                return False

            masks.append(
                _load_saved_mask(path, self.width, self.height)
            )

        self.masks = masks
        self._cache_loaded = True
        return True

    def _manifest_matches(self, manifest: dict[str, Any]) -> bool:
        if manifest.get("version") != CACHE_VERSION:
            return False

        if manifest.get("tracker") != "sam2":
            return False

        if manifest.get("video") != _video_identity(self.video_path):
            return False

        if manifest.get("frames") != len(self.images):
            return False

        if abs(
            float(manifest.get("interval", -1.0)) - self.interval
        ) > 1e-9:
            return False

        if manifest.get("model") != self.model_name:
            return False

        if manifest.get("init_frame") != self.base_index:
            return False

        if manifest.get("stride") != self.stride:
            return False

        if abs(
            float(manifest.get("track_fps", -1.0)) - self.track_fps
        ) > 1e-9:
            return False

        checkpoint = manifest.get("checkpoint", {})

        if checkpoint.get("path") != str(self.checkpoint):
            return False

        current_checkpoint = _checkpoint_identity(self.checkpoint)

        if (
            "size" in checkpoint
            and "size" in current_checkpoint
            and checkpoint["size"] != current_checkpoint["size"]
        ):
            return False

        anchors = {
            str(frame_index): _sha1_file(self._manual_path(frame_index))
            for frame_index in self.anchor_masks
        }

        if manifest.get("anchors") != anchors:
            return False

        return True

    def _save_cache(self) -> None:
        self.auto_dir.mkdir(parents=True, exist_ok=True)
        self.manual_dir.mkdir(parents=True, exist_ok=True)

        masks_hashes: dict[str, str] = {}

        for frame_index, mask in enumerate(self.masks):
            path = self._auto_path(frame_index)
            write_image(path, mask)
            masks_hashes[str(frame_index)] = _sha1_file(path)

        anchors_hashes: dict[str, str] = {}

        for frame_index, mask in sorted(self.anchor_masks.items()):
            path = self._manual_path(frame_index)
            write_image(path, mask)
            anchors_hashes[str(frame_index)] = _sha1_file(path)

        self.anchors_path.write_text(
            json.dumps(
                {
                    "init_frame": self.base_index,
                    "frames": sorted(self.anchor_masks),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.manifest_path.write_text(
            json.dumps(
                {
                    "version": CACHE_VERSION,
                    "tracker": "sam2",
                    "video": _video_identity(self.video_path),
                    "frames": len(self.images),
                    "interval": self.interval,
                    "model": self.model_name,
                    "checkpoint": _checkpoint_identity(self.checkpoint),
                    "init_frame": self.base_index,
                    "stride": self.stride,
                    "track_fps": self.track_fps,
                    "tracking_size": [
                        self.track_width,
                        self.track_height,
                    ],
                    "anchors": anchors_hashes,
                    "masks": masks_hashes,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def load_sam2_tracker(
    video_path: Path,
    frame_files: list[Path],
    images: list[np.ndarray],
    output_dir: Path,
    base_index: int,
    interval: float,
    *,
    model_name: str = DEFAULT_MODEL,
    checkpoint: Path | None = None,
    force: bool = False,
    target_track_fps: float = DEFAULT_TRACK_FPS,
) -> Sam2Tracker:
    tracker = Sam2Tracker(
        video_path,
        frame_files,
        images,
        output_dir,
        base_index,
        interval,
        model_name=model_name,
        checkpoint=checkpoint,
        force=force,
        target_track_fps=target_track_fps,
    )
    tracker.prepare()
    return tracker
