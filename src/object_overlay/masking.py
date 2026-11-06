import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .text import draw_text
from .video import read_image, write_image


MANUAL_ROI_PADDING = 50
INITIAL_BRUSH_RADIUS = 10

# 背景帧人工修正复用 masks_corrected 目录，但文件名与普通候选帧分开，
# 避免改变 --base-frame 后把旧的背景 掩膜 误当成候选帧修正。
BACKGROUND_CACHE_MASK_NAME = "background_mask.png"
BACKGROUND_CACHE_STATE_NAME = "background.json"
BACKGROUND_SOURCE_AUTO = "auto"
BACKGROUND_SOURCE_MANUAL = "manual"
BACKGROUND_SOURCE_SKIPPED = "skipped"

MOTION_SCALE = 0.25
MOTION_ROI_PADDING = 80
MOTION_MIN_AREA_RATIO = 0.00002
MOTION_MAX_AREA_RATIO = 0.08
MOTION_MAX_WIDTH_RATIO = 0.40
MOTION_MAX_HEIGHT_RATIO = 0.40


@dataclass(frozen=True)
class SelectionResult:
    """逐帧复核的结果。

    selected_masks 与抽帧等长，背景帧位置恒为空 掩膜，不参与无损叠加；
    background_mask 是用户最终确认的背景帧 掩膜。None 只表示用户主动
    跳过了背景目标，与"自动检测失败返回空 掩膜"是两种不同状态。
    """

    selected_masks: list[np.ndarray]
    background_mask: np.ndarray | None = None


@dataclass
class TrackingHooks:
    """逐帧复核与跟踪后端之间的最小接口。

    correct/restore 接收候选帧下标和当前 掩膜，返回与所有抽帧等长的更新后
    掩膜 列表。这样 SAM 2.1 可以在用户按 M 后重传播下游帧，而 GUI 本身
    仍然只负责选择和展示。
    """

    name: str = "motion"
    manual_indices: set[int] = field(default_factory=set)
    correct: Callable[[int, np.ndarray], list[np.ndarray]] | None = None
    restore: Callable[[int], list[np.ndarray]] | None = None
    keeps_base_mask: bool = False
    background_manual: bool = False
    manual_dir: Path | None = None
    background_mask_path: Path | None = None


def _load_saved_mask(
    mask_path: Path,
    width: int,
    height: int,
) -> np.ndarray:
    mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise RuntimeError(f"读取 掩膜 失败：{mask_path}")

    if mask.shape != (height, width):
        raise RuntimeError(f"掩膜 尺寸与视频帧不一致：{mask_path}")

    return mask


def _select_target_roi(
    image: np.ndarray,
    frame_number: int,
    window_prefix: str = "1 框选目标",
    instruction: str = "用鼠标框住目标，然后按回车",
) -> tuple[int, int, int, int]:
    """在完整画面中框选目标，返回原分辨率坐标 (x, y, w, h)。"""
    height, width = image.shape[:2]
    max_width = 1500
    max_height = 850
    display_scale = min(
        max_width / width,
        max_height / height,
        1.0,
    )
    preview = cv2.resize(
        image,
        None,
        fx=display_scale,
        fy=display_scale,
        interpolation=cv2.INTER_AREA,
    )

    # 说明直接画在框选窗口里，不能只打印到控制台。
    title_size = max(18, min(30, int(round(preview.shape[0] / 35))))
    hint_size = max(14, int(round(title_size * 0.7)))
    banner_height = title_size + hint_size + 18
    banner = preview.copy()
    cv2.rectangle(
        banner,
        (0, 0),
        (preview.shape[1], banner_height),
        (0, 0, 0),
        -1,
    )
    preview = cv2.addWeighted(banner, 0.55, preview, 0.45, 0)
    draw_text(
        preview,
        instruction,
        (12, 4),
        size=title_size,
        color=(0, 255, 255),
    )
    draw_text(
        preview,
        "回车/空格：确认    C：取消",
        (12, 4 + title_size + 2),
        size=hint_size,
        color=(255, 255, 255),
        stroke=1,
    )
    select_window = f"{window_prefix} - frame_{frame_number:04d}"

    print()
    print("=" * 60)
    print(f"frame_{frame_number:04d}")
    print("第一步：鼠标框住完整目标对象")
    print("框好以后按回车")
    print("=" * 60)

    try:
        roi_small = cv2.selectROI(
            select_window,
            preview,
            showCrosshair=True,
            fromCenter=False,
        )
    finally:
        cv2.destroyWindow(select_window)

    sx, sy, sw, sh = roi_small

    if sw == 0 or sh == 0:
        raise RuntimeError(f"frame_{frame_number:04d} 未完成框选")

    return (
        int(sx / display_scale),
        int(sy / display_scale),
        int(sw / display_scale),
        int(sh / display_scale),
    )


def _edit_manual_frame(
    frame_index: int,
    images: list[np.ndarray],
    mask_dir: Path,
    force: bool,
    width: int,
    height: int,
    mask_path: Path | None = None,
) -> np.ndarray:
    frame_number = frame_index + 1
    image = images[frame_index]
    mask_path = (
        mask_path
        if mask_path is not None
        else mask_dir / f"frame_{frame_number:04d}.png"
    )

    if mask_path.exists() and not force:
        print(f"frame_{frame_number:04d}：读取已有手动 掩膜")
        return _load_saved_mask(mask_path, width, height)

    # 第一步：在完整画面中框选目标对象。
    x, y, object_width, object_height = _select_target_roi(
        image,
        frame_number,
    )

    # 第二步：在局部区域内使用 GrabCut 生成初始 掩膜。
    x0 = max(0, x - MANUAL_ROI_PADDING)
    y0 = max(0, y - MANUAL_ROI_PADDING)
    x1 = min(width, x + object_width + MANUAL_ROI_PADDING)
    y1 = min(height, y + object_height + MANUAL_ROI_PADDING)

    crop = image[y0:y1, x0:x1].copy()
    crop_height, crop_width = crop.shape[:2]
    grabcut_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)

    rect_x = x - x0
    rect_y = y - y0
    rect_width = object_width
    rect_height = object_height
    rect = (
        max(1, rect_x),
        max(1, rect_y),
        max(2, min(rect_width, crop_width - rect_x - 1)),
        max(2, min(rect_height, crop_height - rect_y - 1)),
    )

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        crop,
        grabcut_mask,
        rect,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_RECT,
    )

    initial_mask = np.where(
        (grabcut_mask == cv2.GC_FGD)
        | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    # 第三步：由用户补充或擦除 掩膜。
    work_mask = initial_mask.copy()
    state = {
        "drawing": False,
        "erase": False,
        "radius": INITIAL_BRUSH_RADIUS,
    }

    zoom = min(1200 / crop_width, 800 / crop_height)
    zoom = max(1.0, min(5.0, zoom))
    edit_window = f"2 编辑掩膜 - frame_{frame_number:04d}"
    cv2.namedWindow(edit_window, cv2.WINDOW_AUTOSIZE)

    def draw_at(mouse_x: int, mouse_y: int, erase: bool = False) -> None:
        pixel_x = int(mouse_x / zoom)
        pixel_y = int(mouse_y / zoom)
        pixel_x = int(np.clip(pixel_x, 0, crop_width - 1))
        pixel_y = int(np.clip(pixel_y, 0, crop_height - 1))
        cv2.circle(
            work_mask,
            (pixel_x, pixel_y),
            state["radius"],
            0 if erase else 255,
            -1,
        )

    def mouse_callback(
        event: int,
        mouse_x: int,
        mouse_y: int,
        flags: int,
        param: object,
    ) -> None:
        del flags, param

        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["erase"] = False
            draw_at(mouse_x, mouse_y, False)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["drawing"] = True
            state["erase"] = True
            draw_at(mouse_x, mouse_y, True)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            draw_at(mouse_x, mouse_y, state["erase"])
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            state["drawing"] = False

    cv2.setMouseCallback(edit_window, mouse_callback)

    print("第二步：检查绿色区域")
    print("左键拖动：补充目标")
    print("右键拖动：擦除背景")
    print("] 或 +：增大画笔")
    print("[ 或 -：减小画笔")
    print("C：清空掩膜")
    print("R：恢复 GrabCut 初始掩膜")
    print("A：将整个框选矩形作为前景")
    print("回车：保存并进入下一帧")

    while True:
        display = crop.copy()
        overlay = display.copy()
        overlay[work_mask > 0] = (0, 255, 0)
        display = cv2.addWeighted(display, 0.72, overlay, 0.28, 0)
        draw_text(
            display,
            f"frame_{frame_number:04d}  画笔={state['radius']} 像素",
            (10, 5),
            size=20,
            color=(0, 255, 255),
        )
        display_big = cv2.resize(
            display,
            None,
            fx=zoom,
            fy=zoom,
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow(edit_window, display_big)
        key = cv2.waitKey(20) & 0xFF

        if key in (10, 13):
            break
        if key in (ord("c"), ord("C")):
            work_mask[:] = 0
        elif key in (ord("r"), ord("R")):
            work_mask[:] = initial_mask
        elif key in (ord("a"), ord("A")):
            work_mask[:] = 0
            work_mask[
                rect_y : rect_y + rect_height,
                rect_x : rect_x + rect_width,
            ] = 255
        elif key in (ord("]"), ord("+"), ord("=")):
            state["radius"] = min(100, state["radius"] + 3)
        elif key in (ord("["), ord("-"), ord("_")):
            state["radius"] = max(1, state["radius"] - 3)

    cv2.destroyWindow(edit_window)

    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = work_mask
    write_image(mask_path, full_mask)
    print(f"保存手动 掩膜：{mask_path.name}")
    return full_mask


def _build_motion_background(
    images: list[np.ndarray],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    small_images = [
        cv2.resize(
            image,
            None,
            fx=MOTION_SCALE,
            fy=MOTION_SCALE,
            interpolation=cv2.INTER_AREA,
        )
        for image in images
    ]
    sample_count = min(21, len(small_images))
    sample_ids = np.linspace(
        0,
        len(small_images) - 1,
        sample_count,
        dtype=int,
    )
    stack = np.stack([small_images[index] for index in sample_ids], axis=0)
    background = np.median(stack, axis=0).astype(np.uint8)
    return small_images, background, sample_ids


def _detect_motion_bbox(
    small_image: np.ndarray,
    background_lab: np.ndarray,
    previous_center: tuple[float, float] | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int, tuple[float, float]] | None:
    current_lab = cv2.cvtColor(
        small_image,
        cv2.COLOR_BGR2LAB,
    ).astype(np.int16)
    difference = np.linalg.norm(current_lab - background_lab, axis=2)

    min_area = width * height * MOTION_MIN_AREA_RATIO * MOTION_SCALE**2
    max_area = width * height * MOTION_MAX_AREA_RATIO * MOTION_SCALE**2
    max_width = width * MOTION_MAX_WIDTH_RATIO * MOTION_SCALE
    max_height = height * MOTION_MAX_HEIGHT_RATIO * MOTION_SCALE

    for threshold in (24, 20, 16, 12, 9):
        binary = (difference > threshold).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )
        count, labels, stats, centers = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        candidates: list[tuple[float, int, int, int, int, float, float]] = []

        for component_index in range(1, count):
            x, y, object_width, object_height, area = stats[component_index]
            center_x, center_y = centers[component_index]

            if area < min_area or area > max_area:
                continue
            if object_width > max_width or object_height > max_height:
                continue

            component = labels == component_index
            mean_difference = float(difference[component].mean())
            score = mean_difference * np.sqrt(float(area))

            if previous_center is not None:
                previous_x, previous_y = previous_center
                distance = np.hypot(
                    center_x - previous_x,
                    center_y - previous_y,
                )
                score -= 0.8 * distance

            candidates.append(
                (
                    score,
                    x,
                    y,
                    object_width,
                    object_height,
                    center_x,
                    center_y,
                )
            )

        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            _, x, y, object_width, object_height, center_x, center_y = candidates[0]
            return (
                int(x / MOTION_SCALE),
                int(y / MOTION_SCALE),
                max(2, int(object_width / MOTION_SCALE)),
                max(2, int(object_height / MOTION_SCALE)),
                (center_x, center_y),
            )

    return None


def _refine_motion_mask(
    frame_index: int,
    bbox: tuple[int, int, int, int],
    sample_ids: np.ndarray,
    images: list[np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    image = images[frame_index]
    x, y, object_width, object_height = bbox

    x0 = max(0, x - MOTION_ROI_PADDING)
    y0 = max(0, y - MOTION_ROI_PADDING)
    x1 = min(width, x + object_width + MOTION_ROI_PADDING)
    y1 = min(height, y + object_height + MOTION_ROI_PADDING)

    crop = image[y0:y1, x0:x1].copy()
    crop_height, crop_width = crop.shape[:2]
    references = [
        images[int(index)][y0:y1, x0:x1]
        for index in sample_ids
        if int(index) != frame_index
    ]

    if len(references) < 3:
        references = [
            images[index][y0:y1, x0:x1]
            for index in range(len(images))
            if index != frame_index
        ][:5]

    if not references:
        raise RuntimeError("自动提取运动目标至少需要两张抽帧。")

    local_background = np.median(
        np.stack(references, axis=0),
        axis=0,
    ).astype(np.uint8)
    current_lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.int16)
    reference_lab = cv2.cvtColor(
        local_background,
        cv2.COLOR_BGR2LAB,
    ).astype(np.int16)
    local_difference = np.linalg.norm(current_lab - reference_lab, axis=2)

    # 使用 GrabCut 细化自动定位得到的目标区域。
    grabcut_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
    rect_x = max(1, x - x0 - 12)
    rect_y = max(1, y - y0 - 12)
    rect_width = max(2, min(crop_width - rect_x - 1, object_width + 24))
    rect_height = max(2, min(crop_height - rect_y - 1, object_height + 24))
    rect = (rect_x, rect_y, rect_width, rect_height)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        crop,
        grabcut_mask,
        rect,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_RECT,
    )
    grabcut_result = (
        (grabcut_mask == cv2.GC_FGD)
        | (grabcut_mask == cv2.GC_PR_FGD)
    ).astype(np.uint8)

    # 补回时间差明显的像素，但只允许出现在目标框附近。
    allowed = np.zeros((crop_height, crop_width), dtype=np.uint8)
    extra_padding = 35
    allowed_x0 = max(0, x - x0 - extra_padding)
    allowed_y0 = max(0, y - y0 - extra_padding)
    allowed_x1 = min(
        crop_width,
        x + object_width - x0 + extra_padding,
    )
    allowed_y1 = min(
        crop_height,
        y + object_height - y0 + extra_padding,
    )
    allowed[allowed_y0:allowed_y1, allowed_x0:allowed_x1] = 1

    difference_seed = (local_difference > 12.0).astype(np.uint8)
    mask = grabcut_result | (difference_seed & allowed)
    mask &= allowed
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    # 只保留中心靠近自动目标框的连通区域。
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    kept = np.zeros_like(mask)
    target_center_x = x + object_width / 2.0 - x0
    target_center_y = y + object_height / 2.0 - y0
    max_distance = max(object_width, object_height) * 1.25 + 50

    for component_index in range(1, count):
        area = stats[component_index, cv2.CC_STAT_AREA]

        if area < 3:
            continue

        center_x, center_y = centers[component_index]

        if (
            np.hypot(
                center_x - target_center_x,
                center_y - target_center_y,
            )
            <= max_distance
        ):
            kept[labels == component_index] = 1

    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = kept * 255
    return full_mask


def detect_background_object(
    images: list[np.ndarray],
    base_index: int,
) -> np.ndarray:
    """检测固定在背景帧里的目标。

    背景帧通常直接当作画布，但它里面往往也有目标。这里单独检测一次，
    不参与 generate_motion_masks 的时序先验，因此不会改变其它帧的 掩膜。
    背景帧没有可检测目标时返回空 掩膜。
    """
    height, width = images[0].shape[:2]
    empty = np.zeros((height, width), dtype=np.uint8)

    if len(images) < 2:
        return empty

    small_images, background, sample_ids = _build_motion_background(images)
    background_lab = cv2.cvtColor(background, cv2.COLOR_BGR2LAB).astype(np.int16)
    detected = _detect_motion_bbox(
        small_images[base_index],
        background_lab,
        None,
        width,
        height,
    )

    if detected is None:
        return empty

    x, y, object_width, object_height, _center = detected

    return _refine_motion_mask(
        base_index,
        (x, y, object_width, object_height),
        sample_ids,
        images,
        width,
        height,
    )


def generate_motion_masks(
    images: list[np.ndarray],
    mask_dir: Path,
    output_dir: Path,
    force: bool,
    base_index: int,
) -> list[np.ndarray]:
    """通过背景差分和位置连续性生成运动目标 掩膜。"""
    if len(images) < 2:
        raise RuntimeError("自动提取运动目标至少需要两张抽帧。")

    mask_dir.mkdir(parents=True, exist_ok=True)
    height, width = images[0].shape[:2]
    print()
    print("正在自动获取运动目标...")

    small_images, background, sample_ids = _build_motion_background(images)
    write_image(output_dir / "motion_detection_background.png", background)
    background_lab = cv2.cvtColor(
        background,
        cv2.COLOR_BGR2LAB,
    ).astype(np.int16)

    masks: list[np.ndarray] = []
    previous_center: tuple[float, float] | None = None

    for frame_index, small_image in enumerate(small_images):
        frame_number = frame_index + 1

        if frame_index == base_index:
            masks.append(np.zeros((height, width), dtype=np.uint8))
            print(f"frame_{frame_number:04d}：固定背景帧")
            continue

        mask_path = mask_dir / f"frame_{frame_number:04d}.png"

        if mask_path.exists() and not force:
            mask = _load_saved_mask(mask_path, width, height)
            masks.append(mask)
            mask_y, mask_x = np.where(mask > 0)

            if len(mask_x) > 0:
                previous_center = (
                    float(mask_x.mean() * MOTION_SCALE),
                    float(mask_y.mean() * MOTION_SCALE),
                )

            print(f"frame_{frame_number:04d}：读取已有运动 掩膜")
            continue

        detected = _detect_motion_bbox(
            small_image,
            background_lab,
            previous_center,
            width,
            height,
        )

        if detected is None:
            mask = np.zeros((height, width), dtype=np.uint8)
            write_image(mask_path, mask)
            masks.append(mask)
            print(
                f"frame_{frame_number:04d}："
                "自动定位失败，本帧使用空 掩膜"
            )
            continue

        x, y, object_width, object_height, previous_center = detected
        mask = _refine_motion_mask(
            frame_index,
            (x, y, object_width, object_height),
            sample_ids,
            images,
            width,
            height,
        )
        write_image(mask_path, mask)
        masks.append(mask)
        print(f"frame_{frame_number:04d}：运动 掩膜 已保存")

    return masks


def _background_cache_paths(
    corrected_mask_dir: Path,
) -> tuple[Path, Path]:
    return (
        corrected_mask_dir / BACKGROUND_CACHE_MASK_NAME,
        corrected_mask_dir / BACKGROUND_CACHE_STATE_NAME,
    )


def _load_background_selection(
    corrected_mask_dir: Path,
    base_index: int,
    width: int,
    height: int,
) -> tuple[str, np.ndarray | None]:
    """读取背景帧缓存，返回 (来源, 掩膜)。

    没有缓存时来源为 auto。state 文件记录 base_frame，换了背景帧编号就
    不再复用旧结果；跳过状态必须由 state 明确记录，不能靠空 掩膜 推断。
    """
    mask_path, state_path = _background_cache_paths(corrected_mask_dir)

    if not state_path.exists():
        return BACKGROUND_SOURCE_AUTO, None

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"背景帧缓存无法读取：{state_path}") from exc

    if state.get("base_frame") != base_index + 1:
        return BACKGROUND_SOURCE_AUTO, None

    source = state.get("source")

    if source == BACKGROUND_SOURCE_SKIPPED:
        return BACKGROUND_SOURCE_SKIPPED, None

    if source != BACKGROUND_SOURCE_MANUAL:
        raise RuntimeError(f"背景帧缓存状态无效：{state_path}")

    if not mask_path.exists():
        print("背景帧人工 掩膜 缓存缺失，改用自动检测。")
        return BACKGROUND_SOURCE_AUTO, None

    return (
        BACKGROUND_SOURCE_MANUAL,
        _load_saved_mask(mask_path, width, height),
    )


def _save_background_selection(
    corrected_mask_dir: Path,
    base_index: int,
    source: str,
    mask: np.ndarray | None,
) -> None:
    """保存背景帧人工 掩膜 或跳过状态。"""
    corrected_mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path, state_path = _background_cache_paths(corrected_mask_dir)

    if source == BACKGROUND_SOURCE_MANUAL:
        if mask is None:
            raise ValueError("保存背景帧人工 掩膜 时缺少 掩膜。")

        write_image(mask_path, mask)
    elif mask_path.exists():
        mask_path.unlink()

    state_path.write_text(
        json.dumps(
            {
                "base_frame": base_index + 1,
                "source": source,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _clear_background_selection(corrected_mask_dir: Path) -> None:
    """删除背景帧缓存，让下次运行重新使用自动检测。"""
    mask_path, state_path = _background_cache_paths(corrected_mask_dir)

    for path in (mask_path, state_path):
        if path.exists():
            path.unlink()


def _tracker_label(name: str) -> str:
    return {
        "sam2": "SAM 2.1",
        "motion": "背景差分",
    }.get(name, name)


def _mask_source_label(source: str) -> str:
    return {
        "auto": "自动",
        "manual": "人工",
        "skipped": "已跳过",
        "empty": "空",
    }.get(source.lower(), source)


def _review_background_frame(
    images: list[np.ndarray],
    auto_mask: np.ndarray,
    base_index: int,
    corrected_mask_dir: Path,
    interval: float,
    force: bool,
    *,
    tracker_name: str = "motion",
    initial_source: str = BACKGROUND_SOURCE_AUTO,
    cache_background: bool = True,
    correct_mask: Callable[[int, np.ndarray], np.ndarray] | None = None,
    restore_mask: Callable[[int], np.ndarray] | None = None,
    manual_mask_path: Path | None = None,
) -> np.ndarray | None:
    """复核背景帧目标；返回确认的 掩膜，用户跳过时返回 None。"""
    height, width = images[0].shape[:2]
    frame_name = f"frame_{base_index + 1:04d}"
    display_scale = min(1500 / width, 850 / height, 1.0)
    window_name = f"复核背景帧 - {frame_name}"
    corrected_mask_dir.mkdir(parents=True, exist_ok=True)
    edit_mask_path = (
        manual_mask_path
        if manual_mask_path is not None
        else corrected_mask_dir / BACKGROUND_CACHE_MASK_NAME
    )

    if cache_background:
        state_path = corrected_mask_dir / BACKGROUND_CACHE_STATE_NAME

        if force and state_path.exists():
            print("--force：忽略已有背景帧人工缓存，重新使用自动检测。")
            # 先不删除旧缓存：用户可能只是看一眼自动结果就取消。确认接受
            # 自动 掩膜 时再清理，避免 --force 意外丢掉修正结果。
            mask_source, cached_mask = BACKGROUND_SOURCE_AUTO, None
        else:
            mask_source, cached_mask = _load_background_selection(
                corrected_mask_dir,
                base_index,
                width,
                height,
            )
    else:
        mask_source, cached_mask = initial_source, None

    if mask_source == BACKGROUND_SOURCE_MANUAL and cached_mask is not None:
        working_mask = cached_mask
        print(f"{frame_name}：读取已有背景帧人工 掩膜")
    elif mask_source == BACKGROUND_SOURCE_MANUAL:
        working_mask = auto_mask.copy()
        print(f"{frame_name}：当前背景帧使用人工修正 掩膜")
    elif mask_source == BACKGROUND_SOURCE_SKIPPED:
        working_mask = np.zeros((height, width), dtype=np.uint8)
        print(f"{frame_name}：读取已有背景帧跳过状态")
    else:
        working_mask = auto_mask.copy()
        mask_source = BACKGROUND_SOURCE_AUTO

    print()
    print("=" * 60)
    print(f"背景帧：{frame_name}")
    print("回车或空格：接受当前背景掩膜")
    print("M：手动重新框选并修正背景掩膜")
    print("R：恢复背景帧自动检测掩膜")
    print("S：跳过背景目标，不擦除也不纳入残影序列")
    print("Q 或 Esc：取消本次选择")
    print("=" * 60)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    title_size = max(20, int(round(height / 45)))
    hint_size = max(16, int(round(height / 60)))

    try:
        while True:
            preview = images[base_index].copy()

            if mask_source != BACKGROUND_SOURCE_SKIPPED:
                selected = working_mask > 0

                if np.any(selected):
                    preview[selected] = images[base_index][selected]
                    current_pixels = preview[selected].astype(np.float32)
                    green = np.zeros_like(current_pixels)
                    green[:, 1] = 255
                    preview[selected] = (
                        current_pixels * 0.68 + green * 0.32
                    ).clip(0, 255).astype(np.uint8)

            time_seconds = base_index * interval
            state_text = (
                "已跳过"
                if mask_source == BACKGROUND_SOURCE_SKIPPED
                else "未确认"
            )
            draw_text(
                preview,
                f"背景帧  {frame_name}  时间 {time_seconds:.1f} 秒  "
                f"跟踪器：{_tracker_label(tracker_name)}  "
                f"掩膜：{_mask_source_label(mask_source)}  "
                f"当前：{state_text}",
                (30, 20),
                size=title_size,
                color=(0, 255, 255),
            )
            draw_text(
                preview,
                "回车/空格：接受    M：手动修正    R：恢复自动",
                (30, 20 + int(title_size * 1.7)),
                size=hint_size,
                color=(0, 255, 255),
            )
            draw_text(
                preview,
                "S：跳过背景目标    Q/Esc：取消",
                (30, 20 + int(title_size * 1.7) + int(hint_size * 1.6)),
                size=hint_size,
                color=(0, 255, 255),
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
                if mask_source == BACKGROUND_SOURCE_SKIPPED:
                    if cache_background:
                        _save_background_selection(
                            corrected_mask_dir,
                            base_index,
                            BACKGROUND_SOURCE_SKIPPED,
                            None,
                        )
                    print(f"{frame_name}：跳过背景目标")
                    return None

                if cache_background:
                    if mask_source == BACKGROUND_SOURCE_MANUAL:
                        _save_background_selection(
                            corrected_mask_dir,
                            base_index,
                            BACKGROUND_SOURCE_MANUAL,
                            working_mask,
                        )
                    else:
                        _clear_background_selection(corrected_mask_dir)

                print(
                    f"{frame_name}：接受背景帧 掩膜"
                    f"（{mask_source.upper()}）"
                )
                return working_mask.copy()

            if key in (ord("s"), ord("S")):
                if cache_background:
                    _save_background_selection(
                        corrected_mask_dir,
                        base_index,
                        BACKGROUND_SOURCE_SKIPPED,
                        None,
                    )
                print(f"{frame_name}：跳过背景目标")
                return None

            if key in (ord("m"), ord("M")):
                edited_mask = _edit_manual_frame(
                    base_index,
                    images,
                    corrected_mask_dir,
                    True,
                    width,
                    height,
                    mask_path=edit_mask_path,
                )
                working_mask = (
                    correct_mask(base_index, edited_mask)
                    if correct_mask is not None
                    else edited_mask
                )
                mask_source = BACKGROUND_SOURCE_MANUAL
                if cache_background:
                    _save_background_selection(
                        corrected_mask_dir,
                        base_index,
                        BACKGROUND_SOURCE_MANUAL,
                        working_mask,
                    )
                print(f"{frame_name}：背景帧已使用人工修正 掩膜")
                continue

            if key in (ord("r"), ord("R")):
                working_mask = (
                    restore_mask(base_index)
                    if restore_mask is not None
                    else auto_mask.copy()
                )
                mask_source = BACKGROUND_SOURCE_AUTO
                if cache_background:
                    _clear_background_selection(corrected_mask_dir)
                print(f"{frame_name}：已恢复背景帧自动 掩膜")
                continue

            if key in (27, ord("q"), ord("Q")):
                raise RuntimeError("已取消本次选择。")
    finally:
        cv2.destroyWindow(window_name)


def select_motion_masks(
    images: list[np.ndarray],
    motion_masks: list[np.ndarray],
    corrected_mask_dir: Path,
    base_index: int,
    interval: float,
    *,
    background_auto_mask: np.ndarray | None = None,
    force: bool = False,
    tracking: TrackingHooks | None = None,
) -> SelectionResult:
    """逐帧预览自动 掩膜，由用户决定是否纳入最终叠加图。

    background_auto_mask 不为 None 时，先复核背景帧再进入普通候选帧；
    返回的 background_mask 为最终确认结果，None 表示用户主动跳过。
    """
    if len(images) != len(motion_masks):
        raise RuntimeError("预览帧数量与运动 掩膜 数量不一致。")

    tracking = tracking or TrackingHooks()
    height, width = images[0].shape[:2]

    if (
        background_auto_mask is not None
        and background_auto_mask.shape != (height, width)
    ):
        raise RuntimeError("背景帧 掩膜 尺寸与视频帧不一致。")

    selections: dict[str, bool] = {}
    display_scale = min(1500 / width, 850 / height, 1.0)
    window_name = "逐帧筛选目标时刻"
    candidate_indices = [
        index
        for index in range(len(images))
        if index != base_index
    ]
    corrected_mask_dir.mkdir(parents=True, exist_ok=True)
    manual_dir = tracking.manual_dir or corrected_mask_dir
    manual_dir.mkdir(parents=True, exist_ok=True)
    working_masks = [mask.copy() for mask in motion_masks]
    corrected_indices: set[int] = set(tracking.manual_indices)
    background_auto_mask = (
        background_auto_mask.copy()
        if background_auto_mask is not None
        else None
    )

    def frame_name_at(frame_index: int) -> str:
        return f"frame_{frame_index + 1:04d}"

    def corrected_mask_path(frame_index: int) -> Path:
        return corrected_mask_dir / f"{frame_name_at(frame_index)}.png"

    if tracking.name == "motion" and not force:
        for frame_index in candidate_indices:
            mask_path = corrected_mask_path(frame_index)

            if mask_path.exists():
                working_masks[frame_index] = _load_saved_mask(
                    mask_path,
                    width,
                    height,
                )
                corrected_indices.add(frame_index)

        if corrected_indices:
            print(f"复用已有人工修正 掩膜：{len(corrected_indices)} 张")

    def build_selected_masks() -> list[np.ndarray]:
        result: list[np.ndarray] = []

        for frame_index, working_mask in enumerate(working_masks):
            if frame_index == base_index:
                result.append(
                    np.zeros((height, width), dtype=np.uint8)
                )
                continue

            result.append(
                working_mask.copy()
                if selections.get(frame_name_at(frame_index), False)
                else np.zeros((height, width), dtype=np.uint8)
            )

        return result

    def build_cumulative(
        before_position: int,
    ) -> tuple[np.ndarray, int]:
        cumulative = images[base_index].copy()
        included_count = 0

        for position in range(before_position):
            frame_index = candidate_indices[position]
            frame_name = frame_name_at(frame_index)

            if not selections.get(frame_name, False):
                continue

            selected = working_masks[frame_index] > 0
            cumulative[selected] = images[frame_index][selected]
            included_count += 1

        return cumulative, included_count

    background_mask: np.ndarray | None = None

    def apply_tracker_masks(updated_masks: list[np.ndarray]) -> None:
        if len(updated_masks) != len(working_masks):
            raise RuntimeError("跟踪器返回的 掩膜 数量与抽帧数量不一致。")

        for index, mask in enumerate(updated_masks):
            working_masks[index] = mask.copy()

    def correct_frame(
        frame_index: int,
        mask: np.ndarray,
    ) -> np.ndarray:
        nonlocal background_auto_mask, background_mask

        working_masks[frame_index] = mask.copy()

        if tracking.correct is not None:
            apply_tracker_masks(tracking.correct(frame_index, mask))
            working_masks[frame_index] = mask.copy()

        corrected_indices.add(frame_index)

        if tracking.keeps_base_mask:
            background_auto_mask = working_masks[base_index].copy()

            if background_mask is not None:
                background_mask = working_masks[base_index].copy()

        return working_masks[frame_index].copy()

    def restore_frame(frame_index: int) -> np.ndarray:
        nonlocal background_auto_mask, background_mask

        if tracking.restore is not None:
            apply_tracker_masks(tracking.restore(frame_index))
        else:
            working_masks[frame_index] = motion_masks[frame_index].copy()
            mask_path = corrected_mask_path(frame_index)

            if mask_path.exists():
                mask_path.unlink()

        corrected_indices.discard(frame_index)

        if tracking.keeps_base_mask:
            background_auto_mask = working_masks[base_index].copy()

            if background_mask is not None:
                background_mask = working_masks[base_index].copy()

        return working_masks[frame_index].copy()

    review_background = background_auto_mask is not None
    cursor = 0
    cumulative, included_count = build_cumulative(cursor)

    print()
    print("开始选择需要放入叠加图的时刻：")
    print("Enter 或空格：纳入当前目标")
    print("S：跳过当前目标")
    print("B 或 Backspace：回退上一帧")
    print("M：手动重新框选并修正当前目标")
    print("R：恢复当前帧的自动 掩膜")
    print("Q 或 Esc：取消本次选择")
    print(
        "首个普通候选帧再按 B 会返回背景帧复核。"
    )

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    title_size = max(20, int(round(height / 45)))
    hint_size = max(16, int(round(height / 60)))

    try:
        while True:
            if review_background:
                background_mask = _review_background_frame(
                    images,
                    background_auto_mask,
                    base_index,
                    corrected_mask_dir,
                    interval,
                    force,
                    tracker_name=tracking.name,
                    initial_source=(
                        BACKGROUND_SOURCE_MANUAL
                        if tracking.background_manual
                        else BACKGROUND_SOURCE_AUTO
                    ),
                    cache_background=tracking.name == "motion",
                    correct_mask=(
                        correct_frame
                        if tracking.correct is not None
                        else None
                    ),
                    restore_mask=(
                        restore_frame
                        if tracking.restore is not None
                        else None
                    ),
                    manual_mask_path=(
                        tracking.background_mask_path
                        if tracking.background_mask_path is not None
                        else corrected_mask_dir / BACKGROUND_CACHE_MASK_NAME
                    ),
                )
                review_background = False
                cursor = 0
                cumulative, included_count = build_cumulative(cursor)

            back_to_background = False

            while cursor <= len(candidate_indices):
                if cursor == len(candidate_indices):
                    preview = cumulative.copy()
                    draw_text(
                        preview,
                        f"筛选完成  已纳入：{included_count}",
                        (30, 20),
                        size=title_size,
                        color=(0, 255, 255),
                    )
                    draw_text(
                        preview,
                        "回车/空格：完成    B/退格：回退    Q：取消",
                        (30, 20 + int(title_size * 1.7)),
                        size=hint_size,
                        color=(0, 255, 255),
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
                        break

                    if key in (8, 127, ord("b"), ord("B")):
                        cursor -= 1
                        cumulative, included_count = build_cumulative(cursor)
                        continue

                    if key in (27, ord("q"), ord("Q")):
                        raise RuntimeError("已取消本次选择。")

                    continue

                frame_index = candidate_indices[cursor]
                frame_name = frame_name_at(frame_index)
                working_mask = working_masks[frame_index]
                preview = cumulative.copy()
                selected = working_mask > 0

                # 当前候选使用原始像素并叠加绿色，已接受目标保留原始颜色。
                preview[selected] = images[frame_index][selected]
                current_pixels = preview[selected].astype(np.float32)
                green = np.zeros_like(current_pixels)
                green[:, 1] = 255
                preview[selected] = (
                    current_pixels * 0.68 + green * 0.32
                ).clip(0, 255).astype(np.uint8)

                time_seconds = frame_index * interval
                current_state = selections.get(frame_name)
                mask_source = (
                    "manual"
                    if frame_index in corrected_indices
                    else "empty"
                    if not np.any(selected)
                    else "auto"
                )
                state_text = (
                    "已纳入"
                    if current_state is True
                    else "已跳过"
                    if current_state is False
                    else "未选择"
                )
                draw_text(
                    preview,
                    f"{frame_name}  时间 {time_seconds:.1f} 秒  "
                    f"跟踪器：{_tracker_label(tracking.name)}  "
                    f"掩膜：{_mask_source_label(mask_source)}  "
                    f"当前：{state_text}",
                    (30, 20),
                    size=title_size,
                    color=(0, 255, 255),
                )
                draw_text(
                    preview,
                    f"已纳入：{included_count}    "
                    "回车/空格：纳入    S：跳过    B：回退    Q：退出",
                    (30, 20 + int(title_size * 1.7)),
                    size=hint_size,
                    color=(0, 255, 255),
                )
                draw_text(
                    preview,
                    "M：手动修正    R：恢复自动掩膜",
                    (30, 20 + int(title_size * 1.7) + int(hint_size * 1.6)),
                    size=hint_size,
                    color=(0, 255, 255),
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
                    selections[frame_name] = True
                    cumulative[selected] = images[frame_index][selected]
                    included_count += 1
                    cursor += 1
                    print(
                        f"{frame_name}：纳入叠加图，"
                        f"当前共 {included_count} 个"
                    )
                    continue

                if key in (ord("s"), ord("S")):
                    selections[frame_name] = False
                    cursor += 1
                    print(f"{frame_name}：跳过")
                    continue

                if key in (8, 127, ord("b"), ord("B")):
                    if cursor > 0:
                        cursor -= 1
                        cumulative, included_count = build_cumulative(cursor)
                        print(
                            f"回退到 {frame_name_at(candidate_indices[cursor])}"
                        )
                    elif background_auto_mask is not None:
                        print("返回背景帧复核")
                        back_to_background = True
                        break
                    continue

                if key in (ord("m"), ord("M")):
                    edited_mask = _edit_manual_frame(
                        frame_index,
                        images,
                        manual_dir,
                        True,
                        width,
                        height,
                    )
                    correct_frame(frame_index, edited_mask)
                    cumulative, included_count = build_cumulative(cursor)
                    print(f"{frame_name}：已使用人工修正 掩膜")
                    continue

                if key in (ord("r"), ord("R")):
                    had_manual = frame_index in corrected_indices
                    restore_frame(frame_index)
                    cumulative, included_count = build_cumulative(cursor)

                    if had_manual:
                        print(f"{frame_name}：已恢复自动 掩膜")
                    else:
                        print(
                            f"{frame_name}：没有人工修正，"
                            "保持自动 掩膜"
                        )

                    continue

                if key in (27, ord("q"), ord("Q")):
                    raise RuntimeError("已取消本次选择。")

            if back_to_background:
                review_background = True
                continue

            break
    finally:
        cv2.destroyWindow(window_name)

    included_count = sum(selections.values())
    print(f"选择完成：纳入 {included_count} 个目标位置")
    return SelectionResult(
        selected_masks=build_selected_masks(),
        background_mask=background_mask,
    )
