from pathlib import Path

import cv2
import numpy as np

from .video import read_image, write_image


MANUAL_ROI_PADDING = 50
INITIAL_BRUSH_RADIUS = 10

MOTION_SCALE = 0.25
MOTION_ROI_PADDING = 80
MOTION_MIN_AREA_RATIO = 0.00002
MOTION_MAX_AREA_RATIO = 0.08
MOTION_MAX_WIDTH_RATIO = 0.40
MOTION_MAX_HEIGHT_RATIO = 0.40


def _load_saved_mask(
    mask_path: Path,
    width: int,
    height: int,
) -> np.ndarray:
    mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise RuntimeError(f"读取 Mask 失败：{mask_path}")

    if mask.shape != (height, width):
        raise RuntimeError(f"Mask 尺寸与视频帧不一致：{mask_path}")

    return mask


def _edit_manual_frame(
    frame_index: int,
    images: list[np.ndarray],
    mask_dir: Path,
    force: bool,
    width: int,
    height: int,
) -> np.ndarray:
    frame_number = frame_index + 1
    image = images[frame_index]
    mask_path = mask_dir / f"frame_{frame_number:04d}.png"

    if mask_path.exists() and not force:
        print(f"frame_{frame_number:04d}：读取已有手动 Mask")
        return _load_saved_mask(mask_path, width, height)

    # 第一步：在完整画面中框选目标对象。
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
    select_window = f"1 Select Target - frame_{frame_number:04d}"

    print()
    print("=" * 60)
    print(f"frame_{frame_number:04d}")
    print("第一步：鼠标框住完整目标对象")
    print("框好以后按 Enter")
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

    x = int(sx / display_scale)
    y = int(sy / display_scale)
    object_width = int(sw / display_scale)
    object_height = int(sh / display_scale)

    # 第二步：在局部区域内使用 GrabCut 生成初始 Mask。
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

    # 第三步：由用户补充或擦除 Mask。
    work_mask = initial_mask.copy()
    state = {
        "drawing": False,
        "erase": False,
        "radius": INITIAL_BRUSH_RADIUS,
    }

    zoom = min(1200 / crop_width, 800 / crop_height)
    zoom = max(1.0, min(5.0, zoom))
    edit_window = f"2 Edit Mask - frame_{frame_number:04d}"
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
    print("C：清空 Mask")
    print("R：恢复 GrabCut 初始 Mask")
    print("A：将整个框选矩形作为前景")
    print("Enter：保存并进入下一帧")

    while True:
        display = crop.copy()
        overlay = display.copy()
        overlay[work_mask > 0] = (0, 255, 0)
        display = cv2.addWeighted(display, 0.72, overlay, 0.28, 0)
        cv2.putText(
            display,
            f"frame_{frame_number:04d}  brush={state['radius']}px",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
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
    print(f"保存手动 Mask：{mask_path.name}")
    return full_mask


def generate_manual_masks(
    images: list[np.ndarray],
    mask_dir: Path,
    force: bool,
    base_index: int,
) -> list[np.ndarray]:
    """逐帧生成由用户确认的目标 Mask。"""
    mask_dir.mkdir(parents=True, exist_ok=True)
    height, width = images[0].shape[:2]
    masks: list[np.ndarray] = []

    for frame_index in range(len(images)):
        frame_number = frame_index + 1

        if frame_index == base_index:
            masks.append(np.zeros((height, width), dtype=np.uint8))
            print(f"frame_{frame_number:04d}：固定背景帧")
            continue

        masks.append(
            _edit_manual_frame(
                frame_index,
                images,
                mask_dir,
                force,
                width,
                height,
            )
        )

    return masks


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
        raise RuntimeError("运动目标模式至少需要两张抽帧。")

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


def generate_motion_masks(
    images: list[np.ndarray],
    mask_dir: Path,
    output_dir: Path,
    force: bool,
    base_index: int,
) -> list[np.ndarray]:
    """通过背景差分和位置连续性生成运动目标 Mask。"""
    if len(images) < 2:
        raise RuntimeError("运动目标模式至少需要两张抽帧。")

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

            print(f"frame_{frame_number:04d}：读取已有运动 Mask")
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
                "自动定位失败，本帧使用空 Mask"
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
        print(f"frame_{frame_number:04d}：运动 Mask 已保存")

    return masks


def select_motion_masks(
    images: list[np.ndarray],
    motion_masks: list[np.ndarray],
    corrected_mask_dir: Path,
    base_index: int,
    interval: float,
) -> list[np.ndarray]:
    """逐帧预览自动 Mask，由用户决定是否纳入最终叠加图。"""
    if len(images) != len(motion_masks):
        raise RuntimeError("预览帧数量与运动 Mask 数量不一致。")

    selections: dict[str, bool] = {}
    height, width = images[0].shape[:2]
    display_scale = min(1500 / width, 850 / height, 1.0)
    window_name = "Select Target Moments"
    candidate_indices = [
        index
        for index in range(len(images))
        if index != base_index
    ]
    corrected_mask_dir.mkdir(parents=True, exist_ok=True)
    working_masks = [mask.copy() for mask in motion_masks]
    corrected_indices: set[int] = set()

    def frame_name_at(frame_index: int) -> str:
        return f"frame_{frame_index + 1:04d}"

    def corrected_mask_path(frame_index: int) -> Path:
        return corrected_mask_dir / f"{frame_name_at(frame_index)}.png"

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
        print(f"复用已有人工修正 Mask：{len(corrected_indices)} 张")

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

    cursor = 0
    cumulative, included_count = build_cumulative(cursor)

    print()
    print("开始选择需要放入叠加图的时刻：")
    print("Enter 或空格：纳入当前目标")
    print("S：跳过当前目标")
    print("B 或 Backspace：回退上一帧")
    print("M：手动重新框选并修正当前目标")
    print("R：恢复当前帧的自动 Mask")
    print("Q 或 Esc：取消本次选择")

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while cursor <= len(candidate_indices):
            if cursor == len(candidate_indices):
                preview = cumulative.copy()
                cv2.putText(
                    preview,
                    f"Selection complete  Included: {included_count}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (0, 255, 255),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    preview,
                    "Enter/Space=finish   B/Backspace=back   Q=cancel",
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
                "MANUAL"
                if frame_index in corrected_indices
                else "AUTO"
            )
            state_text = (
                "INCLUDED"
                if current_state is True
                else "SKIPPED"
                if current_state is False
                else "UNDECIDED"
            )
            cv2.putText(
                preview,
                f"{frame_name}  t~{time_seconds:.1f}s  "
                f"Mask: {mask_source}  Current: {state_text}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                f"Included: {included_count}   "
                "Enter/Space=include   S=skip   B=back   Q=quit",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                "M=manual correction   R=restore auto mask",
                (30, 150),
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
                continue

            if key in (ord("m"), ord("M")):
                working_masks[frame_index] = _edit_manual_frame(
                    frame_index,
                    images,
                    corrected_mask_dir,
                    True,
                    width,
                    height,
                )
                corrected_indices.add(frame_index)
                print(f"{frame_name}：已使用人工修正 Mask")
                continue

            if key in (ord("r"), ord("R")):
                working_masks[frame_index] = motion_masks[
                    frame_index
                ].copy()
                corrected_indices.discard(frame_index)
                mask_path = corrected_mask_path(frame_index)

                if mask_path.exists():
                    mask_path.unlink()

                print(f"{frame_name}：已恢复自动 Mask")
                continue

            if key in (27, ord("q"), ord("Q")):
                raise RuntimeError("已取消本次选择。")
    finally:
        cv2.destroyWindow(window_name)

    included_count = sum(selections.values())
    print(f"选择完成：纳入 {included_count} 个目标位置")
    return build_selected_masks()
