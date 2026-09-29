"""目标中心提取、轨迹拟合与渐变轨迹线绘制。"""

from collections.abc import Sequence

import cv2
import numpy as np

from .video import cast_pixels


TRAJECTORY_SAMPLES_PER_SEGMENT = 32

# Centripetal 参数化系数。uniform 参数化（0.0）在控制点间距不均匀时
# 会在段内产生尖点甚至自交小环，而逐帧人工筛选出的时刻间距恰好是不均匀的。
CENTRIPETAL_ALPHA = 0.5
KNOT_EPSILON = 1e-6

def parse_hex_color(text: str) -> tuple[int, int, int]:
    """把 #RRGGBB 或 #RGB 解析成 OpenCV 使用的 BGR 三元组。"""
    value = text.strip().lstrip("#")

    if len(value) == 3:
        value = "".join(character * 2 for character in value)

    if len(value) != 6:
        raise ValueError(f"颜色需要是 #RRGGBB 或 #RGB 形式：{text}")

    try:
        red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError as exc:
        raise ValueError(f"颜色不是合法的十六进制值：{text}") from exc

    return (blue, green, red)


def format_hex_color(color: tuple[int, int, int]) -> str:
    """把 BGR 三元组还原成 #RRGGBB，用于回显给用户。"""
    blue, green, red = color
    return f"#{red:02X}{green:02X}{blue:02X}"


# 轨迹线颜色（BGR）。选色要在浅色天空和深色树/建筑上都跳得出来：白色在
# 浅色天空上会融成一片，明度偏低的颜色（如 #65548F）也会糊，琥珀色两头都稳。
# 由 --trajectory-color 覆盖。
TRAJECTORY_COLOR = (0, 165, 255)  # #FFA500

# 彩色线下方那条更宽的深色描边，保证浅色线在浅色背景上也可读。
# 宽度按线宽的倍数表示，可由 --no-trajectory-outline 关闭。
TRAJECTORY_OUTLINE_COLOR = (20, 20, 20)
TRAJECTORY_OUTLINE_RATIO = 2.0

# 最早处的线宽占最大线宽的比例。和不透明度一起构成彗尾的"由细到粗"。
# 这个比例要拉得足够开才读得出来：接近 1 时线宽几乎不变，看起来只是一条
# 被调了透明度的等宽线。
TRAJECTORY_TAPER = 0.25


def mask_center(mask: np.ndarray) -> tuple[float, float] | None:
    """返回掩膜非零像素的外接矩形中心，掩膜为空时返回 None。

    使用外接矩形中心而不是质心：目标掩膜中螺旋桨与机身常常分成多块，
    像素密度不均，质心会被面积更大的一侧拉偏。
    """
    rows, columns = np.nonzero(mask)

    if columns.size == 0:
        return None

    return (
        (float(columns.min()) + float(columns.max())) / 2.0,
        (float(rows.min()) + float(rows.max())) / 2.0,
    )


def _knot_step(start: np.ndarray, end: np.ndarray) -> float:
    """相邻控制点之间的 centripetal 节点间距。"""
    return max(float(np.linalg.norm(end - start)), KNOT_EPSILON) ** CENTRIPETAL_ALPHA


def _sample_segment(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    fourth: np.ndarray,
    fractions: np.ndarray,
) -> np.ndarray:
    """在 second → third 段上按 Barry-Goldman 递推采样。

    fractions 取 [0, 1)，因此段末控制点由调用方单独补上。
    """
    t0 = 0.0
    t1 = t0 + _knot_step(first, second)
    t2 = t1 + _knot_step(second, third)
    t3 = t2 + _knot_step(third, fourth)

    t = (t1 + np.asarray(fractions, dtype=np.float64) * (t2 - t1)).reshape(-1, 1)

    row_a1 = ((t1 - t) / (t1 - t0)) * first + ((t - t0) / (t1 - t0)) * second
    row_a2 = ((t2 - t) / (t2 - t1)) * second + ((t - t1) / (t2 - t1)) * third
    row_a3 = ((t3 - t) / (t3 - t2)) * third + ((t - t2) / (t3 - t2)) * fourth

    row_b1 = ((t2 - t) / (t2 - t0)) * row_a1 + ((t - t0) / (t2 - t0)) * row_a2
    row_b2 = ((t3 - t) / (t3 - t1)) * row_a2 + ((t - t1) / (t3 - t1)) * row_a3

    return ((t2 - t) / (t2 - t1)) * row_b1 + ((t - t1) / (t2 - t1)) * row_b2


def smooth_path(
    points: Sequence[tuple[float, float]],
    taus: Sequence[float] | None = None,
    samples_per_segment: int = TRAJECTORY_SAMPLES_PER_SEGMENT,
) -> tuple[np.ndarray, np.ndarray]:
    """用 centripetal Catmull-Rom 拟合穿过全部控制点的光滑轨迹。

    曲线严格穿过每个控制点，只把折角磨圆；返回稠密采样点与每个采样点
    对应的时间参数。taus 缺省时按等间隔处理。
    """
    control = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    count = len(control)

    if taus is None:
        control_taus = (
            np.linspace(0.0, 1.0, count) if count > 0 else np.zeros(0)
        )
    else:
        control_taus = np.asarray(taus, dtype=np.float64).reshape(-1)

    if control_taus.size != count:
        raise ValueError("轨迹时间参数与目标位置数量不一致。")

    if count == 0:
        return control.reshape(0, 2), control_taus
    if count == 1:
        return control.copy(), control_taus.copy()

    fractions = np.linspace(
        0.0,
        1.0,
        max(2, samples_per_segment),
        endpoint=False,
    )

    if count == 2:
        weights = fractions.reshape(-1, 1)
        samples = control[0] + (control[1] - control[0]) * weights
        sample_taus = control_taus[0] + (
            (control_taus[1] - control_taus[0]) * fractions
        )
        return (
            np.vstack([samples, control[-1]]),
            np.append(sample_taus, control_taus[-1]),
        )

    # 首尾各补一个镜像虚拟控制点，避免重复点导致节点间距为零。
    extended = np.vstack(
        [
            2.0 * control[0] - control[1],
            control,
            2.0 * control[-1] - control[-2],
        ]
    )

    sample_blocks: list[np.ndarray] = []
    tau_blocks: list[np.ndarray] = []

    for index in range(1, count):
        sample_blocks.append(
            _sample_segment(
                extended[index - 1],
                extended[index],
                extended[index + 1],
                extended[index + 2],
                fractions,
            )
        )
        tau_blocks.append(
            control_taus[index - 1]
            + (control_taus[index] - control_taus[index - 1]) * fractions
        )

    samples = np.vstack([*sample_blocks, control[-1]])
    sample_taus = np.concatenate([*tau_blocks, control_taus[-1:]])
    return samples, sample_taus


def _blend_region(
    region: np.ndarray,
    stencil: np.ndarray,
    color: tuple[int, int, int],
    alpha_map: np.ndarray,
) -> None:
    """把单色按 stencil 覆盖率混合进画布区域（就地修改）。"""
    coverage = stencil.astype(np.float32) / 255.0 * alpha_map
    selected = coverage > 0.0

    if not np.any(selected):
        return

    # 灰度画布只有单个通道，此时颜色退化为第一个分量。
    target = np.asarray(color, dtype=np.float32)
    if region.ndim == 2:
        target = target[:1]

    # 布尔索引会去掉一维，所以这里的补维数要再少一层。
    extra_dims = (1,) * (region.ndim - 2)
    original = region[selected].astype(np.float32)
    region[selected] = cast_pixels(
        original
        + (target - original) * coverage[selected].reshape(-1, *extra_dims),
        region.dtype,
    )


def _ribbon_edges(
    samples: np.ndarray,
    half_widths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """沿路径法线方向向两侧外扩，得到缎带的左右两条边界。"""
    deltas = np.gradient(samples, axis=0)
    lengths = np.linalg.norm(deltas, axis=1, keepdims=True)
    lengths[lengths < KNOT_EPSILON] = KNOT_EPSILON
    normals = np.column_stack([-deltas[:, 1], deltas[:, 0]]) / lengths

    # 相邻采样点重合时法线是零向量，会画出零宽度的四边形形成断口，
    # 沿用上一个有效法线即可。
    for index in range(1, len(normals)):
        if not normals[index].any() and normals[index - 1].any():
            normals[index] = normals[index - 1]

    offset = normals * np.asarray(half_widths, dtype=np.float64)[:, None]
    return samples + offset, samples - offset


def _quad(left: np.ndarray, right: np.ndarray, index: int) -> np.ndarray:
    """第 index 段的四边形：左右边界各取两点。"""
    return np.array(
        [
            left[index],
            left[index + 1],
            right[index + 1],
            right[index],
        ]
    )


def _stamp_quad(
    stencil: np.ndarray,
    tau_map: np.ndarray,
    quad: np.ndarray,
    tau: float,
) -> tuple[int, int, int, int]:
    """把四边形填进 stencil，并记录它覆盖像素的时间参数。

    返回覆盖的包围盒，调用方用它累计最终要混合的区域。
    """
    height, width = stencil.shape
    x0 = max(0, int(np.floor(quad[:, 0].min())) - 2)
    y0 = max(0, int(np.floor(quad[:, 1].min())) - 2)
    x1 = min(width, int(np.ceil(quad[:, 0].max())) + 3)
    y1 = min(height, int(np.ceil(quad[:, 1].max())) + 3)

    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0)

    local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(
        local,
        [(quad - np.array([x0, y0])).astype(np.int32)],
        255,
        cv2.LINE_AA,
    )

    view = stencil[y0:y1, x0:x1]
    np.maximum(view, local, out=view)

    # 用 local 而不是累加后的 stencil：后者含其它段覆盖的像素，会把它们的
    # 时间参数改成本段的值。
    tau_map[y0:y1, x0:x1][local > 0] = tau

    return (x0, y0, x1, y1)


def draw_trajectory(
    canvas: np.ndarray,
    samples: np.ndarray,
    taus: np.ndarray,
    min_alpha: float,
    color: tuple[int, int, int] = TRAJECTORY_COLOR,
    outline: bool = True,
) -> None:
    """按时间渐变就地绘制轨迹线。

    亮度与宽度同时随时间渐变：最早的一段最淡最细，最新的一段最实最粗，
    两样合起来才是彗尾的形态。线宽随画面宽度缩放。

    用四边形拼出宽度连续变化的缎带，而不是逐段调用 cv2.line——后者的
    thickness 只能是整数，宽度渐变会在曲线上留下阶梯状缺口。

    两种 stencil 都要各自累积完成后再混合一次，且描边先于彩线：逐段交替
    绘制的话，后一段的描边会盖住前一段的彩线，交界处反复叠加会形成串珠状
    暗斑。

    outline 为真时在彩线下方铺一条更宽的深色描边，代价是浅色背景上会多出
    一道深边，好处是浅色线条在浅色天空上也读得出来。
    """
    if len(samples) < 2:
        return

    height, width = canvas.shape[:2]
    max_line_width = max(3.0, width / 750.0)

    times = np.clip(np.asarray(taus, dtype=np.float64), 0.0, 1.0)
    ramp = TRAJECTORY_TAPER + (1.0 - TRAJECTORY_TAPER) * times

    line_left, line_right = _ribbon_edges(samples, max_line_width * ramp / 2.0)
    line_stencil = np.zeros((height, width), dtype=np.uint8)

    if outline:
        outline_left, outline_right = _ribbon_edges(
            samples,
            max_line_width * TRAJECTORY_OUTLINE_RATIO * ramp / 2.0,
        )
        outline_stencil: np.ndarray | None = np.zeros(
            (height, width),
            dtype=np.uint8,
        )
    else:
        outline_left = outline_right = outline_stencil = None

    tau_map = np.zeros((height, width), dtype=np.float32)

    lower_x, lower_y = width, height
    upper_x = upper_y = 0

    for index in range(len(samples) - 1):
        tau = float((times[index] + times[index + 1]) / 2.0)
        boxes = []

        if outline_stencil is not None:
            boxes.append(
                _stamp_quad(
                    outline_stencil,
                    tau_map,
                    _quad(outline_left, outline_right, index),
                    tau,
                )
            )

        boxes.append(
            _stamp_quad(
                line_stencil,
                tau_map,
                _quad(line_left, line_right, index),
                tau,
            )
        )

        for x0, y0, x1, y1 in boxes:
            if x1 > x0 and y1 > y0:
                lower_x, lower_y = min(lower_x, x0), min(lower_y, y0)
                upper_x, upper_y = max(upper_x, x1), max(upper_y, y1)

    if upper_x <= lower_x or upper_y <= lower_y:
        return

    region = canvas[lower_y:upper_y, lower_x:upper_x]
    tau_region = tau_map[lower_y:upper_y, lower_x:upper_x]
    alpha_map = min_alpha + (1.0 - min_alpha) * tau_region

    if outline_stencil is not None:
        _blend_region(
            region,
            outline_stencil[lower_y:upper_y, lower_x:upper_x],
            TRAJECTORY_OUTLINE_COLOR,
            alpha_map,
        )

    _blend_region(
        region,
        line_stencil[lower_y:upper_y, lower_x:upper_x],
        color,
        alpha_map,
    )
