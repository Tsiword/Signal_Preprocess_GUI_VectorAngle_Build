# -*- coding: utf-8 -*-
"""
Signal / Accelerometer Preprocess GUI
-------------------------------------
独立数据预处理软件：
1. 导入单列位移/角度数据，进行异常点插值、中值滤波、低通滤波、起点归零、终点回零，并绘图比较处理前后的波形；
2. 导入三轴加速度计数据，选择 X/Y/Z/Sum 等某一列绘图，拖拽或手动截取一段，转换为基于重力加速度的角度位移数据，再导出给摆台控制软件使用；
3. 支持三轴矢量法：当没有任何一个传感器轴正好沿重力方向时，可绕手动输入或自动估计的旋转轴计算角度。

依赖：numpy, matplotlib
打包：PyInstaller --onefile --windowed
"""

from __future__ import annotations

import json
import math
import re
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

APP_TITLE = "采集信号/三轴加速度转角度预处理工具"
APP_VERSION = "1.2.1"

DEFAULT_DEMO = """1.0569257
1.0306822
1.0518241
1.0701149
1.0341399
1.0233878
1.0444518
1.0624153
1.0071562
1.0226961
1.0252296
1.0336929
1.0036045
1.0767168
1.0103481
0.99388437
1.0450443
1.0366730
1.0997759
1.1624672
1.1370396
1.1418913
1.1685129
1.2304965
1.2243574
1.2112345
1.2132502
1.2385659
1.2353591
1.2489225
1.2594997
1.2943118
1.2753441
1.2731093
1.3431321
1.3448296
1.3331020
1.3312174
1.3138672
1.3131927
1.3631598
1.3593076
1.3395823
1.3675697
1.4105218
1.4048274
1.3903677
1.3956442
1.4166273
1.4583469
1.4785520
1.5075684
1.5041950
1.4978849
1.4797676
1.5406589
1.5706481
1.5508355
1.5444620
1.5719799
1.5814741
1.5597439
1.5611894
1.5863796
1.5951803
1.5855434
1.5857638
1.5631681
1.5589766
1.4015221
1.2175910
1.1637459
1.2277338
1.3446915
1.3585312
1.2808010
1.1261914
1.0468717
1.0761044
1.1261280
1.1341168
1.1292628
1.0277046
1.0255062
1.0679623
1.0643829
1.0588909
1.0288403
1.0022929
1.0304896
1.0397526
1.0466433
0.99539161
0.96956504
1.0234354
1.0787483
1.0299019
1.0141744
1.0539398
1.0615716
"""


@dataclass
class PreprocessConfig:
    interval_s: float = 0.040
    column_index: str = ""  # 从 1 开始；空表示自动解析单列或表格第一可用列
    scale_factor: float = 1.0

    invalid_enabled: bool = True
    min_value: float = 0.0
    max_value: float = 2.0

    median_enabled: bool = True
    median_window: int = 5

    lowpass_enabled: bool = True
    lowpass_cutoff_hz: float = 0.7

    zero_start: bool = True
    zero_end: bool = True

    plot_max_points: int = 12000
    output_precision: int = 8


@dataclass
class PreprocessReport:
    input_count: int
    output_count: int
    invalid_count: int
    raw_start: float
    raw_end: float
    raw_min: float
    raw_max: float
    processed_start: float
    processed_end: float
    processed_min: float
    processed_max: float
    raw_max_abs_velocity: float
    processed_max_abs_velocity: float
    raw_max_abs_acceleration: float
    processed_max_abs_acceleration: float


@dataclass
class AngleConversionConfig:
    axis_name: str = "Z"
    reference_axis_name: str = "X"
    method: str = "单轴 asin(a/g)"
    g_value: float = 1.0
    start_index_1based: int = 1
    end_index_1based: int = 0  # 0 表示到末尾
    baseline_points: int = 10
    angle_scale_factor: float = 1.0
    angle_zero_end: bool = False
    angle_reverse: bool = False
    rotation_axis_source: str = "自动估计"  # 自动估计 / 手动输入
    rotation_axis_x: float = 0.0
    rotation_axis_y: float = 1.0
    rotation_axis_z: float = 0.0


def parse_float_token(token: str) -> Optional[float]:
    token = token.strip()
    if not token:
        return None
    try:
        v = float(token)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def split_line_to_tokens(line: str) -> list[str]:
    # 兼容逗号、分号、空白、Tab。用于单列/普通数字解析。
    return [p for p in re.split(r"[,;\s]+", line.strip()) if p]


def split_table_line(line: str) -> list[str]:
    """用于表格解析：逗号/分号/Tab 优先保留表头中的空格，例如 Packet num。"""
    s = line.strip()
    if not s:
        return []
    if "," in s:
        return [p.strip() for p in s.split(",")]
    if ";" in s:
        return [p.strip() for p in s.split(";")]
    if "\t" in s:
        return [p.strip() for p in s.split("\t")]
    return [p for p in re.split(r"\s+", s) if p]


def read_text_file(path: str) -> str:
    encodings = ["utf-8-sig", "utf-8", "gbk", "cp936"]
    last_exc: Optional[Exception] = None
    for enc in encodings:
        try:
            return Path(path).read_text(encoding=enc)
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
    raise ValueError(f"读取文件失败，可能不是文本文件: {last_exc}")


def parse_sequence_text(text: str, column_index: str = "") -> np.ndarray:
    """从文本框内容解析数字。column_index 为空时会读取所有数值；否则读取指定列。"""
    lines = []
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        if line.strip():
            lines.append(line)

    values: list[float] = []
    col: Optional[int] = None
    if column_index.strip():
        try:
            c = int(float(column_index.strip()))
            if c < 1:
                raise ValueError
            col = c - 1
        except Exception as exc:
            raise ValueError("列号必须为空，或为从 1 开始的正整数。") from exc

    for line_no, line in enumerate(lines, start=1):
        tokens = split_line_to_tokens(line.replace("[", " ").replace("]", " ").replace("(", " ").replace(")", " "))
        if not tokens:
            continue
        if col is None:
            for token in tokens:
                v = parse_float_token(token)
                if v is not None:
                    values.append(v)
        else:
            if col >= len(tokens):
                continue
            v = parse_float_token(tokens[col])
            if v is not None:
                values.append(v)

    if not values:
        raise ValueError("未解析到任何有效数字。请检查文件格式或列号设置。")
    return np.asarray(values, dtype=float)


def parse_numeric_table(text: str) -> Tuple[list[str], np.ndarray]:
    """解析带表头或无表头的数值表格。返回 headers, data。"""
    lines: list[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    if not lines:
        raise ValueError("文件为空。")

    first_tokens = split_table_line(lines[0])
    first_numeric = [parse_float_token(t) for t in first_tokens]
    has_header = any(v is None for v in first_numeric)

    if has_header:
        headers = [h.strip() or f"col{i+1}" for i, h in enumerate(first_tokens)]
        data_lines = lines[1:]
    else:
        headers = [f"col{i+1}" for i in range(len(first_tokens))]
        data_lines = lines

    rows: list[list[float]] = []
    width = len(headers)
    for line in data_lines:
        tokens = split_table_line(line)
        if not tokens:
            continue
        numeric: list[float] = []
        for token in tokens[:width]:
            v = parse_float_token(token)
            if v is None:
                numeric = []
                break
            numeric.append(v)
        if len(numeric) == width:
            rows.append(numeric)

    if not rows:
        raise ValueError("未解析到有效表格数据。")
    return headers, np.asarray(rows, dtype=float)


def read_sequence_file(path: str, column_index: str = "") -> np.ndarray:
    text = read_text_file(path)
    return parse_sequence_text(text, column_index=column_index)


def read_table_file(path: str) -> Tuple[list[str], np.ndarray]:
    text = read_text_file(path)
    return parse_numeric_table(text)


def interpolate_invalid_points(values: np.ndarray, min_value: float, max_value: float) -> Tuple[np.ndarray, int]:
    x = np.asarray(values, dtype=float).copy()
    if x.size == 0:
        raise ValueError("序列为空。")
    mask_valid = np.isfinite(x) & (x >= min_value) & (x <= max_value)
    invalid_count = int(x.size - np.count_nonzero(mask_valid))
    if invalid_count == 0:
        return x, 0
    if not np.any(mask_valid):
        raise ValueError("所有点都被判定为异常点，无法插值。请放宽有效范围。")
    idx = np.arange(x.size)
    x[~mask_valid] = np.interp(idx[~mask_valid], idx[mask_valid], x[mask_valid])
    return x, invalid_count


def median_filter_centered(values: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if window <= 1 or x.size <= 2:
        return x.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = np.empty_like(x)
    for i in range(x.size):
        lo = max(0, i - half)
        hi = min(x.size, i + half + 1)
        out[i] = float(np.median(x[lo:hi]))
    return out


def lowpass_first_order_zero_phase(values: np.ndarray, interval_s: float, cutoff_hz: float) -> np.ndarray:
    """一阶低通 + 反向再滤一次，减少相位滞后；避免 scipy 依赖。"""
    x = np.asarray(values, dtype=float)
    if x.size <= 2:
        return x.copy()
    if interval_s <= 0:
        raise ValueError("低通滤波需要正的时间间隔。")
    if cutoff_hz <= 0:
        raise ValueError("低通截止频率必须大于 0。")
    fs = 1.0 / interval_s
    if cutoff_hz >= fs / 2.0:
        return x.copy()

    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    alpha = interval_s / (rc + interval_s)

    def forward_filter(seq: np.ndarray) -> np.ndarray:
        y = np.empty_like(seq)
        last = float(seq[0])
        y[0] = last
        for i in range(1, seq.size):
            last = last + alpha * (float(seq[i]) - last)
            y[i] = last
        return y

    y = forward_filter(x)
    y2 = forward_filter(y[::-1])[::-1]
    return y2


def velocity_and_acceleration(values: np.ndarray, interval_s: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    v = np.diff(x) / interval_s
    if v.size < 2:
        a = np.asarray([], dtype=float)
    else:
        a = np.diff(v) / interval_s
    return v, a


def preprocess_data(raw: np.ndarray, cfg: PreprocessConfig) -> Tuple[np.ndarray, PreprocessReport]:
    if raw.size == 0:
        raise ValueError("原始数据为空。")
    if cfg.interval_s <= 0:
        raise ValueError("时间间隔必须大于 0。")
    if cfg.scale_factor < 0 or not math.isfinite(cfg.scale_factor):
        raise ValueError("缩放系数必须是大于等于 0 的有限数值。")
    if cfg.invalid_enabled and cfg.min_value >= cfg.max_value:
        raise ValueError("异常点有效范围中，最小值必须小于最大值。")
    if cfg.median_window < 1:
        raise ValueError("中值滤波窗口必须 >= 1。")
    if cfg.plot_max_points < 100:
        raise ValueError("绘图最大点数建议至少 100。")

    processed = np.asarray(raw, dtype=float).copy()
    invalid_count = 0

    if cfg.invalid_enabled:
        processed, invalid_count = interpolate_invalid_points(processed, cfg.min_value, cfg.max_value)

    if cfg.median_enabled and cfg.median_window > 1:
        processed = median_filter_centered(processed, int(cfg.median_window))

    if cfg.lowpass_enabled:
        processed = lowpass_first_order_zero_phase(processed, cfg.interval_s, cfg.lowpass_cutoff_hz)

    if cfg.zero_start:
        processed = processed - processed[0]

    if cfg.zero_end and processed.size >= 2:
        drift = np.linspace(0.0, float(processed[-1]), processed.size)
        processed = processed - drift

    if cfg.scale_factor != 1.0:
        processed = processed * cfg.scale_factor

    raw_v, raw_a = velocity_and_acceleration(raw, cfg.interval_s)
    proc_v, proc_a = velocity_and_acceleration(processed, cfg.interval_s)

    report = PreprocessReport(
        input_count=int(raw.size),
        output_count=int(processed.size),
        invalid_count=invalid_count,
        raw_start=float(raw[0]),
        raw_end=float(raw[-1]),
        raw_min=float(np.nanmin(raw)),
        raw_max=float(np.nanmax(raw)),
        processed_start=float(processed[0]),
        processed_end=float(processed[-1]),
        processed_min=float(np.nanmin(processed)),
        processed_max=float(np.nanmax(processed)),
        raw_max_abs_velocity=float(np.nanmax(np.abs(raw_v))) if raw_v.size else 0.0,
        processed_max_abs_velocity=float(np.nanmax(np.abs(proc_v))) if proc_v.size else 0.0,
        raw_max_abs_acceleration=float(np.nanmax(np.abs(raw_a))) if raw_a.size else 0.0,
        processed_max_abs_acceleration=float(np.nanmax(np.abs(proc_a))) if proc_a.size else 0.0,
    )
    return processed, report


def downsample_for_plot(x: np.ndarray, y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    step = int(math.ceil(x.size / max_points))
    return x[::step], y[::step]


def report_to_text(report: PreprocessReport, cfg: PreprocessConfig) -> str:
    lines = []
    lines.append("========== 预处理结果 ==========")
    lines.append(f"点数: {report.input_count} -> {report.output_count}")
    lines.append(f"时间间隔: {cfg.interval_s:g} s，采样率: {1.0 / cfg.interval_s:g} Hz")
    lines.append(f"异常点插值: {'开启' if cfg.invalid_enabled else '关闭'}" + (f"，有效范围 [{cfg.min_value:g}, {cfg.max_value:g}]，替换 {report.invalid_count} 点" if cfg.invalid_enabled else ""))
    lines.append(f"中值滤波: {'开启' if cfg.median_enabled and cfg.median_window > 1 else '关闭'}" + (f"，窗口 {cfg.median_window}" if cfg.median_enabled and cfg.median_window > 1 else ""))
    lines.append(f"低通滤波: {'开启' if cfg.lowpass_enabled else '关闭'}" + (f"，截止频率 {cfg.lowpass_cutoff_hz:g} Hz" if cfg.lowpass_enabled else ""))
    lines.append(f"起点归零: {'是' if cfg.zero_start else '否'}，终点回零/去线性漂移: {'是' if cfg.zero_end else '否'}")
    lines.append(f"缩放系数: {cfg.scale_factor:g}")
    lines.append("-- 原始数据 --")
    lines.append(f"start={report.raw_start:.9g}, end={report.raw_end:.9g}, min={report.raw_min:.9g}, max={report.raw_max:.9g}")
    lines.append(f"最大绝对速度≈{report.raw_max_abs_velocity:.9g} 单位/s，最大绝对加速度≈{report.raw_max_abs_acceleration:.9g} 单位/s²")
    lines.append("-- 处理后数据 --")
    lines.append(f"start={report.processed_start:.9g}, end={report.processed_end:.9g}, min={report.processed_min:.9g}, max={report.processed_max:.9g}")
    lines.append(f"最大绝对速度≈{report.processed_max_abs_velocity:.9g} 单位/s，最大绝对加速度≈{report.processed_max_abs_acceleration:.9g} 单位/s²")
    lines.append("================================")
    return "\n".join(lines)


def clip_unit(values: np.ndarray, g_value: float) -> Tuple[np.ndarray, int]:
    r = np.asarray(values, dtype=float) / g_value
    clipped = np.clip(r, -1.0, 1.0)
    count = int(np.count_nonzero(np.abs(r) > 1.0))
    return clipped, count


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """把 N×3 加速度向量归一化为单位方向向量。"""
    v = np.asarray(vectors, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("三轴矢量数据必须是 N×3 数组。")
    if v.shape[0] == 0:
        raise ValueError("三轴矢量数据为空。")
    norms = np.linalg.norm(v, axis=1)
    ok = np.isfinite(norms) & (norms > 1e-12)
    if not np.all(ok):
        raise ValueError("三轴数据中存在模长为 0 或非有限值的点，无法归一化。")
    return v / norms[:, None]


def estimate_rotation_axis_pca(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    自动估计单轴摆动的旋转轴。

    大角度、点云圆弧明显时：单位重力向量满足 n·v = 常数，点云所在平面的法向量 n 就是旋转轴。
    小角度、点云近似一条线时：平面法向量不稳定，改用 n ≈ mean(v) × principal_direction。
    返回：单位旋转轴 n、协方差特征值、估计模式说明。
    """
    v = normalize_vectors(vectors)
    if v.shape[0] < 3:
        raise ValueError("自动估计旋转轴至少需要 3 个三轴数据点。")
    mean_v = np.mean(v, axis=0)
    mean_norm = float(np.linalg.norm(mean_v))
    if mean_norm <= 1e-12:
        raise ValueError("平均重力方向异常，无法自动估计旋转轴。")
    centered = v - mean_v
    if float(np.linalg.norm(centered)) < 1e-12:
        raise ValueError("选段内姿态几乎没有变化，无法自动估计旋转轴。请扩大截取范围或手动输入旋转轴。")
    cov = centered.T @ centered / max(1, v.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)
    vals = eigvals[order]
    vecs = eigvecs[:, order]

    # 如果点云基本是一条线，平面法向量会很容易受噪声影响。
    # 此时主变化方向近似切向量 d，旋转轴 n 满足 d ≈ n × mean(v)，因此 n ≈ mean(v) × d。
    line_like = vals[2] > 0 and (vals[1] / vals[2] < 0.08)
    if line_like:
        principal = np.asarray(vecs[:, 2], dtype=float)
        axis = np.cross(mean_v / mean_norm, principal)
        mode = "小角度线性化估计：点云近似一条线，使用 平均重力方向 × 主变化方向 估计旋转轴。"
    else:
        axis = np.asarray(vecs[:, 0], dtype=float)
        mode = "平面拟合估计：点云圆弧较明显，使用点云所在平面的法向量作为旋转轴。"

    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        raise ValueError("自动估计旋转轴失败。请改用手动输入旋转轴。")
    return axis / axis_norm, vals, mode


def vector_angle_degrees(xyz_values: np.ndarray, cfg: AngleConversionConfig) -> Tuple[np.ndarray, str]:
    """三轴矢量法：绕旋转轴把重力方向变化转换为相对角度。"""
    xyz = np.asarray(xyz_values, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("三轴矢量法需要 X/Y/Z 三列数据。")
    if xyz.shape[0] == 0:
        raise ValueError("截取后的三轴数据为空。")
    if cfg.g_value <= 0 or not math.isfinite(cfg.g_value):
        raise ValueError("g 基准值必须大于 0。若加速度单位为 g，通常填 1；若单位为 m/s²，通常填 9.80665。")

    v = normalize_vectors(xyz)
    source = cfg.rotation_axis_source.strip() or "自动估计"
    eigvals = None
    if source.startswith("自动"):
        n, eigvals, auto_mode = estimate_rotation_axis_pca(xyz)
        source_desc = "自动估计旋转轴：" + auto_mode
    else:
        n = np.asarray([cfg.rotation_axis_x, cfg.rotation_axis_y, cfg.rotation_axis_z], dtype=float)
        if not np.all(np.isfinite(n)):
            raise ValueError("手动旋转轴包含非有限数值。")
        norm_n = float(np.linalg.norm(n))
        if norm_n <= 1e-12:
            raise ValueError("手动旋转轴不能为 [0,0,0]。")
        n = n / norm_n
        source_desc = "手动输入旋转轴：软件将三轴重力方向投影到垂直于该旋转轴的平面，再计算相对角度。"

    # 把每一帧的重力方向投影到垂直于旋转轴的平面。
    p = v - (v @ n)[:, None] * n[None, :]
    p_norm = np.linalg.norm(p, axis=1)
    if float(np.nanmax(p_norm)) <= 1e-9:
        raise ValueError("重力方向几乎平行于旋转轴，无法由加速度计重力分量稳定计算该轴转角。")
    bad = p_norm < 1e-9
    if np.any(bad):
        # 极少数奇异点用相邻非奇异点线性插值不方便，这里直接剔除会破坏长度；改用最小范数夹紧。
        p_norm[bad] = 1e-9
    p_unit = p / p_norm[:, None]

    baseline_n = int(max(1, min(cfg.baseline_points, p_unit.shape[0])))
    p0 = np.mean(p_unit[:baseline_n], axis=0)
    p0 = p0 - float(np.dot(p0, n)) * n
    p0_norm = float(np.linalg.norm(p0))
    if p0_norm <= 1e-9:
        raise ValueError("基线段投影过小，无法建立角度零位。请更换基线点数或旋转轴。")
    p0 = p0 / p0_norm

    cross_terms = np.cross(p0[None, :], p_unit)
    sin_terms = cross_terms @ n
    cos_terms = p_unit @ p0
    angle = np.degrees(np.unwrap(np.arctan2(sin_terms, cos_terms)))
    baseline = float(np.mean(angle[:baseline_n]))
    displacement = angle - baseline

    if cfg.angle_reverse:
        displacement = -displacement

    if cfg.angle_reverse:
        displacement = -displacement

    if cfg.angle_zero_end and displacement.size >= 2:
        drift = np.linspace(0.0, float(displacement[-1]), displacement.size)
        displacement = displacement - drift

    if cfg.angle_scale_factor != 1.0:
        displacement = displacement * cfg.angle_scale_factor

    text = []
    text.append("========== 三轴矢量法加速度转角度 ==========")
    text.append("转换方法: 三轴矢量法(绕旋转轴)")
    text.append(source_desc)
    text.append(f"截取点数: {xyz.shape[0]}")
    text.append(f"旋转轴 n = [{n[0]:.9g}, {n[1]:.9g}, {n[2]:.9g}]")
    if eigvals is not None:
        text.append(f"PCA特征值(小→大): {eigvals[0]:.4g}, {eigvals[1]:.4g}, {eigvals[2]:.4g}")
        if eigvals[2] > 0 and eigvals[1] / eigvals[2] < 0.08:
            text.append("提示: 选段角度幅度较小，软件已使用小角度线性化方法估计旋转轴。")
        elif eigvals[1] > 0 and eigvals[0] / eigvals[1] > 0.25:
            text.append("提示: 最小/中间特征值比值偏大，说明选段噪声较大或并非严格单轴摆动，自动轴估计可信度可能下降。")
    text.append(f"基线点数: {baseline_n}，基线角度: {baseline:.9g} °")
    text.append(f"角度取反: {'是' if cfg.angle_reverse else '否'}")
    text.append(f"终点回零/去线性漂移: {'是' if cfg.angle_zero_end else '否'}")
    text.append(f"角度取反: {'是' if cfg.angle_reverse else '否'}")
    text.append(f"终点回零/去线性漂移: {'是' if cfg.angle_zero_end else '否'}")
    text.append(f"角度缩放系数: {cfg.angle_scale_factor:g}")
    text.append(f"输出起点={displacement[0]:.9g} °，终点={displacement[-1]:.9g} °，min={np.nanmin(displacement):.9g} °，max={np.nanmax(displacement):.9g} °")
    text.append("说明: 该方法不要求 X/Y/Z 任一轴对准重力方向，但要求运动主要是单轴摆动，且线性加速度/振动相对较小。")
    text.append("================================")
    return displacement, "\n".join(text)


def canonical_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.strip().lower())


def find_header_index(headers: Sequence[str], candidates: Sequence[str]) -> Optional[int]:
    lookup = {canonical_header(h): i for i, h in enumerate(headers)}
    for c in candidates:
        idx = lookup.get(canonical_header(c))
        if idx is not None:
            return idx
    return None


def axis_to_angle_degrees(axis_values: np.ndarray, cfg: AngleConversionConfig, reference_values: Optional[np.ndarray] = None) -> Tuple[np.ndarray, str]:
    a = np.asarray(axis_values, dtype=float)
    if a.size == 0:
        raise ValueError("截取后的轴数据为空。")
    if cfg.g_value <= 0 or not math.isfinite(cfg.g_value):
        raise ValueError("g 基准值必须大于 0。若加速度单位为 g，通常填 1；若单位为 m/s²，通常填 9.80665。")
    method = cfg.method
    clip_count = 0

    if method.startswith("单轴 asin"):
        r, clip_count = clip_unit(a, cfg.g_value)
        angle = np.degrees(np.arcsin(r))
        desc = "单轴 asin(a/g)：适合所选轴在目标角度附近过 0g 的情况。"
    elif method.startswith("单轴 acos"):
        r, clip_count = clip_unit(a, cfg.g_value)
        angle = np.degrees(np.arccos(r))
        desc = "单轴 acos(a/g)：适合所选轴在目标角度附近接近 ±1g 的情况。"
    elif method.startswith("双轴 atan2"):
        if reference_values is None:
            raise ValueError("双轴 atan2 需要参考轴数据。")
        ref = np.asarray(reference_values, dtype=float)
        if ref.size != a.size:
            raise ValueError("参考轴长度与所选轴长度不一致。")
        angle = np.degrees(np.arctan2(a, ref))
        desc = "双轴 atan2(所选轴, 参考轴)：通常比单轴 asin/acos 更稳，适合例如 atan2(Z, X)。"
    else:
        raise ValueError(f"未知角度转换方法: {method}")

    baseline_n = int(max(1, min(cfg.baseline_points, angle.size)))
    baseline = float(np.mean(angle[:baseline_n]))
    displacement = angle - baseline

    if cfg.angle_reverse:
        displacement = -displacement

    if cfg.angle_zero_end and displacement.size >= 2:
        drift = np.linspace(0.0, float(displacement[-1]), displacement.size)
        displacement = displacement - drift

    if cfg.angle_scale_factor != 1.0:
        displacement = displacement * cfg.angle_scale_factor

    text = []
    text.append("========== 加速度转角度 ==========")
    text.append(f"转换方法: {method}")
    text.append(desc)
    text.append(f"所选轴: {cfg.axis_name}" + (f"，参考轴: {cfg.reference_axis_name}" if method.startswith("双轴 atan2") else ""))
    text.append(f"g 基准值: {cfg.g_value:g}")
    text.append(f"截取点数: {a.size}")
    text.append(f"基线点数: {baseline_n}，基线角度: {baseline:.9g} °")
    if clip_count:
        text.append(f"注意: 有 {clip_count} 个点超出 [-g, +g]，已在反三角函数前夹紧到 [-1, 1]。")
    text.append(f"角度缩放系数: {cfg.angle_scale_factor:g}")
    text.append(f"输出起点={displacement[0]:.9g} °，终点={displacement[-1]:.9g} °，min={np.nanmin(displacement):.9g} °，max={np.nanmax(displacement):.9g} °")
    text.append("================================")
    return displacement, "\n".join(text)


class SignalPreprocessApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_TITLE} v{APP_VERSION}")
        self.geometry("1480x860")
        self.minsize(1180, 760)

        self.raw_data: Optional[np.ndarray] = None
        self.processed_data: Optional[np.ndarray] = None
        self.angle_data: Optional[np.ndarray] = None
        self.loaded_path: Optional[Path] = None
        self.last_report: Optional[PreprocessReport] = None

        self.table_headers: list[str] = []
        self.table_data: Optional[np.ndarray] = None
        self.axis_column_lookup: dict[str, int] = {}
        self.current_source_kind: str = "sequence"

        self.span_selector: Optional[SpanSelector] = None
        self.selection_patch_axes = None

        self._init_vars()
        self._build_ui()
        self.restore_demo_data()

    def _init_vars(self) -> None:
        cfg = PreprocessConfig()
        self.interval_var = tk.StringVar(value=str(cfg.interval_s))
        self.column_var = tk.StringVar(value=cfg.column_index)
        self.scale_var = tk.StringVar(value=str(cfg.scale_factor))
        self.invalid_enabled_var = tk.BooleanVar(value=cfg.invalid_enabled)
        self.min_value_var = tk.StringVar(value=str(cfg.min_value))
        self.max_value_var = tk.StringVar(value=str(cfg.max_value))
        self.median_enabled_var = tk.BooleanVar(value=cfg.median_enabled)
        self.median_window_var = tk.StringVar(value=str(cfg.median_window))
        self.lowpass_enabled_var = tk.BooleanVar(value=cfg.lowpass_enabled)
        self.lowpass_cutoff_var = tk.StringVar(value=str(cfg.lowpass_cutoff_hz))
        self.zero_start_var = tk.BooleanVar(value=cfg.zero_start)
        self.zero_end_var = tk.BooleanVar(value=cfg.zero_end)
        self.plot_max_points_var = tk.StringVar(value=str(cfg.plot_max_points))
        self.output_precision_var = tk.StringVar(value=str(cfg.output_precision))
        self.plot_mode_var = tk.StringVar(value="位移+速度+加速度")

        acfg = AngleConversionConfig()
        self.axis_var = tk.StringVar(value=acfg.axis_name)
        self.reference_axis_var = tk.StringVar(value=acfg.reference_axis_name)
        self.angle_method_var = tk.StringVar(value=acfg.method)
        self.g_value_var = tk.StringVar(value=str(acfg.g_value))
        self.segment_start_var = tk.StringVar(value=str(acfg.start_index_1based))
        self.segment_end_var = tk.StringVar(value="")
        self.baseline_points_var = tk.StringVar(value=str(acfg.baseline_points))
        self.angle_scale_var = tk.StringVar(value=str(acfg.angle_scale_factor))
        self.angle_zero_end_var = tk.BooleanVar(value=acfg.angle_zero_end)
        self.angle_reverse_var = tk.BooleanVar(value=acfg.angle_reverse)
        self.rotation_axis_source_var = tk.StringVar(value=acfg.rotation_axis_source)
        self.rotation_axis_x_var = tk.StringVar(value=str(acfg.rotation_axis_x))
        self.rotation_axis_y_var = tk.StringVar(value=str(acfg.rotation_axis_y))
        self.rotation_axis_z_var = tk.StringVar(value=str(acfg.rotation_axis_z))

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        left_container = ttk.Frame(root)
        left_container.pack(side="left", fill="y", padx=(0, 8))
        right = ttk.Frame(root)
        right.pack(side="right", fill="both", expand=True)

        # 左侧参数较多，放入滚动区域。
        canvas = tk.Canvas(left_container, width=470, highlightthickness=0)
        scroll = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        left = ttk.Frame(canvas)
        left.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=left, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="y", expand=False)
        scroll.pack(side="right", fill="y")

        self._build_file_frame(left)
        self._build_accel_frame(left)
        self._build_param_frame(left)
        self._build_action_frame(left)
        self._build_log_frame(left)
        self._build_plot_frame(right)

    def _entry_row(self, parent, row, label, var, width=12, unit="") -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e", padx=4, pady=3)
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=1, sticky="w", padx=4, pady=3)
        if unit:
            ttk.Label(parent, text=unit).grid(row=row, column=2, sticky="w", padx=4, pady=3)

    def _build_file_frame(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="数据导入", padding=8)
        frame.pack(fill="x", pady=(0, 8))

        ttk.Button(frame, text="打开 TXT/CSV", command=self.load_file).grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(frame, text="粘贴/编辑数据", command=self.open_text_editor).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(frame, text="恢复单列示例", command=self.restore_demo_data).grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(frame, text="清空", command=self.clear_data).grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        ttk.Label(frame, text="列号").grid(row=2, column=0, sticky="e", padx=4, pady=3)
        ttk.Entry(frame, textvariable=self.column_var, width=10).grid(row=2, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(frame, text="单列数据可留空；普通表格可填 1/2/3...").grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

    def _build_accel_frame(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="三轴加速度 → 摆台角度位移", padding=8)
        frame.pack(fill="x", pady=(0, 8))

        ttk.Label(frame, text="所选轴").grid(row=0, column=0, sticky="e", padx=4, pady=3)
        self.axis_combo = ttk.Combobox(frame, textvariable=self.axis_var, values=["X", "Y", "Z", "Sum"], width=18, state="readonly")
        self.axis_combo.grid(row=0, column=1, sticky="w", padx=4, pady=3)
        self.axis_combo.bind("<<ComboboxSelected>>", lambda _e: self.use_selected_axis())
        ttk.Button(frame, text="绘制所选轴", command=self.use_selected_axis).grid(row=0, column=2, sticky="ew", padx=4, pady=3)

        ttk.Label(frame, text="转换方法").grid(row=1, column=0, sticky="e", padx=4, pady=3)
        method_combo = ttk.Combobox(
            frame,
            textvariable=self.angle_method_var,
            values=["单轴 asin(a/g)", "单轴 acos(a/g)", "双轴 atan2(轴,参考轴)", "三轴矢量法(绕旋转轴)"],
            width=24,
            state="readonly",
        )
        method_combo.grid(row=1, column=1, columnspan=2, sticky="w", padx=4, pady=3)

        ttk.Label(frame, text="参考轴").grid(row=2, column=0, sticky="e", padx=4, pady=3)
        self.ref_axis_combo = ttk.Combobox(frame, textvariable=self.reference_axis_var, values=["X", "Y", "Z", "Sum"], width=18, state="readonly")
        self.ref_axis_combo.grid(row=2, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(frame, text="仅 atan2 使用").grid(row=2, column=2, sticky="w", padx=4, pady=3)

        self._entry_row(frame, 3, "g 基准", self.g_value_var, 10, "g单位填1；m/s²填9.80665")

        ttk.Separator(frame, orient="horizontal").grid(row=4, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Label(frame, text="旋转轴来源").grid(row=5, column=0, sticky="e", padx=4, pady=3)
        ttk.Combobox(
            frame,
            textvariable=self.rotation_axis_source_var,
            values=["自动估计", "手动输入"],
            width=18,
            state="readonly",
        ).grid(row=5, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(frame, text="仅三轴矢量法使用").grid(row=5, column=2, sticky="w", padx=4, pady=3)

        ttk.Label(frame, text="手动旋转轴").grid(row=6, column=0, sticky="e", padx=4, pady=3)
        axis_frame = ttk.Frame(frame)
        axis_frame.grid(row=6, column=1, columnspan=2, sticky="w", padx=4, pady=3)
        ttk.Label(axis_frame, text="nx").pack(side="left")
        ttk.Entry(axis_frame, textvariable=self.rotation_axis_x_var, width=6).pack(side="left", padx=(2, 6))
        ttk.Label(axis_frame, text="ny").pack(side="left")
        ttk.Entry(axis_frame, textvariable=self.rotation_axis_y_var, width=6).pack(side="left", padx=(2, 6))
        ttk.Label(axis_frame, text="nz").pack(side="left")
        ttk.Entry(axis_frame, textvariable=self.rotation_axis_z_var, width=6).pack(side="left", padx=(2, 0))

        ttk.Separator(frame, orient="horizontal").grid(row=7, column=0, columnspan=3, sticky="ew", pady=5)
        self._entry_row(frame, 8, "起始点序号", self.segment_start_var, 10, "从1开始")
        self._entry_row(frame, 9, "结束点序号", self.segment_end_var, 10, "留空=到末尾")
        ttk.Button(frame, text="拖拽图中波形可自动填写截取范围", command=self.update_plot).grid(row=10, column=0, columnspan=3, sticky="ew", padx=4, pady=3)

        self._entry_row(frame, 11, "基线点数", self.baseline_points_var, 10, "用前N点做角度零点")
        self._entry_row(frame, 12, "角度缩放", self.angle_scale_var, 10, "转换后再乘")
        ttk.Checkbutton(frame, text="角度取反", variable=self.angle_reverse_var).grid(row=13, column=0, sticky="w", padx=4, pady=3)
        ttk.Checkbutton(frame, text="角度终点回零 / 去线性漂移", variable=self.angle_zero_end_var).grid(row=13, column=1, columnspan=2, sticky="w", padx=4, pady=3)

        ttk.Button(frame, text="截取选段 → 角度位移", command=self.convert_segment_to_angle).grid(row=14, column=0, columnspan=3, sticky="ew", padx=4, pady=4)
        ttk.Button(frame, text="导出角度位移 TXT", command=self.export_angle_data).grid(row=15, column=0, columnspan=3, sticky="ew", padx=4, pady=4)

        help_text = "提示：没有任何轴对准重力方向时，优先用“三轴矢量法(绕旋转轴)”；可自动估计旋转轴，也可手动输入 nx,ny,nz。"
        ttk.Label(frame, text=help_text, wraplength=420, foreground="#555555").grid(row=16, column=0, columnspan=3, sticky="w", padx=4, pady=(4, 0))

    def _build_param_frame(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="通用预处理参数", padding=8)
        frame.pack(fill="x", pady=(0, 8))

        self._entry_row(frame, 0, "时间间隔", self.interval_var, 10, "s")
        self._entry_row(frame, 1, "缩放系数", self.scale_var, 10, "预处理导出前乘")

        ttk.Separator(frame, orient="horizontal").grid(row=2, column=0, columnspan=3, sticky="ew", pady=6)
        ttk.Checkbutton(frame, text="异常点插值", variable=self.invalid_enabled_var).grid(row=3, column=0, sticky="w", padx=4, pady=3)
        ttk.Label(frame, text="有效范围").grid(row=3, column=1, sticky="e", padx=4, pady=3)
        range_frame = ttk.Frame(frame)
        range_frame.grid(row=3, column=2, sticky="w")
        ttk.Entry(range_frame, textvariable=self.min_value_var, width=8).pack(side="left")
        ttk.Label(range_frame, text="~").pack(side="left", padx=3)
        ttk.Entry(range_frame, textvariable=self.max_value_var, width=8).pack(side="left")

        ttk.Checkbutton(frame, text="中值滤波", variable=self.median_enabled_var).grid(row=4, column=0, sticky="w", padx=4, pady=3)
        self._entry_row(frame, 4, "窗口", self.median_window_var, 10, "建议 3/5/7")

        ttk.Checkbutton(frame, text="低通滤波", variable=self.lowpass_enabled_var).grid(row=5, column=0, sticky="w", padx=4, pady=3)
        self._entry_row(frame, 5, "截止频率", self.lowpass_cutoff_var, 10, "Hz")

        ttk.Checkbutton(frame, text="起点归零", variable=self.zero_start_var).grid(row=6, column=0, sticky="w", padx=4, pady=3)
        ttk.Checkbutton(frame, text="终点回零 / 去线性漂移", variable=self.zero_end_var).grid(row=6, column=1, columnspan=2, sticky="w", padx=4, pady=3)

        ttk.Separator(frame, orient="horizontal").grid(row=7, column=0, columnspan=3, sticky="ew", pady=6)
        self._entry_row(frame, 8, "绘图最大点数", self.plot_max_points_var, 10, "过大可能变慢")
        self._entry_row(frame, 9, "导出小数位", self.output_precision_var, 10, "位")

    def _build_action_frame(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="操作", padding=8)
        frame.pack(fill="x", pady=(0, 8))
        ttk.Button(frame, text="预处理并绘图", command=self.apply_and_plot).pack(fill="x", padx=4, pady=4)
        ttk.Button(frame, text="只重绘当前数据", command=self.update_plot).pack(fill="x", padx=4, pady=4)
        ttk.Button(frame, text="导出处理后 TXT", command=self.export_processed).pack(fill="x", padx=4, pady=4)
        ttk.Button(frame, text="保存当前图 PNG", command=self.save_plot_png).pack(fill="x", padx=4, pady=4)
        ttk.Button(frame, text="保存配置 JSON", command=self.save_config).pack(fill="x", padx=4, pady=4)
        ttk.Button(frame, text="读取配置 JSON", command=self.load_config).pack(fill="x", padx=4, pady=4)

    def _build_log_frame(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="日志 / 统计", padding=8)
        frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(frame, height=16, width=55, wrap="word", state="disabled")
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=yscroll.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

    def _build_plot_frame(self, parent) -> None:
        control = ttk.Frame(parent)
        control.pack(fill="x", pady=(0, 6))
        ttk.Label(control, text="绘图模式").pack(side="left", padx=4)
        combo = ttk.Combobox(
            control,
            textvariable=self.plot_mode_var,
            values=["位移+速度+加速度", "仅位移", "仅速度", "仅加速度"],
            width=18,
            state="readonly",
        )
        combo.pack(side="left", padx=4)
        combo.bind("<<ComboboxSelected>>", lambda _e: self.update_plot())
        ttk.Label(control, text="在波形图上横向拖拽可选择截取区间").pack(side="left", padx=14)

        self.figure = Figure(figsize=(9.2, 6.4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()

    def current_config(self) -> PreprocessConfig:
        try:
            cfg = PreprocessConfig(
                interval_s=float(self.interval_var.get()),
                column_index=self.column_var.get().strip(),
                scale_factor=float(self.scale_var.get()),
                invalid_enabled=bool(self.invalid_enabled_var.get()),
                min_value=float(self.min_value_var.get()),
                max_value=float(self.max_value_var.get()),
                median_enabled=bool(self.median_enabled_var.get()),
                median_window=int(float(self.median_window_var.get())),
                lowpass_enabled=bool(self.lowpass_enabled_var.get()),
                lowpass_cutoff_hz=float(self.lowpass_cutoff_var.get()),
                zero_start=bool(self.zero_start_var.get()),
                zero_end=bool(self.zero_end_var.get()),
                plot_max_points=int(float(self.plot_max_points_var.get())),
                output_precision=int(float(self.output_precision_var.get())),
            )
            if cfg.interval_s <= 0:
                raise ValueError("时间间隔必须大于 0。")
            if cfg.invalid_enabled and cfg.min_value >= cfg.max_value:
                raise ValueError("异常点有效范围的最小值必须小于最大值。")
            if cfg.median_window < 1:
                raise ValueError("中值滤波窗口必须 >= 1。")
            if cfg.output_precision < 0 or cfg.output_precision > 16:
                raise ValueError("导出小数位建议在 0~16。")
            return cfg
        except Exception as exc:
            raise ValueError(f"参数设置错误：{exc}") from exc

    def current_angle_config(self) -> AngleConversionConfig:
        try:
            start = int(float(self.segment_start_var.get())) if self.segment_start_var.get().strip() else 1
            end = int(float(self.segment_end_var.get())) if self.segment_end_var.get().strip() else 0
            cfg = AngleConversionConfig(
                axis_name=self.axis_var.get().strip(),
                reference_axis_name=self.reference_axis_var.get().strip(),
                method=self.angle_method_var.get().strip(),
                g_value=float(self.g_value_var.get()),
                start_index_1based=start,
                end_index_1based=end,
                baseline_points=int(float(self.baseline_points_var.get())),
                angle_scale_factor=float(self.angle_scale_var.get()),
                angle_zero_end=bool(self.angle_zero_end_var.get()),
                angle_reverse=bool(self.angle_reverse_var.get()),
                rotation_axis_source=self.rotation_axis_source_var.get().strip(),
                rotation_axis_x=float(self.rotation_axis_x_var.get()),
                rotation_axis_y=float(self.rotation_axis_y_var.get()),
                rotation_axis_z=float(self.rotation_axis_z_var.get()),
            )
            if cfg.start_index_1based < 1:
                raise ValueError("起始点序号必须 >= 1。")
            if cfg.end_index_1based and cfg.end_index_1based < cfg.start_index_1based:
                raise ValueError("结束点序号必须大于等于起始点序号。")
            if cfg.baseline_points < 1:
                raise ValueError("基线点数必须 >= 1。")
            if cfg.angle_scale_factor < 0 or not math.isfinite(cfg.angle_scale_factor):
                raise ValueError("角度缩放系数必须是大于等于 0 的有限数值。")
            if cfg.method.startswith("三轴矢量法") and cfg.rotation_axis_source.startswith("手动"):
                n = np.asarray([cfg.rotation_axis_x, cfg.rotation_axis_y, cfg.rotation_axis_z], dtype=float)
                if not np.all(np.isfinite(n)) or float(np.linalg.norm(n)) <= 1e-12:
                    raise ValueError("手动旋转轴必须是非零的有限三维向量。")
            return cfg
        except Exception as exc:
            raise ValueError(f"角度转换参数错误：{exc}") from exc

    def log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def set_data(self, data: np.ndarray, source: str = "", kind: str = "sequence") -> None:
        self.raw_data = np.asarray(data, dtype=float)
        self.processed_data = None
        self.last_report = None
        self.current_source_kind = kind
        self.clear_log()
        self.log(f"已载入数据: {source or '文本/示例'}")
        self.log(f"点数: {self.raw_data.size}")
        self.log(f"范围: min={np.nanmin(self.raw_data):.9g}, max={np.nanmax(self.raw_data):.9g}")
        self.log(f"起点={self.raw_data[0]:.9g}, 终点={self.raw_data[-1]:.9g}")
        self.update_plot()

    def load_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择采集数据文件",
            filetypes=[("Text / CSV", "*.txt *.csv *.dat *.tsv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            cfg = self.current_config()
            self.loaded_path = Path(path)
            text = read_text_file(path)

            # 优先尝试按表格读取。若表格列数 >= 2，启用轴选择；否则按单列序列处理。
            headers, table = parse_numeric_table(text)
            if table.ndim == 2 and table.shape[1] >= 2:
                self.table_headers = headers
                self.table_data = table
                self.refresh_axis_choices()
                default_axis = self.pick_default_axis(headers)
                self.axis_var.set(default_axis)
                # 如果参考轴可用，给 atan2 默认参考轴设为 X。
                if "X" in self.axis_column_lookup:
                    self.reference_axis_var.set("X")
                self.use_selected_axis(silent=True)
                self.clear_log()
                self.log(f"已载入表格数据: {self.loaded_path.name}")
                self.log(f"列: {', '.join(headers)}")
                self.log(f"行数: {table.shape[0]}，列数: {table.shape[1]}")
                self.log(f"当前绘制轴: {self.axis_var.get()}")
                self.log("可在“三轴加速度→摆台角度位移”区域切换 X/Y/Z/Sum，并拖拽图中波形截取片段；没有单轴对准重力方向时可选三轴矢量法。")
                self.update_plot()
            else:
                data = parse_sequence_text(text, column_index=cfg.column_index)
                self.table_headers = []
                self.table_data = None
                self.refresh_axis_choices()
                self.set_data(data, source=str(self.loaded_path.name), kind="sequence")
        except Exception as exc:
            messagebox.showerror("导入失败", str(exc))

    def pick_default_axis(self, headers: Sequence[str]) -> str:
        for name in ["Z", "Y", "X", "Sum", "sum"]:
            if name in headers:
                return name
        # 避免默认选择 Packet num / Packet length，优先第三列。
        if len(headers) >= 5:
            return headers[4] if headers[4].upper() == "Z" else headers[2]
        return headers[0]

    def refresh_axis_choices(self) -> None:
        if self.table_data is not None and self.table_headers:
            values = list(self.table_headers)
        else:
            values = ["X", "Y", "Z", "Sum"]
        self.axis_column_lookup = {name: i for i, name in enumerate(values)}
        self.axis_combo.configure(values=values)
        self.ref_axis_combo.configure(values=values)

    def use_selected_axis(self, silent: bool = False) -> None:
        if self.table_data is None or not self.table_headers:
            if not silent:
                messagebox.showwarning("没有三轴表格数据", "请先导入包含 X/Y/Z 列的三轴加速度文件。")
            return
        name = self.axis_var.get().strip()
        if name not in self.table_headers:
            raise ValueError(f"找不到列: {name}")
        idx = self.table_headers.index(name)
        self.raw_data = np.asarray(self.table_data[:, idx], dtype=float)
        self.processed_data = None
        self.last_report = None
        self.current_source_kind = "axis"
        if not silent:
            self.clear_log()
            self.log(f"当前绘制轴: {name}")
            self.log(f"点数: {self.raw_data.size}")
            self.log(f"范围: min={np.nanmin(self.raw_data):.9g}, max={np.nanmax(self.raw_data):.9g}")
            self.update_plot()

    def restore_demo_data(self) -> None:
        try:
            data = parse_sequence_text(DEFAULT_DEMO)
            self.loaded_path = None
            self.table_headers = []
            self.table_data = None
            self.refresh_axis_choices()
            self.set_data(data, source="内置单列示例", kind="sequence")
        except Exception as exc:
            messagebox.showerror("错误", str(exc))

    def clear_data(self) -> None:
        self.raw_data = None
        self.processed_data = None
        self.angle_data = None
        self.last_report = None
        self.table_headers = []
        self.table_data = None
        self.refresh_axis_choices()
        self.figure.clear()
        self.canvas.draw_idle()
        self.clear_log()
        self.log("已清空数据。")

    def open_text_editor(self) -> None:
        win = tk.Toplevel(self)
        win.title("粘贴/编辑数据")
        win.geometry("760x580")
        ttk.Label(win, text="可粘贴一列数字；支持逗号、空格、换行分隔；# 后为注释。三轴表格建议用“打开TXT/CSV”。", padding=6).pack(fill="x")
        text = tk.Text(win, wrap="none", undo=True)
        text.pack(fill="both", expand=True, padx=8, pady=6)
        if self.raw_data is not None:
            preview = "\n".join(f"{v:.10g}" for v in self.raw_data[:5000])
            if self.raw_data.size > 5000:
                preview += "\n# 这里只显示前 5000 点；若要保留完整数据，请从文件重新导入。"
            text.insert("1.0", preview)
        else:
            text.insert("1.0", DEFAULT_DEMO)

        def use_text() -> None:
            try:
                cfg = self.current_config()
                data = parse_sequence_text(text.get("1.0", "end"), cfg.column_index)
                self.loaded_path = None
                self.table_headers = []
                self.table_data = None
                self.refresh_axis_choices()
                self.set_data(data, source="文本框", kind="sequence")
                win.destroy()
            except Exception as exc:
                messagebox.showerror("解析失败", str(exc), parent=win)

        buttons = ttk.Frame(win)
        buttons.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(buttons, text="使用这些数据", command=use_text).pack(side="left", padx=4)
        ttk.Button(buttons, text="取消", command=win.destroy).pack(side="right", padx=4)

    def apply_and_plot(self) -> None:
        if self.raw_data is None:
            messagebox.showwarning("没有数据", "请先导入数据。")
            return
        try:
            cfg = self.current_config()
            processed, report = preprocess_data(self.raw_data, cfg)
            self.processed_data = processed
            self.last_report = report
            self.clear_log()
            if self.loaded_path:
                self.log(f"文件: {self.loaded_path}")
            self.log(report_to_text(report, cfg))
            self.update_plot()
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("预处理失败", str(exc))

    def get_segment_indices(self, n: int) -> Tuple[int, int]:
        cfg = self.current_angle_config()
        start0 = max(0, cfg.start_index_1based - 1)
        end1 = cfg.end_index_1based if cfg.end_index_1based else n
        end1 = min(n, end1)
        if start0 >= end1:
            raise ValueError("截取范围为空。请检查起始点和结束点序号。")
        return start0, end1

    def on_span_select(self, xmin: float, xmax: float) -> None:
        try:
            cfg = self.current_config()
            n = self.raw_data.size if self.raw_data is not None else 0
            if n <= 0:
                return
            lo = min(xmin, xmax)
            hi = max(xmin, xmax)
            start0 = max(0, int(math.floor(lo / cfg.interval_s)))
            end0 = min(n - 1, int(math.ceil(hi / cfg.interval_s)))
            self.segment_start_var.set(str(start0 + 1))
            self.segment_end_var.set(str(end0 + 1))
            self.log(f"已选择截取范围: 第 {start0 + 1} 点 到 第 {end0 + 1} 点")
            self.update_plot()
        except Exception:
            pass

    def convert_segment_to_angle(self) -> None:
        if self.table_data is None or not self.table_headers:
            messagebox.showwarning("没有三轴加速度数据", "请先导入包含 X/Y/Z 列的加速度表格数据。")
            return
        try:
            acfg = self.current_angle_config()
            start0, end1 = self.get_segment_indices(self.table_data.shape[0])

            if acfg.method.startswith("三轴矢量法"):
                x_idx = find_header_index(self.table_headers, ["X", "AccX", "AccelX", "Ax", "acc_x", "accel_x"])
                y_idx = find_header_index(self.table_headers, ["Y", "AccY", "AccelY", "Ay", "acc_y", "accel_y"])
                z_idx = find_header_index(self.table_headers, ["Z", "AccZ", "AccelZ", "Az", "acc_z", "accel_z"])
                if x_idx is None or y_idx is None or z_idx is None:
                    raise ValueError("三轴矢量法需要能识别 X/Y/Z 三列。请确认表头包含 X、Y、Z。")
                xyz_segment = np.asarray(self.table_data[start0:end1, [x_idx, y_idx, z_idx]], dtype=float)
                angle, text = vector_angle_degrees(xyz_segment, acfg)
            else:
                if acfg.axis_name not in self.table_headers:
                    raise ValueError(f"找不到所选轴列: {acfg.axis_name}")
                axis_idx = self.table_headers.index(acfg.axis_name)
                axis_segment = np.asarray(self.table_data[start0:end1, axis_idx], dtype=float)

                ref_segment: Optional[np.ndarray] = None
                if acfg.method.startswith("双轴 atan2"):
                    if acfg.reference_axis_name not in self.table_headers:
                        raise ValueError(f"找不到参考轴列: {acfg.reference_axis_name}")
                    ref_idx = self.table_headers.index(acfg.reference_axis_name)
                    ref_segment = np.asarray(self.table_data[start0:end1, ref_idx], dtype=float)

                angle, text = axis_to_angle_degrees(axis_segment, acfg, reference_values=ref_segment)
            self.angle_data = angle
            # 把角度位移作为当前原始数据，方便继续使用通用预处理和导出。
            self.raw_data = angle.copy()
            self.processed_data = None
            self.last_report = None
            self.current_source_kind = "angle"
            self.clear_log()
            if self.loaded_path:
                self.log(f"文件: {self.loaded_path.name}")
            self.log(f"截取范围: 第 {start0 + 1} 点 到 第 {end1} 点")
            self.log(text)
            self.log("已将角度位移设为当前数据。可继续使用通用预处理，或直接点击“导出角度位移TXT”。")
            self.update_plot()
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("角度转换失败", str(exc))

    def update_plot(self) -> None:
        try:
            cfg = self.current_config()
        except Exception:
            cfg = PreprocessConfig()
        self.figure.clear()
        self.span_selector = None
        if self.raw_data is None:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Please load data first", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        raw = np.asarray(self.raw_data, dtype=float)
        proc = np.asarray(self.processed_data, dtype=float) if self.processed_data is not None else None
        mode = self.plot_mode_var.get()
        max_points = max(100, int(cfg.plot_max_points))
        t = np.arange(raw.size) * cfg.interval_s

        main_ax = None
        if mode == "位移+速度+加速度":
            axes = [self.figure.add_subplot(311), self.figure.add_subplot(312), self.figure.add_subplot(313)]
            main_ax = axes[0]
            self._plot_displacement(axes[0], t, raw, proc, max_points)
            self._plot_velocity(axes[1], raw, proc, cfg.interval_s, max_points)
            self._plot_acceleration(axes[2], raw, proc, cfg.interval_s, max_points)
        elif mode == "仅位移":
            ax = self.figure.add_subplot(111)
            main_ax = ax
            self._plot_displacement(ax, t, raw, proc, max_points)
        elif mode == "仅速度":
            ax = self.figure.add_subplot(111)
            self._plot_velocity(ax, raw, proc, cfg.interval_s, max_points)
        else:
            ax = self.figure.add_subplot(111)
            self._plot_acceleration(ax, raw, proc, cfg.interval_s, max_points)

        if main_ax is not None:
            self.draw_selection_span(main_ax, raw.size, cfg.interval_s)
            self.span_selector = SpanSelector(
                main_ax,
                self.on_span_select,
                "horizontal",
                useblit=True,
                props=dict(alpha=0.18),
                interactive=False,
            )

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def draw_selection_span(self, ax, n: int, interval_s: float) -> None:
        try:
            start0, end1 = self.get_segment_indices(n)
            x0 = start0 * interval_s
            x1 = max(start0, end1 - 1) * interval_s
            if x1 > x0:
                ax.axvspan(x0, x1, alpha=0.12, label="Selected segment")
        except Exception:
            return

    def _plot_displacement(self, ax, t: np.ndarray, raw: np.ndarray, proc: Optional[np.ndarray], max_points: int) -> None:
        tx, y = downsample_for_plot(t, raw, max_points)
        raw_label = "Raw angle displacement" if self.current_source_kind == "angle" else (f"Selected axis {self.axis_var.get()}" if self.current_source_kind == "axis" else "Raw")
        ax.plot(tx, y, label=raw_label, linewidth=0.8, alpha=0.75)
        if proc is not None:
            tp = np.arange(proc.size) * (t[1] - t[0] if t.size > 1 else 1.0)
            tx2, y2 = downsample_for_plot(tp, proc, max_points)
            ax.plot(tx2, y2, label="Processed", linewidth=1.2)
        title = "Angle displacement waveform" if self.current_source_kind == "angle" else ("Acceleration axis waveform" if self.current_source_kind == "axis" else "Signal / displacement waveform")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (deg)" if self.current_source_kind == "angle" else "Value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    def _plot_velocity(self, ax, raw: np.ndarray, proc: Optional[np.ndarray], interval_s: float, max_points: int) -> None:
        raw_v, _ = velocity_and_acceleration(raw, interval_s)
        tv = (np.arange(raw_v.size) + 0.5) * interval_s
        tx, y = downsample_for_plot(tv, raw_v, max_points)
        ax.plot(tx, y, label="Raw velocity", linewidth=0.8, alpha=0.75)
        if proc is not None:
            proc_v, _ = velocity_and_acceleration(proc, interval_s)
            tv2 = (np.arange(proc_v.size) + 0.5) * interval_s
            tx2, y2 = downsample_for_plot(tv2, proc_v, max_points)
            ax.plot(tx2, y2, label="Processed velocity", linewidth=1.2)
        ax.set_title("Velocity estimate between adjacent points")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("deg/s" if self.current_source_kind == "angle" else "unit/s")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    def _plot_acceleration(self, ax, raw: np.ndarray, proc: Optional[np.ndarray], interval_s: float, max_points: int) -> None:
        _, raw_a = velocity_and_acceleration(raw, interval_s)
        ta = (np.arange(raw_a.size) + 1.0) * interval_s
        tx, y = downsample_for_plot(ta, raw_a, max_points)
        ax.plot(tx, y, label="Raw acceleration", linewidth=0.8, alpha=0.75)
        if proc is not None:
            _, proc_a = velocity_and_acceleration(proc, interval_s)
            ta2 = (np.arange(proc_a.size) + 1.0) * interval_s
            tx2, y2 = downsample_for_plot(ta2, proc_a, max_points)
            ax.plot(tx2, y2, label="Processed acceleration", linewidth=1.2)
        ax.set_title("Acceleration estimate from velocity differences")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("deg/s^2" if self.current_source_kind == "angle" else "unit/s^2")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    def export_processed(self) -> None:
        if self.processed_data is None:
            messagebox.showwarning("没有处理后数据", "请先点击“预处理并绘图”。")
            return
        cfg = self.current_config()
        default_name = "processed_sequence.txt"
        if self.loaded_path:
            default_name = self.loaded_path.stem + "_processed.txt"
        path = filedialog.asksaveasfilename(
            title="导出处理后数据",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Text", "*.txt"), ("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        fmt = f"{{:.{cfg.output_precision}f}}"
        try:
            with Path(path).open("w", encoding="utf-8", newline="") as f:
                for v in self.processed_data:
                    f.write(fmt.format(float(v)) + "\n")
            self.log(f"已导出处理后数据: {path}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    def export_angle_data(self) -> None:
        if self.angle_data is None:
            messagebox.showwarning("没有角度位移数据", "请先点击“截取选段 → 角度位移”。")
            return
        cfg = self.current_config()
        default_name = "angle_displacement.txt"
        if self.loaded_path:
            default_name = self.loaded_path.stem + "_angle_displacement.txt"
        path = filedialog.asksaveasfilename(
            title="导出角度位移数据",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Text", "*.txt"), ("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        fmt = f"{{:.{cfg.output_precision}f}}"
        try:
            with Path(path).open("w", encoding="utf-8", newline="") as f:
                for v in self.angle_data:
                    f.write(fmt.format(float(v)) + "\n")
            self.log(f"已导出角度位移数据: {path}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    def save_plot_png(self) -> None:
        path = filedialog.asksaveasfilename(
            title="保存当前图",
            defaultextension=".png",
            initialfile="preprocess_comparison.png",
            filetypes=[("PNG", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=160)
            self.log(f"已保存图像: {path}")
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))

    def save_config(self) -> None:
        try:
            cfg = self.current_config()
            acfg = self.current_angle_config()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        path = filedialog.asksaveasfilename(
            title="保存配置",
            defaultextension=".json",
            initialfile="preprocess_config.json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            data = {"preprocess": asdict(cfg), "angle_conversion": asdict(acfg)}
            Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.log(f"已保存配置: {path}")
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))

    def load_config(self) -> None:
        path = filedialog.askopenfilename(
            title="读取配置",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if "preprocess" in data:
                pdata = data.get("preprocess", {})
                adata = data.get("angle_conversion", {})
            else:
                # 兼容旧版直接保存 PreprocessConfig 的 JSON。
                pdata = data
                adata = {}
            cfg = PreprocessConfig(**{**asdict(PreprocessConfig()), **pdata})
            acfg = AngleConversionConfig(**{**asdict(AngleConversionConfig()), **adata})

            self.interval_var.set(str(cfg.interval_s))
            self.column_var.set(str(cfg.column_index))
            self.scale_var.set(str(cfg.scale_factor))
            self.invalid_enabled_var.set(bool(cfg.invalid_enabled))
            self.min_value_var.set(str(cfg.min_value))
            self.max_value_var.set(str(cfg.max_value))
            self.median_enabled_var.set(bool(cfg.median_enabled))
            self.median_window_var.set(str(cfg.median_window))
            self.lowpass_enabled_var.set(bool(cfg.lowpass_enabled))
            self.lowpass_cutoff_var.set(str(cfg.lowpass_cutoff_hz))
            self.zero_start_var.set(bool(cfg.zero_start))
            self.zero_end_var.set(bool(cfg.zero_end))
            self.plot_max_points_var.set(str(cfg.plot_max_points))
            self.output_precision_var.set(str(cfg.output_precision))

            self.axis_var.set(str(acfg.axis_name))
            self.reference_axis_var.set(str(acfg.reference_axis_name))
            self.angle_method_var.set(str(acfg.method))
            self.g_value_var.set(str(acfg.g_value))
            self.segment_start_var.set(str(acfg.start_index_1based))
            self.segment_end_var.set("" if acfg.end_index_1based == 0 else str(acfg.end_index_1based))
            self.baseline_points_var.set(str(acfg.baseline_points))
            self.angle_scale_var.set(str(acfg.angle_scale_factor))
            self.angle_zero_end_var.set(bool(acfg.angle_zero_end))
            self.angle_reverse_var.set(bool(acfg.angle_reverse))
            self.rotation_axis_source_var.set(str(acfg.rotation_axis_source))
            self.rotation_axis_x_var.set(str(acfg.rotation_axis_x))
            self.rotation_axis_y_var.set(str(acfg.rotation_axis_y))
            self.rotation_axis_z_var.set(str(acfg.rotation_axis_z))

            self.log(f"已读取配置: {path}")
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc))


def main() -> None:
    app = SignalPreprocessApp()
    app.mainloop()


if __name__ == "__main__":
    main()
