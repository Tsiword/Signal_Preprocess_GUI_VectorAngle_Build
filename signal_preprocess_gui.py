# -*- coding: utf-8 -*-
"""
Signal Preprocess GUI
---------------------
独立数据预处理软件：用于导入采集到的一列位移/角度数据，进行异常点插值、中值滤波、低通滤波、起点归零、终点回零，并绘图比较处理前后的波形。

依赖：numpy, matplotlib
打包：PyInstaller --onefile --windowed
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
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

APP_TITLE = "采集信号数据预处理与波形对比工具"
APP_VERSION = "1.0.0"

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
    column_index: str = ""  # 从 1 开始；空表示自动解析所有单列数字
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
    # 兼容逗号、分号、空白、Tab。
    return [p for p in re.split(r"[,;\s]+", line.strip()) if p]


def parse_sequence_text(text: str, column_index: str = "") -> np.ndarray:
    """从文本框内容解析数字。column_index 为空时会读取所有数值；否则读取指定列。"""
    lines = []
    for line in text.splitlines():
        # 支持 # 注释。
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
                # 允许表头/短行跳过，但如果整文件最终无数据会报错。
                continue
            v = parse_float_token(tokens[col])
            if v is not None:
                values.append(v)

    if not values:
        raise ValueError("未解析到任何有效数字。请检查文件格式或列号设置。")
    return np.asarray(values, dtype=float)


def read_sequence_file(path: str, column_index: str = "") -> np.ndarray:
    # 尝试多种常见编码。
    encodings = ["utf-8-sig", "utf-8", "gbk", "cp936"]
    last_exc: Optional[Exception] = None
    for enc in encodings:
        try:
            text = Path(path).read_text(encoding=enc)
            return parse_sequence_text(text, column_index=column_index)
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
    raise ValueError(f"读取文件失败，可能不是文本文件: {last_exc}")


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
    # 纯 numpy rolling median 不方便，这里用简单循环。对 9 万点、窗口 5/7 足够快。
    for i in range(x.size):
        lo = max(0, i - half)
        hi = min(x.size, i + half + 1)
        out[i] = float(np.median(x[lo:hi]))
    return out


def lowpass_first_order_zero_phase(values: np.ndarray, interval_s: float, cutoff_hz: float) -> np.ndarray:
    """
    一阶低通 + 反向再滤一次，减少相位滞后。
    与摆台控制 GUI 中集成的预处理逻辑保持一致，避免额外依赖 scipy。
    """
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
        # 去除线性漂移，使最后一个点回到 0。
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


class SignalPreprocessApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_TITLE} v{APP_VERSION}")
        self.geometry("1320x820")
        self.minsize(1100, 720)

        self.raw_data: Optional[np.ndarray] = None
        self.processed_data: Optional[np.ndarray] = None
        self.loaded_path: Optional[Path] = None
        self.last_report: Optional[PreprocessReport] = None

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

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        left = ttk.Frame(root)
        left.pack(side="left", fill="y", padx=(0, 8))
        right = ttk.Frame(root)
        right.pack(side="right", fill="both", expand=True)

        self._build_file_frame(left)
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
        ttk.Button(frame, text="恢复示例", command=self.restore_demo_data).grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(frame, text="清空", command=self.clear_data).grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        ttk.Label(frame, text="列号").grid(row=2, column=0, sticky="e", padx=4, pady=3)
        ttk.Entry(frame, textvariable=self.column_var, width=10).grid(row=2, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(frame, text="空=自动读取单列；表格可填 1/2/3...").grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

    def _build_param_frame(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="预处理参数", padding=8)
        frame.pack(fill="x", pady=(0, 8))

        self._entry_row(frame, 0, "时间间隔", self.interval_var, 10, "s")
        self._entry_row(frame, 1, "缩放系数", self.scale_var, 10, "导出前乘以该系数")

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
        self.log_text = tk.Text(frame, height=18, width=48, wrap="word", state="disabled")
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

        self.figure = Figure(figsize=(8.5, 6.2), dpi=100)
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
            # 触发验证
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

    def log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def set_data(self, data: np.ndarray, source: str = "") -> None:
        self.raw_data = np.asarray(data, dtype=float)
        self.processed_data = None
        self.last_report = None
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
            data = read_sequence_file(path, cfg.column_index)
            self.loaded_path = Path(path)
            self.set_data(data, source=str(self.loaded_path.name))
        except Exception as exc:
            messagebox.showerror("导入失败", str(exc))

    def restore_demo_data(self) -> None:
        try:
            data = parse_sequence_text(DEFAULT_DEMO)
            self.loaded_path = None
            self.set_data(data, source="内置示例")
        except Exception as exc:
            messagebox.showerror("错误", str(exc))

    def clear_data(self) -> None:
        self.raw_data = None
        self.processed_data = None
        self.last_report = None
        self.figure.clear()
        self.canvas.draw_idle()
        self.clear_log()
        self.log("已清空数据。")

    def open_text_editor(self) -> None:
        win = tk.Toplevel(self)
        win.title("粘贴/编辑数据")
        win.geometry("720x560")
        ttk.Label(win, text="可粘贴一列数字；支持逗号、空格、换行分隔；# 后为注释。", padding=6).pack(fill="x")
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
                self.set_data(data, source="文本框")
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

    def update_plot(self) -> None:
        try:
            cfg = self.current_config()
        except Exception:
            cfg = PreprocessConfig()
        self.figure.clear()
        if self.raw_data is None:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "请先导入数据", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        raw = np.asarray(self.raw_data, dtype=float)
        proc = np.asarray(self.processed_data, dtype=float) if self.processed_data is not None else None
        mode = self.plot_mode_var.get()
        max_points = max(100, int(cfg.plot_max_points))
        t = np.arange(raw.size) * cfg.interval_s

        if mode == "位移+速度+加速度":
            axes = [self.figure.add_subplot(311), self.figure.add_subplot(312), self.figure.add_subplot(313)]
            self._plot_displacement(axes[0], t, raw, proc, max_points)
            self._plot_velocity(axes[1], raw, proc, cfg.interval_s, max_points)
            self._plot_acceleration(axes[2], raw, proc, cfg.interval_s, max_points)
        elif mode == "仅位移":
            ax = self.figure.add_subplot(111)
            self._plot_displacement(ax, t, raw, proc, max_points)
        elif mode == "仅速度":
            ax = self.figure.add_subplot(111)
            self._plot_velocity(ax, raw, proc, cfg.interval_s, max_points)
        else:
            ax = self.figure.add_subplot(111)
            self._plot_acceleration(ax, raw, proc, cfg.interval_s, max_points)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _plot_displacement(self, ax, t: np.ndarray, raw: np.ndarray, proc: Optional[np.ndarray], max_points: int) -> None:
        tx, y = downsample_for_plot(t, raw, max_points)
        ax.plot(tx, y, label="原始", linewidth=0.8, alpha=0.75)
        if proc is not None:
            tp = np.arange(proc.size) * (t[1] - t[0] if t.size > 1 else 1.0)
            tx2, y2 = downsample_for_plot(tp, proc, max_points)
            ax.plot(tx2, y2, label="处理后", linewidth=1.2)
        ax.set_title("位移/信号波形")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("数值")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    def _plot_velocity(self, ax, raw: np.ndarray, proc: Optional[np.ndarray], interval_s: float, max_points: int) -> None:
        raw_v, _ = velocity_and_acceleration(raw, interval_s)
        tv = (np.arange(raw_v.size) + 0.5) * interval_s
        tx, y = downsample_for_plot(tv, raw_v, max_points)
        ax.plot(tx, y, label="原始速度", linewidth=0.8, alpha=0.75)
        if proc is not None:
            proc_v, _ = velocity_and_acceleration(proc, interval_s)
            tv2 = (np.arange(proc_v.size) + 0.5) * interval_s
            tx2, y2 = downsample_for_plot(tv2, proc_v, max_points)
            ax.plot(tx2, y2, label="处理后速度", linewidth=1.2)
        ax.set_title("相邻点速度估计")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("单位/s")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    def _plot_acceleration(self, ax, raw: np.ndarray, proc: Optional[np.ndarray], interval_s: float, max_points: int) -> None:
        _, raw_a = velocity_and_acceleration(raw, interval_s)
        ta = (np.arange(raw_a.size) + 1.0) * interval_s
        tx, y = downsample_for_plot(ta, raw_a, max_points)
        ax.plot(tx, y, label="原始加速度", linewidth=0.8, alpha=0.75)
        if proc is not None:
            _, proc_a = velocity_and_acceleration(proc, interval_s)
            ta2 = (np.arange(proc_a.size) + 1.0) * interval_s
            tx2, y2 = downsample_for_plot(ta2, proc_a, max_points)
            ax.plot(tx2, y2, label="处理后加速度", linewidth=1.2)
        ax.set_title("相邻速度变化/加速度估计")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("单位/s²")
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
            Path(path).write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
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
            cfg = PreprocessConfig(**{**asdict(PreprocessConfig()), **data})
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
            self.log(f"已读取配置: {path}")
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc))


def main() -> None:
    app = SignalPreprocessApp()
    app.mainloop()


if __name__ == "__main__":
    main()
