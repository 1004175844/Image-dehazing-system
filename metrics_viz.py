"""
无参考去雾评价可视化：直方图与指标图，风格简洁适合论文。
"""
import os

import cv2
import numpy as np
import matplotlib
from matplotlib.figure import Figure

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 论文用色：输入 / 去雾后
COLOR_IN = "#64748b"
COLOR_OUT = "#2563eb"
BINS = 64


def _hist_1d(values, bins, range_=(0, 1)):
    h, _ = np.histogram(values.ravel(), bins=bins, range=range_)
    return h.astype(np.float64) / (h.sum() + 1e-12)


def build_rgb_hist_figure(input_bgr, output_bgr):
    """输入 vs 去雾后 RGB 三通道亮度分布（归一化直方图）。"""
    fig = Figure(figsize=(6, 3.2), dpi=110)
    channels = [
        ("R", cv2.COLOR_BGR2RGB, 0),
        ("G", cv2.COLOR_BGR2RGB, 1),
        ("B", cv2.COLOR_BGR2RGB, 2),
    ]
    for i, (label, code, c) in enumerate(channels):
        ax = fig.add_subplot(1, 3, i + 1)
        rgb_in = cv2.cvtColor(input_bgr, code)
        rgb_out = cv2.cvtColor(output_bgr, code)
        h_in = _hist_1d(rgb_in[:, :, c].astype(np.float32) / 255.0, BINS)
        h_out = _hist_1d(rgb_out[:, :, c].astype(np.float32) / 255.0, BINS)
        x = np.linspace(0, 1, BINS, endpoint=False) + 0.5 / BINS
        ax.plot(x, h_in, color=COLOR_IN, linewidth=1.5, label="输入")
        ax.plot(x, h_out, color=COLOR_OUT, linewidth=1.5, label="去雾后")
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, None)
        ax.legend(frameon=False, fontsize=9)
        ax.tick_params(axis="both", labelsize=9)
    fig.suptitle("RGB 亮度分布", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def build_gradient_hist_figure(input_bgr, output_bgr):
    """输入 vs 去雾后 梯度幅值分布（归一化直方图）。"""
    def grad_mag(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy).ravel()

    mag_in = grad_mag(input_bgr)
    mag_out = grad_mag(output_bgr)
    max_m = max(mag_in.max(), mag_out.max()) or 1.0
    h_in = _hist_1d(mag_in, BINS, (0, max_m))
    h_out = _hist_1d(mag_out, BINS, (0, max_m))
    x = np.linspace(0, max_m, BINS, endpoint=False) + 0.5 * max_m / BINS

    fig = Figure(figsize=(6, 3.2), dpi=110)
    ax = fig.add_subplot(111)
    ax.plot(x, h_in, color=COLOR_IN, linewidth=1.5, label="输入")
    ax.plot(x, h_out, color=COLOR_OUT, linewidth=1.5, label="去雾后")
    ax.set_title("梯度幅值分布", fontsize=12)
    ax.set_xlabel("梯度幅值", fontsize=10)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    return fig


def build_saturation_hist_figure(input_bgr, output_bgr):
    """输入 vs 去雾后 饱和度分布（HSV S 通道，归一化直方图）。"""
    hsv_in = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2HSV)
    hsv_out = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2HSV)
    s_in = hsv_in[:, :, 1].astype(np.float32) / 255.0
    s_out = hsv_out[:, :, 1].astype(np.float32) / 255.0
    h_in = _hist_1d(s_in, BINS)
    h_out = _hist_1d(s_out, BINS)
    x = np.linspace(0, 1, BINS, endpoint=False) + 0.5 / BINS

    fig = Figure(figsize=(6, 3.2), dpi=110)
    ax = fig.add_subplot(111)
    ax.plot(x, h_in, color=COLOR_IN, linewidth=1.5, label="输入")
    ax.plot(x, h_out, color=COLOR_OUT, linewidth=1.5, label="去雾后")
    ax.set_title("饱和度分布", fontsize=12)
    ax.set_xlabel("饱和度", fontsize=10)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    return fig


def _extract_series(metrics_result):
    rows = metrics_result.get("metrics", [])
    names = [row["name"] for row in rows]
    input_values = np.array([row["input"] for row in rows], dtype=np.float64)
    output_values = np.array([row["output"] for row in rows], dtype=np.float64)
    return names, input_values, output_values


def build_bar_figure(metrics_result):
    names, input_values, output_values = _extract_series(metrics_result)
    x = np.arange(len(names))
    width = 0.36

    fig = Figure(figsize=(6.6, 4.2), dpi=110)
    ax = fig.add_subplot(111)
    ax.bar(x - width / 2, input_values, width=width, label="输入", color=COLOR_IN)
    ax.bar(x + width / 2, output_values, width=width, label="去雾后", color=COLOR_OUT)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, fontsize=9)
    ax.set_title("无参考指标对比", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    return fig


def build_radar_figure(metrics_result):
    names, input_values, output_values = _extract_series(metrics_result)
    max_values = np.maximum(input_values, output_values)
    max_values[max_values <= 0] = 1.0
    in_norm = input_values / max_values
    out_norm = output_values / max_values

    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    in_norm = np.concatenate([in_norm, [in_norm[0]]])
    out_norm = np.concatenate([out_norm, [out_norm[0]]])

    fig = Figure(figsize=(6.6, 4.2), dpi=110)
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, in_norm, color=COLOR_IN, linewidth=2, label="输入")
    ax.fill(angles, in_norm, color=COLOR_IN, alpha=0.2)
    ax.plot(angles, out_norm, color=COLOR_OUT, linewidth=2, label="去雾后")
    ax.fill(angles, out_norm, color=COLOR_OUT, alpha=0.2)
    ax.set_ylim(0, 1.05)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels=names, fontsize=9)
    ax.set_title("指标雷达图", fontsize=12, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.12), frameon=False, fontsize=10)
    fig.tight_layout()
    return fig


def save_metric_figures(metrics_result, output_dir, prefix="metrics", input_bgr=None, output_bgr=None):
    """导出所有图表（含直方图需传入 input_bgr, output_bgr）。"""
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    if input_bgr is not None and output_bgr is not None:
        for name, build_fn in [
            ("rgb_hist", build_rgb_hist_figure),
            ("gradient_hist", build_gradient_hist_figure),
            ("saturation_hist", build_saturation_hist_figure),
        ]:
            fig = build_fn(input_bgr, output_bgr)
            p = os.path.join(output_dir, f"{prefix}_{name}.png")
            fig.savefig(p, dpi=300, bbox_inches="tight")
            paths.append(p)
            fig.clear()

    for name, build_fn in [
        ("bar", build_bar_figure),
        ("radar", build_radar_figure),
    ]:
        fig = build_fn(metrics_result)
        p = os.path.join(output_dir, f"{prefix}_{name}.png")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
        fig.clear()

    return paths
