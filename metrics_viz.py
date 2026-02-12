import os

import numpy as np
from matplotlib.figure import Figure


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
    ax.bar(x - width / 2, input_values, width=width, label="Input", color="#94a3b8")
    ax.bar(x + width / 2, output_values, width=width, label="Dehazed", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_title("Metric Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
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
    ax.plot(angles, in_norm, color="#64748b", linewidth=2, label="Input")
    ax.fill(angles, in_norm, color="#cbd5e1", alpha=0.35)
    ax.plot(angles, out_norm, color="#1d4ed8", linewidth=2, label="Dehazed")
    ax.fill(angles, out_norm, color="#93c5fd", alpha=0.35)
    ax.set_ylim(0, 1.05)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels=names)
    ax.set_title("Normalized Metric Radar")
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.15), frameon=False)
    fig.tight_layout()
    return fig


def save_metric_figures(metrics_result, output_dir, prefix="metrics"):
    os.makedirs(output_dir, exist_ok=True)
    bar_path = os.path.join(output_dir, f"{prefix}_bar.png")
    radar_path = os.path.join(output_dir, f"{prefix}_radar.png")

    bar_fig = build_bar_figure(metrics_result)
    radar_fig = build_radar_figure(metrics_result)
    bar_fig.savefig(bar_path, dpi=300, bbox_inches="tight")
    radar_fig.savefig(radar_path, dpi=300, bbox_inches="tight")
    bar_fig.clear()
    radar_fig.clear()
    return bar_path, radar_path

