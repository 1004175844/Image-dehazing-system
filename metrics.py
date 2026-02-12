"""
无参考图像质量指标（适用于去雾等任务）。
包含清晰度/对比度/熵、以及可选的 NIQE/BRISQUE（需安装 piq, torch）。
"""
import cv2
import numpy as np


METRIC_ORDER = (
    "Tenengrad",
    "Laplacian Var",
    "Entropy",
    "RMS Contrast",
)
METRIC_DISPLAY = {
    "Tenengrad": "特南格拉斯梯度",
    "Laplacian Var": "拉普拉斯方差",
    "Entropy": "熵",
    "RMS Contrast": "RMS 对比度",
}

# 可选：NIQE / BRISQUE（需 piq + torch）
_has_piq = False
try:
    import torch
    import piq
    _has_piq = True
except ImportError:
    pass


def _bgr_to_rgb_tensor_01(bgr):
    """BGR (H,W,3) uint8 -> RGB (1,3,H,W) float [0,1]."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return x


def _compute_brisque(bgr):
    if not _has_piq:
        return None
    try:
        x = _bgr_to_rgb_tensor_01(bgr)
        score = piq.brisque(x, data_range=1.0, reduction="mean")
        return float(score.item())
    except Exception:
        return None


def _compute_niqe(bgr):
    if not _has_piq:
        return None
    try:
        if not hasattr(piq, "niqe"):
            return None
        x = _bgr_to_rgb_tensor_01(bgr)
        score = piq.niqe(x, data_range=1.0, reduction="mean")
        return float(score.item())
    except Exception:
        return None


def _validate_images(input_bgr, output_bgr):
    if input_bgr is None or output_bgr is None:
        raise ValueError("Input and output images must not be None.")
    if input_bgr.size == 0 or output_bgr.size == 0:
        raise ValueError("Input and output images must not be empty.")
    if input_bgr.shape[:2] != output_bgr.shape[:2]:
        raise ValueError("Input and output images must have the same resolution.")


def _to_gray(input_bgr):
    return cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0


def _tenengrad(gray):
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    return float(np.mean(magnitude))


def _laplacian_variance(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap))


def _entropy(gray):
    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    hist = cv2.calcHist([gray_u8], [0], None, [256], [0, 256]).ravel()
    total = float(np.sum(hist))
    if total <= 0:
        return 0.0
    prob = hist / total
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def _rms_contrast(gray):
    return float(np.std(gray))


def calculate_metrics(input_bgr, output_bgr):
    _validate_images(input_bgr, output_bgr)

    gray_in = _to_gray(input_bgr)
    gray_out = _to_gray(output_bgr)

    in_values = {
        "Tenengrad": _tenengrad(gray_in),
        "Laplacian Var": _laplacian_variance(gray_in),
        "Entropy": _entropy(gray_in),
        "RMS Contrast": _rms_contrast(gray_in),
    }
    out_values = {
        "Tenengrad": _tenengrad(gray_out),
        "Laplacian Var": _laplacian_variance(gray_out),
        "Entropy": _entropy(gray_out),
        "RMS Contrast": _rms_contrast(gray_out),
    }

    rows = []
    for name in METRIC_ORDER:
        before = in_values[name]
        after = out_values[name]
        delta = after - before
        ratio = (delta / (abs(before) + 1e-12)) * 100.0
        rows.append(
            {
                "name": METRIC_DISPLAY.get(name, name),
                "input": before,
                "output": after,
                "delta": delta,
                "improve_ratio": ratio,
            }
        )

    result = {"metrics": rows}

    # 可选无参考指标（BRISQUE 越低越好；NIQE 若存在则越低越好）
    brisque_in = _compute_brisque(input_bgr)
    brisque_out = _compute_brisque(output_bgr)
    niqe_in = _compute_niqe(input_bgr)
    niqe_out = _compute_niqe(output_bgr)
    if brisque_in is not None:
        result["brisque_input"] = brisque_in
        result["brisque_output"] = brisque_out
    if niqe_in is not None:
        result["niqe_input"] = niqe_in
        result["niqe_output"] = niqe_out

    return result
