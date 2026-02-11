import cv2
import numpy as np


def _dark_channel(image, size):
    min_rgb = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(min_rgb, kernel)


def _estimate_atmospheric_light(image, dark, top_percent=0.001):
    h, w = dark.shape
    count = max(1, int(h * w * top_percent))
    flat_dark = dark.reshape(-1)
    indices = np.argsort(flat_dark)[::-1][:count]
    flat_img = image.reshape(-1, 3)
    return np.mean(flat_img[indices], axis=0)


def _estimate_transmission(image, atmospheric, size, omega):
    normed = image / (atmospheric.reshape(1, 1, 3) + 1e-6)
    dark = _dark_channel(normed, size)
    return 1.0 - omega * dark


def _guided_filter(guide, src, radius, eps):
    ksize = (radius * 2 + 1, radius * 2 + 1)
    mean_guide = cv2.boxFilter(guide, -1, ksize)
    mean_src = cv2.boxFilter(src, -1, ksize)
    corr_guide = cv2.boxFilter(guide * guide, -1, ksize)
    corr_guide_src = cv2.boxFilter(guide * src, -1, ksize)
    var_guide = corr_guide - mean_guide * mean_guide
    cov_guide_src = corr_guide_src - mean_guide * mean_src
    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide
    mean_a = cv2.boxFilter(a, -1, ksize)
    mean_b = cv2.boxFilter(b, -1, ksize)
    return mean_a * guide + mean_b


def _refine_transmission(image, transmission, radius, eps):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    guide = gray.astype(np.float32) / 255.0
    return _guided_filter(guide, transmission, radius, eps)


def _recover_radiance(image, transmission, atmospheric, t0):
    transmission = np.maximum(transmission, t0)
    recovered = (image - atmospheric) / transmission[..., None] + atmospheric
    return np.clip(recovered, 0.0, 1.0)


def dehaze_bgr(
    image_bgr,
    *,
    omega=0.95,
    t0=0.1,
    dark_channel_size=15,
    guided_radius=40,
    guided_eps=1e-3,
    top_percent=0.001,
):
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image.")

    image = image_bgr.astype(np.float32) / 255.0
    dark = _dark_channel(image, dark_channel_size)
    atmospheric = _estimate_atmospheric_light(image, dark, top_percent=top_percent)
    transmission = _estimate_transmission(
        image, atmospheric, dark_channel_size, omega
    )
    transmission = _refine_transmission(image, transmission, guided_radius, guided_eps)
    recovered = _recover_radiance(image, transmission, atmospheric, t0)
    return (recovered * 255).astype(np.uint8)
