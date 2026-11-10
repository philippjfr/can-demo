"""Deterministic synthetic dataset: top-down views of cans on a conveyor plate.

Class 0 (normal): cans standing upright, seen from above as discs.
Class 1 (fallen_can): one can lies on its side across the plate, seen as an elongated body.

Every image is drawn from its own RNG seeded with ``(seed, index)``, so the result does not depend
on generation order. Rendering and blur use NumPy only to keep the pixels, and therefore the dataset
hash, independent of the Pillow version.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ("normal", "fallen_can")

_PALETTE = np.array(
    [
        [0.78, 0.79, 0.81],  # silver
        [0.72, 0.18, 0.16],  # red
        [0.16, 0.35, 0.68],  # blue
        [0.80, 0.66, 0.25],  # gold
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class DataConfig:
    n_images: int = 2000
    size: int = 128
    seed: int = 7
    fallen_fraction: float = 0.5
    # Probability that a normal image shows two touching cans (peanut-shaped blob).
    hard_negative_prob: float = 0.4
    gain_range: tuple[float, float] = (0.7, 1.3)
    noise_range: tuple[float, float] = (0.02, 0.07)
    blur_range: tuple[float, float] = (0.4, 1.2)

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, d: dict) -> DataConfig:
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def shifted_config(config: DataConfig, seed: int = 99, n_images: int = 200) -> DataConfig:
    """A deliberately degraded camera: darker, noisier, blurrier. Used for drift monitoring."""
    return DataConfig(
        n_images=n_images,
        size=config.size,
        seed=seed,
        fallen_fraction=config.fallen_fraction,
        hard_negative_prob=config.hard_negative_prob,
        gain_range=(0.4, 0.75),
        noise_range=(0.07, 0.14),
        blur_range=(1.3, 2.2),
    )


@dataclass
class Dataset:
    images: np.ndarray  # (N, H, W, 3) uint8
    labels: np.ndarray  # (N,) int64
    config: DataConfig
    _hash: str | None = field(default=None, repr=False)

    @property
    def sha256(self) -> str:
        if self._hash is None:
            self._hash = dataset_hash(self.images, self.labels, self.config)
        return self._hash


def dataset_hash(images: np.ndarray, labels: np.ndarray, config: DataConfig) -> str:
    """Content hash over generator config, labels and pixel bytes."""
    h = hashlib.sha256()
    h.update(json.dumps(config.to_dict(), sort_keys=True).encode())
    h.update(np.ascontiguousarray(labels, dtype="<i8").tobytes())
    h.update(str(images.shape).encode())
    h.update(np.ascontiguousarray(images).tobytes())
    return h.hexdigest()


def _coverage(signed_dist: np.ndarray, soft: float = 1.0) -> np.ndarray:
    """Anti-aliased coverage from a signed distance (negative inside)."""
    return np.clip(0.5 - signed_dist / soft, 0.0, 1.0)[..., None]


def _over(img: np.ndarray, alpha: np.ndarray, color: np.ndarray | float) -> None:
    img *= 1.0 - alpha
    img += alpha * np.asarray(color, dtype=np.float64)


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.05:
        return img
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()
    out = img
    for axis in (0, 1):
        pad = [(0, 0)] * img.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="reflect")
        acc = np.zeros_like(out)
        n = out.shape[axis]
        for i, w in enumerate(k):
            sl = [slice(None)] * img.ndim
            sl[axis] = slice(i, i + n)
            acc += w * padded[tuple(sl)]
        out = acc
    return out


def _window(img, yy, xx, cy, cx, reach):
    """Views of the image and coordinate grids around (cy, cx); drawing stays local and fast."""
    s = img.shape[0]
    y0, y1 = max(0, int(cy - reach)), min(s, int(cy + reach) + 1)
    x0, x1 = max(0, int(cx - reach)), min(s, int(cx + reach) + 1)
    return img[y0:y1, x0:x1], yy[y0:y1, x0:x1], xx[y0:y1, x0:x1]


def _draw_can_top(img, yy, xx, cy, cx, r, color):
    """Upright can seen from above: rim ring, flat lid, pull tab."""
    img, yy, xx = _window(img, yy, xx, cy, cx, r + 3)
    if img.size == 0:
        return
    d = np.hypot(yy - cy, xx - cx)
    _over(img, _coverage(d - r), color * 0.62)
    _over(img, _coverage(d - (r - 1.8)), color * 0.95)
    _over(img, _coverage(d - 0.62 * r), color * 1.05)
    tab = np.hypot((yy - cy) / 0.35, (xx - cx - 0.15 * r) / 0.55) - 0.30 * r
    _over(img, _coverage(tab), color * 0.55)


def _draw_can_side(img, yy, xx, cy, cx, half_len, r, angle, color):
    """Fallen can: rounded box with cylinder shading, end rims and a label band."""
    img, yy, xx = _window(img, yy, xx, cy, cx, half_len + r + 3)
    if img.size == 0:
        return
    ux, uy = np.cos(angle), np.sin(angle)
    px, py = xx - cx, yy - cy
    along = px * ux + py * uy
    across = -px * uy + py * ux
    rc = 0.28 * r
    qx = np.abs(along) - (half_len - rc)
    qy = np.abs(across) - (r - rc)
    sdf = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rc
    _over(img, _coverage(sdf), color * 0.6)
    shade = 0.75 + 0.45 * np.cos(np.clip(across / r, -1, 1) * 1.2)
    body = _coverage(sdf + 1.2)
    _over(img, body, color * shade[..., None])
    for sign in (-1, 1):
        rim = np.abs(along - sign * (half_len - 0.12 * r)) - 0.7
        _over(img, _coverage(rim) * (np.abs(across) < r)[..., None], color * 0.5)
    band = np.abs(along - 0.1 * half_len) - 0.28 * half_len
    _over(img, _coverage(band) * body, np.clip(color[::-1] * 0.9, 0, 1))


def _render(index: int, cfg: DataConfig, label: int, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng([cfg.seed, index, 1])
    s = cfg.size
    scale = s / 128.0
    img = np.empty((s, s, 3), dtype=np.float64)

    # Conveyor belt with streaks and a lighter metal plate in the middle.
    belt = rng.uniform(0.18, 0.30)
    img[:] = belt * np.array([1.0, 1.02, 1.06], dtype=np.float64)
    streaks = rng.uniform(-0.03, 0.03, size=(s, 1, 1)).astype(np.float64)
    img += streaks
    plate_top = s * rng.uniform(0.18, 0.26)
    plate_bot = s * rng.uniform(0.74, 0.82)
    half_plate = (plate_bot - plate_top) / 2
    plate = np.clip(0.5 - (np.abs(yy - (plate_top + plate_bot) / 2) - half_plate), 0, 1)
    _over(img, plate[..., None], rng.uniform(0.42, 0.58) * np.array([1, 1, 1.03], dtype=np.float64))
    for bx in np.linspace(0.08, 0.92, 6) * s:  # bolts
        for by in (plate_top + 3 * scale, plate_bot - 3 * scale):
            _over(img, _coverage(np.hypot(yy - by, xx - bx) - 1.6 * scale), 0.3)

    r = rng.uniform(9.0, 11.5) * scale
    rows = (plate_top + 0.28 * (plate_bot - plate_top), plate_top + 0.72 * (plate_bot - plate_top))
    cols = np.linspace(0.14, 0.86, 4) * s
    slots = [(ry, cx) for ry in rows for cx in cols]
    present = rng.random(len(slots)) < 0.88
    present[rng.integers(len(slots))] = True
    colors = _PALETTE[rng.integers(len(_PALETTE), size=len(slots))]
    colors = colors * rng.uniform(0.9, 1.1, size=(len(slots), 1)).astype(np.float64)

    fallen_slot = int(rng.integers(len(slots))) if label == 1 else -1
    if label == 0 and rng.random() < cfg.hard_negative_prob:
        # Two upright cans touching or overlapping: elongated blob but still "normal".
        i = int(rng.integers(len(slots)))
        j = i + 1 if (i % 4) < 3 else i - 1
        present[i] = present[j] = True
        slots[j] = (slots[i][0] + rng.uniform(-2, 2) * scale, slots[i][1] + 1.75 * r)
        slots[j] = (slots[j][0], float(np.clip(slots[j][1], r, s - r)))

    for k, (cy, cx) in enumerate(slots):
        if k == fallen_slot or not present[k]:
            continue
        jy, jx = rng.normal(0, 1.8 * scale, size=2)
        _draw_can_top(img, yy, xx, cy + jy, cx + jx, r * rng.uniform(0.95, 1.05), colors[k])

    if label == 1:
        cy, cx = slots[fallen_slot]
        cy += rng.normal(0, 2.5 * scale)
        cx += rng.normal(0, 3.0 * scale)
        half_len = r * rng.uniform(2.2, 3.1)
        across_plate = np.pi / 2 + rng.uniform(-0.6, 0.6)
        angle = across_plate if rng.random() < 0.75 else rng.uniform(0, np.pi)
        radius = r * rng.uniform(0.9, 1.0)
        _draw_can_side(img, yy, xx, cy, cx, half_len, radius, angle, colors[fallen_slot])

    # Lighting: global gain, linear gradient, optional specular highlight.
    gain = rng.uniform(*cfg.gain_range)
    theta = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(theta) * (xx / s - 0.5) + np.sin(theta) * (yy / s - 0.5)) * rng.uniform(0.0, 0.6)
    light = gain * (1.0 + grad)
    if rng.random() < 0.35:
        hy, hx = rng.uniform(0.2, 0.8, size=2) * s
        spot = np.exp(-((yy - hy) ** 2 + (xx - hx) ** 2) / (2 * (0.15 * s) ** 2))
        light += gain * rng.uniform(0.1, 0.3) * spot
    img *= light[..., None].astype(np.float64)

    img = _gaussian_blur(img, rng.uniform(*cfg.blur_range))
    img += rng.normal(0, rng.uniform(*cfg.noise_range), size=img.shape).astype(np.float64)
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def generate_dataset(config: DataConfig | None = None) -> Dataset:
    """Generate the full dataset in memory. About 2,000 images take a few seconds on CPU."""
    cfg = config or DataConfig()
    n_fallen = int(round(cfg.n_images * cfg.fallen_fraction))
    order = np.random.default_rng([cfg.seed, 0]).permutation(cfg.n_images)
    labels = (order < n_fallen).astype(np.int64)
    yy, xx = np.mgrid[0 : cfg.size, 0 : cfg.size].astype(np.float64)
    images = np.stack([_render(i, cfg, int(labels[i]), yy, xx) for i in range(cfg.n_images)])
    return Dataset(images=images, labels=labels, config=cfg)


@dataclass(frozen=True)
class Splits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_splits(
    labels: np.ndarray, seed: int = 7, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> Splits:
    """Stratified train/val/test indices."""
    rng = np.random.default_rng([seed, 2])
    parts: list[list[np.ndarray]] = [[], [], []]
    for cls in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        n_train = int(round(fractions[0] * len(idx)))
        n_val = int(round(fractions[1] * len(idx)))
        parts[0].append(idx[:n_train])
        parts[1].append(idx[n_train : n_train + n_val])
        parts[2].append(idx[n_train + n_val :])
    return Splits(*(np.sort(np.concatenate(p)) for p in parts))


def to_nchw_float(images: np.ndarray) -> np.ndarray:
    """uint8 (N, H, W, 3) to float32 (N, 3, H, W) in [0, 1]."""
    return np.ascontiguousarray(images.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0


def write_preview(dataset: Dataset, path: str, per_class: int = 8) -> str:
    """Write a contact sheet with normal images on the top row and fallen cans below."""
    rows = []
    for cls in (0, 1):
        idx = np.flatnonzero(dataset.labels == cls)[:per_class]
        rows.append(np.concatenate(list(dataset.images[idx]), axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)
    return path


def main() -> None:
    """``python -m candetect.data``: generate the default dataset and write its hash and preview."""
    out = Path("data")
    out.mkdir(exist_ok=True)
    started = time.time()
    ds = generate_dataset()
    elapsed = time.time() - started
    print(f"generated {len(ds.images)} images {ds.images.shape[1:]} in {elapsed:.1f}s")
    print(f"fallen share: {ds.labels.mean():.2f}")
    print(f"dataset sha256: {ds.sha256}")
    (out / "dataset.sha256").write_text(ds.sha256 + "\n")
    print(f"preview: {write_preview(ds, str(out / 'preview.png'))}")


if __name__ == "__main__":
    main()
