# -*- coding: utf-8 -*-
"""
Preprocessing pipeline for the Arabic Handwriting Recognition model.

*** VERBATIM MIRROR — DO NOT "IMPROVE" THIS FILE ***
This is a copy of `beginer level copy letters/handwriting_api/preprocessing.py`,
which is itself a step-for-step copy of the training notebook (Section 3
preprocessing -> augmentation's center_and_normalize -> feature extraction).
The model was trained on exactly these numbers; any change here silently
shifts the input distribution and the classifier starts marking correct
letters wrong. If the notebook is retrained, re-copy the file — don't patch it.

  1. Build raw strokes from (x, y) points.
  2. Normalize the whole sample to [0, 1] using its bounding box, then flip Y
     (InkML/canvas Y grows downward, the model was trained on Y growing upward).
  3. Sort strokes by length (number of points), descending.
  4. Resample: the longest stroke -> 60 points, every other stroke -> 5 points
     (arc-length resampling).
  5. Center strokes on their center of mass, then scale by max absolute value
     (center_and_normalize).
  6. Extract per-point features: x, y, dx, dy, angle, pen_state
     (pen_state = 1 for all points except the last point of each stroke, which is 0).
  7. Pad/truncate the flattened point sequence to MAX_LEN.

No filename-based left/right flip is applied here (that rule only applied to a
naming quirk in the original dataset files, not to live drawings).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def resample_stroke(stroke: np.ndarray, n: int) -> np.ndarray:
    """Arc-length resampling of a single stroke to exactly n points."""
    stroke = np.asarray(stroke, dtype=np.float32)
    if len(stroke) < 2:
        return np.tile(stroke[0], (n, 1)) if len(stroke) > 0 else np.zeros((n, 2), dtype=np.float32)

    dist = np.cumsum(np.sqrt(np.sum(np.diff(stroke, axis=0) ** 2, axis=1)))
    dist = np.insert(dist, 0, 0)

    if np.all(dist == 0):
        return np.tile(stroke[0], (n, 1))

    dist_u, idx = np.unique(dist, return_index=True)
    stroke = stroke[idx]
    if len(dist_u) < 2:
        return np.tile(stroke[0], (n, 1))

    f = interp1d(dist_u, stroke, axis=0)
    return f(np.linspace(0, dist_u[-1], n)).astype(np.float32)


def preprocess_strokes(strokes_list: list[list[list[float]]], max_len: int, n_features: int = 6) -> np.ndarray | None:
    """
    Turn raw strokes (as received from the client) into a single (1, max_len, n_features)
    array ready to be fed to model.predict().

    strokes_list: list of strokes, each stroke a list of [x, y] pairs, in any consistent
                  pixel/point unit (absolute scale does not matter, only relative shape,
                  because step 2 normalizes by the sample's own bounding box).

    Returns None if the input does not contain a usable drawing.
    """
    raw_strokes = []
    for s in strokes_list:
        pts = np.array(s, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[1] >= 2 and len(pts) >= 2:
            raw_strokes.append(pts[:, :2])

    if not raw_strokes:
        return None

    # Step 2: normalize to [0, 1] using the bounding box of the whole sample, then y-flip
    all_pts = np.vstack(raw_strokes)
    s_min = all_pts.min(axis=0)
    s_max = all_pts.max(axis=0)
    scale = np.max(s_max - s_min) + 1e-6

    normed = []
    for s in raw_strokes:
        s = s.copy()
        s[:, 0] = (s[:, 0] - s_min[0]) / scale
        s[:, 1] = (s[:, 1] - s_min[1]) / scale
        s[:, 1] = 1 - s[:, 1]  # y-flip
        normed.append(s)

    # Step 3: sort strokes by length, descending
    normed.sort(key=len, reverse=True)

    # Step 4: resample - longest stroke -> 60 pts, others -> 5 pts
    resampled = []
    for i, s in enumerate(normed):
        n_pts = 60 if i == 0 else 5
        resampled.append(resample_stroke(s, n_pts))

    # Step 5: center_and_normalize (center of mass to origin, scale by max abs value)
    all_r = np.vstack(resampled)
    cx, cy = all_r[:, 0].mean(), all_r[:, 1].mean()
    centered = [s - np.array([cx, cy], dtype=np.float32) for s in resampled]
    max_abs = np.abs(np.vstack(centered)).max()
    if max_abs > 0:
        centered = [s / max_abs for s in centered]

    # Step 6: extract_features (x, y, dx, dy, angle, pen_state)
    all_x, all_y, all_pen = [], [], []
    for stroke in centered:
        n = len(stroke)
        all_x.extend(stroke[:, 0].tolist())
        all_y.extend(stroke[:, 1].tolist())
        pen = [1.0] * n
        pen[-1] = 0.0
        all_pen.extend(pen)

    if len(all_x) < 2:
        return None

    x = np.array(all_x, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    pen = np.array(all_pen, dtype=np.float32)
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    angle = np.arctan2(dy, dx).astype(np.float32)
    features = np.stack([x, y, dx, dy, angle, pen], axis=1)

    # Step 7: pad / truncate to max_len
    processed = np.zeros((max_len, n_features), dtype=np.float32)
    length = min(len(features), max_len)
    processed[:length] = features[:length]

    return np.expand_dims(processed, axis=0)
