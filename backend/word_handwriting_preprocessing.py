# -*- coding: utf-8 -*-
"""
Feature pipeline for the ADAB word-recognition Conformer (المستوى المتوسط).

*** VERBATIM MIRROR — DO NOT "IMPROVE" THIS FILE ***
Step-for-step copy of Section 3 of `adab_model/adab_conformer_training.ipynb`
(preprocess_strokes -> segment_to_allographs -> extract_allograph_features).
The Conformer was trained on exactly these 46 numbers per allograph; any change
here shifts the input distribution and the model starts misreading correct
words. If the notebook is retrained, re-copy this file — don't patch it.

This is NOT the beginner pipeline. `handwriting_preprocessing.py` serves the
letter classifier (6 features/point, max_len 85, BiGRU+MHA). The two share
nothing: different features, different geometry, different model. Keep them
apart.

    student strokes -> preprocess_strokes (Chebyshev-II low-pass, fs=125)
                    -> segment_to_allographs (angle/valley split)
                    -> extract_allograph_features -> (T, 46) float32

The 46 features per allograph, in order:
    [0:10]  local   : const 1.0, |v| mean/std, dir mean/std, |a| mean/std,
                      signed curvature mean, |curvature| mean/std
    [10:14] bbox    : width, height, aspect ratio, baseline angle
    [14:18] ref pts : start/end point relative to the bbox corner
    [18:33] beta    : 3 beta-elliptic pulses x (p, q, K, t0, t1)
    [33:41] fuzzy   : 8x8 density grid reduced to 8 bins, normalised
    [41:43] tangent : start/end tangent direction
    [43:46] loop    : closed-loop indicator, split into top/middle/bottom third

TWO DEPLOYMENT ASSUMPTIONS worth verifying against real canvas data
------------------------------------------------------------------
1. Y DIRECTION. ADAB InkML and the Flutter canvas are both assumed to grow Y
   downward, so — unlike the beginner pipeline — no flip is applied here. The
   notebook's `is_valley` test (`y[i] > y[i-1] and y[i] > y[i+1]`) only reads
   as a valley under downward-Y, which is the evidence for this. If recognition
   is systematically wrong on real drawings, flipping Y is the first thing to
   test.
2. SAMPLE RATE. `fs=125.0` is the ADAB tablet rate and sets the low-pass
   cutoff. A Flutter canvas samples at whatever the device gives (typically
   60-120 Hz), so the filter is not exactly the one used in training.
   ADAB_SAMPLE_RATE is env-overridable for that reason; resampling client
   strokes to 125 Hz before calling this would be the faithful fix.
"""
from __future__ import annotations

import logging
import os

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import cheby2, filtfilt

_log = logging.getLogger("word_handwriting")

N_FEATURES = 46

# ADAB tablet sampling rate. Must stay 125.0 to match training (KHATT used 100).
ADAB_SAMPLE_RATE = float(os.environ.get("ADAB_SAMPLE_RATE", "125.0"))
FCUT = 10.0
TARGET_HEIGHT = 100.0


def strokes_from_client(submitted_drawing: list) -> list[np.ndarray]:
    """Turn the canvas payload into the stroke list `parse_inkml` produced.

    The client sends the same shape the copying screen sends: a list of
    strokes, each a list of [x, y] points. Anything shorter than 1 point or
    unparseable is dropped rather than raising — a stray null in one stroke
    should not lose the whole word.
    """
    out: list[np.ndarray] = []
    for stroke in submitted_drawing or []:
        pts = []
        for p in stroke or []:
            try:
                if isinstance(p, dict):
                    x, y = float(p["x"]), float(p["y"])
                else:
                    x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError, KeyError, IndexError):
                continue
            pts.append([x, y])
        if pts:
            out.append(np.asarray(pts, dtype=float))
    return out


def preprocess_strokes(strokes, fcut=FCUT, fs=ADAB_SAMPLE_RATE,
                       target_height=TARGET_HEIGHT):
    """Low-pass each stroke, then translate/scale the whole sample."""
    processed = []
    Wn = fcut / (0.5 * fs)
    b, a = cheby2(N=4, rs=40, Wn=Wn, btype="low", analog=False)
    all_pts = np.vstack(strokes)
    min_y, max_y = all_pts[:, 1].min(), all_pts[:, 1].max()
    min_x = all_pts[:, 0].min()
    scale = target_height / ((max_y - min_y) if max_y != min_y else 1)
    for s in strokes:
        # filtfilt needs more samples than the filter order; short strokes are
        # passed through unfiltered exactly as the notebook does.
        if len(s) <= 15:
            processed.append((s - [min_x, min_y]) * scale)
            continue
        try:
            fx = filtfilt(b, a, s[:, 0])
            fy = filtfilt(b, a, s[:, 1])
            processed.append((np.column_stack((fx, fy)) - [min_x, min_y]) * scale)
        except ValueError:
            processed.append((s - [min_x, min_y]) * scale)
    return processed


def segment_to_allographs(strokes, min_points=5):
    """Split each stroke at sharp turns and at y-valleys."""
    allographs = []
    for stroke in strokes:
        n = len(stroke)
        if n < min_points:
            allographs.append(stroke)
            continue
        split_indices = [0]
        for i in range(2, n - 2):
            v1, v2 = stroke[i] - stroke[i - 1], stroke[i + 1] - stroke[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                continue
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angle = np.arccos(cos_a)
            is_angular = angle > (np.pi / 3)
            is_valley = (stroke[i, 1] > stroke[i - 1, 1]) and (stroke[i, 1] > stroke[i + 1, 1])
            if (is_angular or is_valley) and ((i - split_indices[-1]) >= min_points):
                split_indices.append(i)
        split_indices.append(n)
        for k in range(len(split_indices) - 1):
            start = split_indices[k]
            if k > 0 and start > 0:
                start -= 1          # one-point overlap between segments
            seg = stroke[start:split_indices[k + 1]]
            if len(seg) > 0:
                allographs.append(seg)
    return allographs


def _beta_pulse(t, K, p, q, t0, t1):
    out = np.zeros_like(t)
    mask = (t >= t0) & (t <= t1) & (t1 > t0) & (p > 0) & (q > 0)
    tt = t[mask]
    denom0 = (t1 - t0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[mask] = K * ((tt - t0) / denom0) ** p * ((t1 - tt) / denom0) ** q
    return np.nan_to_num(out)


def _fit_beta_elliptic(velocity_profile, n_pulses=3):
    """Fit `n_pulses` beta pulses to the speed profile (kinematic signature)."""
    T = len(velocity_profile)
    params = []
    if T < 3:
        return [(0.0, 0.0, 0.0, 0.0, 0.0)] * n_pulses
    t = np.arange(T, dtype=float)
    bounds = np.linspace(0, T, n_pulses + 1).astype(int)
    for i in range(n_pulses):
        t0_idx, t1_idx = bounds[i], max(bounds[i + 1], bounds[i] + 2)
        t1_idx = min(t1_idx, T - 1)
        if t1_idx <= t0_idx:
            params.append((0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        seg_t = t[t0_idx:t1_idx + 1]
        seg_v = velocity_profile[t0_idx:t1_idx + 1]
        K0 = max(seg_v.max(), 1e-3)
        t0_0, t1_0 = seg_t[0], seg_t[-1]
        try:
            popt, _ = curve_fit(
                lambda tt, K, p, q: _beta_pulse(tt, K, p, q, t0_0, t1_0),
                seg_t, seg_v, p0=[K0, 2.0, 2.0],
                bounds=([0, 0.1, 0.1], [K0 * 5 + 1, 10, 10]), maxfev=500,
            )
            K, p, q = popt
        except Exception:
            K, p, q = K0, 2.0, 2.0
        params.append((float(p), float(q), float(K), float(t0_0), float(t1_0)))
    return params


def _fuzzy_template_density(allo, grid=(8, 8), reduce_to=8):
    min_x, max_x = allo[:, 0].min(), allo[:, 0].max()
    min_y, max_y = allo[:, 1].min(), allo[:, 1].max()
    w = max(max_x - min_x, 1e-6)
    h = max(max_y - min_y, 1e-6)
    density = np.zeros(grid)
    for pt in allo:
        gx = min(int((pt[0] - min_x) / w * grid[0]), grid[0] - 1)
        gy = min(int((pt[1] - min_y) / h * grid[1]), grid[1] - 1)
        density[gy, gx] += 1
    flat = density.flatten()
    step = max(len(flat) // reduce_to, 1)
    reduced = np.array([flat[i:i + step].mean() for i in range(0, len(flat), step)])[:reduce_to]
    if len(reduced) < reduce_to:
        reduced = np.pad(reduced, (0, reduce_to - len(reduced)))
    total = reduced.sum()
    return reduced / total if total > 0 else reduced


def extract_allograph_features(allographs) -> np.ndarray:
    """(n_allographs, 46) float32 — the Conformer's input sequence."""
    feature_matrix = []
    for allo in allographs:
        if len(allo) < 2:
            feature_matrix.append(np.zeros(N_FEATURES))
            continue
        velocities = np.diff(allo, axis=0)
        vel_magnitude = np.linalg.norm(velocities, axis=1)
        vel_direction = np.arctan2(velocities[:, 1], velocities[:, 0])
        accelerations = np.diff(velocities, axis=0) if len(velocities) > 1 else np.zeros((1, 2))
        acc_magnitude = (np.linalg.norm(accelerations, axis=1)
                         if len(accelerations) > 0 else np.array([0.0]))
        if len(velocities) > 1:
            d_ang = np.diff(vel_direction)
            d_ang = (d_ang + np.pi) % (2 * np.pi) - np.pi   # wrap to [-pi, pi]
            signed_curv, abs_curv = d_ang, np.abs(d_ang)
        else:
            signed_curv, abs_curv = np.array([0.0]), np.array([0.0])
        local_feats = np.array([
            1.0, vel_magnitude.mean(), vel_magnitude.std(),
            vel_direction.mean(), vel_direction.std(),
            acc_magnitude.mean(), acc_magnitude.std(),
            signed_curv.mean(), abs_curv.mean(), abs_curv.std(),
        ])
        min_x, max_x = allo[:, 0].min(), allo[:, 0].max()
        min_y, max_y = allo[:, 1].min(), allo[:, 1].max()
        width, height = max_x - min_x, max_y - min_y
        aspect_ratio = min(width / max(height, 1.0), 50.0)
        dx_be, dy_be = allo[-1, 0] - allo[0, 0], allo[-1, 1] - allo[0, 1]
        baseline_angle = np.arctan2(dy_be, dx_be) if (dx_be != 0 or dy_be != 0) else 0.0
        bbox_feats = np.array([width, height, aspect_ratio, baseline_angle])
        start_pt, end_pt = allo[0], allo[-1]
        ref_feats = np.array([start_pt[0] - min_x, start_pt[1] - min_y,
                              end_pt[0] - min_x, end_pt[1] - min_y])
        beta_feats = np.array(_fit_beta_elliptic(vel_magnitude, n_pulses=3)).flatten()
        fuzzy_feats = _fuzzy_template_density(allo, grid=(8, 8), reduce_to=8)
        start_tangent = vel_direction[0] if len(vel_direction) > 0 else 0.0
        end_tangent = vel_direction[-1] if len(vel_direction) > 0 else 0.0
        tangent_feats = np.array([start_tangent, end_tangent])
        closed = np.linalg.norm(start_pt - end_pt) < 0.1 * max(height, 1.0)
        y_top_third = min_y + (max_y - min_y) / 3.0
        y_bottom_third = max_y - (max_y - min_y) / 3.0
        loop_feats = np.array([
            float(closed and start_pt[1] <= y_top_third),
            float(closed and y_top_third < start_pt[1] < y_bottom_third),
            float(closed and start_pt[1] >= y_bottom_third),
        ])
        feature_vector = np.concatenate([local_feats, bbox_feats, ref_feats,
                                         beta_feats, fuzzy_feats, tangent_feats,
                                         loop_feats])
        feature_matrix.append(feature_vector)
    if not feature_matrix:
        return np.zeros((1, N_FEATURES), dtype=np.float32)
    return np.array(feature_matrix, dtype=np.float32)


def drawing_to_features(submitted_drawing: list) -> np.ndarray | None:
    """Canvas payload -> (T, 46) float32, or None when there is nothing to read."""
    strokes = strokes_from_client(submitted_drawing)
    if not strokes:
        return None
    try:
        clean = preprocess_strokes(strokes)
        allographs = segment_to_allographs(clean)
        feats = extract_allograph_features(allographs)
    except Exception as exc:
        _log.warning("feature extraction failed: %s", exc)
        return None
    if feats.size == 0 or feats.shape[0] == 0:
        return None
    return feats.astype(np.float32)
