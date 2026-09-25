# src/expression.py
"""
Facial expression detection from the full MediaPipe FaceMesh landmarks.

Geometry is measured relative to the person's own *neutral* face: the first
CALIBRATION_FRAMES frames are averaged into a baseline, and every later
frame is classified by how far it moved away from that baseline. This makes
the rules work across different face shapes (someone whose mouth corners
naturally turn down is not "sad" all the time).

All distances are divided by the eye-corner distance and measured after
rotating the face upright, so they don't change with camera distance or
head tilt.

Expressions: neutral, smiling, sad, frowning, surprised, grimacing.
Blinks are counted separately from the (unsmoothed) eye aspect ratio.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import time

import numpy as np

# FaceMesh landmark indices
L_EYE_OUT, R_EYE_OUT = 33, 263
MOUTH_L, MOUTH_R = 61, 291
LIP_TOP, LIP_BOT = 13, 14             # inner lips
L_LID_TOP, L_LID_BOT = 159, 145
R_LID_TOP, R_LID_BOT = 386, 374
L_BROW_MID, R_BROW_MID = 105, 334
L_BROW_IN, R_BROW_IN = 55, 285
# 6-point eye contours for the eye aspect ratio: corner, top, top, corner, bottom, bottom
L_EYE_EAR = (33, 160, 158, 133, 153, 144)
R_EYE_EAR = (362, 385, 387, 263, 373, 380)

CALIBRATION_FRAMES = 30
FEATURE_EMA = 0.6        # smoothing of per-frame features (higher = smoother, slower)
STABLE_FRAMES = 4        # a new label must hold this many frames before it is reported

# Thresholds on (feature - neutral baseline). Tune with the debug overlay ('d').
SMILE_LIFT = 0.030       # mouth corners rise above lip centre
SMILE_WIDTH = 0.07       # mouth width grows by 7%
SAD_LIFT = 0.018         # mouth corners drop below lip centre
FROWN_GAP = 0.93         # inner brows pulled together to 93% of neutral gap
FROWN_BROW = 0.020       # brows lowered toward eyes
SURPRISE_OPEN = 0.10     # mouth opens
SURPRISE_BROW = 0.030    # brows raised
GRIMACE_WIDTH = 0.10     # mouth stretched 10% wider...
GRIMACE_MAX_LIFT = 0.015 # ...without the corners rising like a smile

# Blinks: eye aspect ratio (EAR) relative to the person's open-eye EAR
BLINK_CLOSE_RATIO = 0.70  # below this fraction of open EAR -> eyes closed
BLINK_OPEN_RATIO = 0.85   # back above this -> eyes open again (hysteresis)
BLINK_MAX_S = 0.6         # closures longer than this are "eyes closed", not a blink


@dataclass
class ExpressionResult:
    label: str               # stable label ("calibrating" until baseline is ready)
    raw_label: str           # this frame's label before stability filtering
    deltas: Dict[str, float]  # feature changes vs neutral (for debugging/tuning)
    calib_progress: float     # 0..1
    blinks: int = 0           # blinks counted since calibration
    eyes_closed: bool = False
    just_blinked: bool = False  # a blink completed on this frame


def _upright(mesh: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate landmarks so the eye line is horizontal; returns (points, eye distance)."""
    le, re = mesh[L_EYE_OUT], mesh[R_EYE_OUT]
    d = re - le
    iod = float(np.linalg.norm(d)) + 1e-6
    ang = np.arctan2(d[1], d[0])
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    center = (le + re) / 2.0
    return (mesh - center) @ R.T, iod


def _ear(mesh: np.ndarray, idx) -> float:
    p = [mesh[i] for i in idx]
    vert = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horiz = np.linalg.norm(p[0] - p[3]) + 1e-6
    return float(vert / (2.0 * horiz))


def eye_aspect_ratio(mesh: np.ndarray) -> float:
    """Mean EAR of both eyes: ~0.25-0.35 open, drops toward 0 when closed."""
    return (_ear(mesh, L_EYE_EAR) + _ear(mesh, R_EYE_EAR)) / 2.0


def extract_features(mesh: np.ndarray) -> Dict[str, float]:
    p, iod = _upright(mesh.astype(np.float32))
    y = p[:, 1]  # image coords: y grows downward
    lip_mid_y = (y[LIP_TOP] + y[LIP_BOT]) / 2.0
    return {
        "mouth_w": float(np.linalg.norm(p[MOUTH_L] - p[MOUTH_R]) / iod),
        "corner_lift": float((lip_mid_y - (y[MOUTH_L] + y[MOUTH_R]) / 2.0) / iod),
        "mouth_open": float((y[LIP_BOT] - y[LIP_TOP]) / iod),
        "brow_raise": float(((y[L_LID_TOP] - y[L_BROW_MID]) + (y[R_LID_TOP] - y[R_BROW_MID])) / 2.0 / iod),
        "brow_gap": float(np.linalg.norm(p[L_BROW_IN] - p[R_BROW_IN]) / iod),
        "eye_open": float(((y[L_LID_BOT] - y[L_LID_TOP]) + (y[R_LID_BOT] - y[R_LID_TOP])) / 2.0 / iod),
    }


def classify(feat: Dict[str, float], base: Dict[str, float]) -> tuple[str, Dict[str, float]]:
    d = {
        "lift": feat["corner_lift"] - base["corner_lift"],
        "width": feat["mouth_w"] / (base["mouth_w"] + 1e-6) - 1.0,
        "open": feat["mouth_open"] - base["mouth_open"],
        "brow": feat["brow_raise"] - base["brow_raise"],
        "gap": feat["brow_gap"] / (base["brow_gap"] + 1e-6),
    }
    if d["open"] > SURPRISE_OPEN and d["brow"] > SURPRISE_BROW:
        label = "surprised"
    elif d["width"] > GRIMACE_WIDTH and d["lift"] < GRIMACE_MAX_LIFT:
        label = "grimacing"
    elif d["lift"] > SMILE_LIFT or (d["width"] > SMILE_WIDTH and d["lift"] > SMILE_LIFT / 2):
        label = "smiling"
    elif d["gap"] < FROWN_GAP or d["brow"] < -FROWN_BROW:
        label = "frowning"
    elif d["lift"] < -SAD_LIFT:
        label = "sad"
    else:
        label = "neutral"
    return label, d


class ExpressionDetector:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Forget the baseline; the next CALIBRATION_FRAMES frames recalibrate (keep a neutral face)."""
        self._calib: List[Dict[str, float]] = []
        self._base: Optional[Dict[str, float]] = None
        self._smooth: Optional[Dict[str, float]] = None
        self._label = "neutral"
        self._candidate = "neutral"
        self._candidate_n = 0
        self._ear_calib: List[float] = []
        self._ear_open: Optional[float] = None
        self._closed_since: Optional[float] = None
        self.blinks = 0

    @property
    def calibrated(self) -> bool:
        return self._base is not None

    def _update_blink(self, ear: float, now: float) -> bool:
        """Returns True when a blink just finished."""
        if self._ear_open is None:
            return False
        if self._closed_since is None:
            if ear < BLINK_CLOSE_RATIO * self._ear_open:
                self._closed_since = now
            else:
                # slowly follow lighting / head-pose changes of the open-eye EAR
                self._ear_open = 0.98 * self._ear_open + 0.02 * ear
            return False
        if ear > BLINK_OPEN_RATIO * self._ear_open:
            dur = now - self._closed_since
            self._closed_since = None
            if dur <= BLINK_MAX_S:
                self.blinks += 1
                return True
        return False

    def update(self, mesh: np.ndarray) -> ExpressionResult:
        now = time.monotonic()
        ear = eye_aspect_ratio(mesh)
        feat = extract_features(mesh)
        if self._smooth is None:
            self._smooth = feat
        else:
            a = FEATURE_EMA
            self._smooth = {k: a * self._smooth[k] + (1.0 - a) * v for k, v in feat.items()}

        if self._base is None:
            self._calib.append(feat)
            self._ear_calib.append(ear)
            if len(self._calib) >= CALIBRATION_FRAMES:
                self._base = {k: float(np.median([f[k] for f in self._calib])) for k in feat}
                self._ear_open = float(np.percentile(self._ear_calib, 75))  # ignore blinks during calibration
            return ExpressionResult("calibrating", "calibrating", {}, len(self._calib) / CALIBRATION_FRAMES)

        just_blinked = self._update_blink(ear, now)
        eyes_closed = self._closed_since is not None and (now - self._closed_since) > BLINK_MAX_S

        raw, deltas = classify(self._smooth, self._base)
        deltas["ear"] = ear / (self._ear_open + 1e-6)
        if raw == self._candidate:
            self._candidate_n += 1
        else:
            self._candidate, self._candidate_n = raw, 1
        if self._candidate_n >= STABLE_FRAMES:
            self._label = self._candidate
        return ExpressionResult(self._label, raw, deltas, 1.0, self.blinks, eyes_closed, just_blinked)
