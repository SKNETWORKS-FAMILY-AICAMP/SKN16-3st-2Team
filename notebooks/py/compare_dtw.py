import numpy as np
import math
from typing import Dict, Any, List, Tuple, Optional

# ----------------------------
# COCO17 index constants
# ----------------------------
JOINTS = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16

# ----------------------------
# Geometry helpers
# ----------------------------
def shoulder_center(P: np.ndarray) -> np.ndarray:
    return (P[L_SHO] + P[R_SHO]) / 2.0

def pelvis_center(P: np.ndarray) -> np.ndarray:
    return (P[L_HIP] + P[R_HIP]) / 2.0

def wrist_center(P: np.ndarray) -> np.ndarray:
    return (P[L_WRI] + P[R_WRI]) / 2.0

def vector_angle(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> float:
    un = u / (np.linalg.norm(u) + eps)
    vn = v / (np.linalg.norm(v) + eps)
    c = float(np.clip(np.dot(un, vn), -1.0, 1.0))
    return math.acos(c)

def joint_angle_3p(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    # angle at b of a-b-c (radians)
    return vector_angle(a - b, c - b)

def trunk_tilt_deg(P: np.ndarray) -> float:
    pc = pelvis_center(P)
    sc = shoulder_center(P)
    v = sc - pc
    z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return math.degrees(vector_angle(v, z))

# ----------------------------
# Phase segmentation (wrist z)
# ----------------------------
def phase_segmentation_simple(pose3d: np.ndarray, smooth_win: int = 7, vel_win: int = 5) -> Dict[str, Any]:
    """
    Input: pose3d (T,17,3) in meaningful coordinates.
    Output: {"phases":[(name,s,e),...], "marks":{...}, "z_s":..., "vel_s":...}
    """
    T = int(pose3d.shape[0])
    wc = np.stack([wrist_center(pose3d[t]) for t in range(T)], axis=0) if T > 0 else np.zeros((0, 3), dtype=np.float32)
    z = wc[:, 2].astype(np.float32) if T > 0 else np.zeros((0,), dtype=np.float32)

    def moving_avg(x: np.ndarray, w: int) -> np.ndarray:
        if w <= 1 or x.size == 0:
            return x
        k = w // 2
        xp = np.concatenate([np.repeat(x[0], k), x, np.repeat(x[-1], k)])
        kern = np.ones(w, dtype=np.float32) / float(w)
        y = np.convolve(xp, kern, mode="valid")
        return y[: len(x)]

    z_s = moving_avg(z, smooth_win)
    vel_s = moving_avg(np.gradient(z_s) if z_s.size > 0 else z_s, vel_win)

    t_second = int(np.argmax(vel_s)) if vel_s.size > 0 else 0
    t_peak = int(np.argmax(z_s)) if z_s.size > 0 else 0
    if t_peak < T - 1 and vel_s.size > 0:
        rel = vel_s[t_peak:]
        t_descent = t_peak + (int(np.argmin(rel)) if rel.size > 0 else 0)
    else:
        t_descent = t_peak
    t_catch = min(max(0, T - 1), t_descent + 10)

    base = float(z_s[0]) if z_s.size > 0 else 0.0
    thr_move = 0.02
    diffs = np.where(np.abs(z_s - base) > thr_move)[0] if z_s.size > 0 else np.array([], dtype=int)
    t_move = int(diffs[0]) if diffs.size > 0 else max(1, t_second // 4)

    def clamp_idx(t: int) -> int:
        return int(max(0, min(T - 1, t)))

    t_move = clamp_idx(t_move)
    t_second = clamp_idx(t_second)
    t_peak = clamp_idx(t_peak)
    t_catch = clamp_idx(t_catch)

    if not (t_move <= t_second):
        t_second = t_move
    if not (t_second <= t_peak):
        t_peak = t_second
    if not (t_peak <= t_catch):
        t_catch = t_peak

    def seg(name: str, s: int, e: int) -> Tuple[str, int, int]:
        s = clamp_idx(s)
        e = clamp_idx(e)
        if e < s:
            e = s
        return (name, s, e)

    phases = [
        seg("setup", 0, max(0, t_move - 1)),
        seg("first_pull", t_move, max(t_move, t_second - 1)),
        seg("second_pull", t_second, max(t_second, t_peak - 1)),
        seg("turnover", t_peak, max(t_peak, t_catch - 1)),
        seg("catch_stand", t_catch, max(t_catch, T - 1)),
    ]
    marks = {"second_pull_peak": t_second, "top": t_peak, "catch": t_catch}
    return {"phases": phases, "marks": marks, "z_s": z_s, "vel_s": vel_s}

# ----------------------------
# DTW (1D absolute-difference)
# ----------------------------
def dtw_path(a: np.ndarray, b: np.ndarray) -> Tuple[List[Tuple[int, int]], float]:
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return [], float("inf")
    D = np.full((na + 1, nb + 1), np.inf, dtype=np.float32)
    D[0, 0] = 0.0
    for i in range(1, na + 1):
        ai = a[i - 1]
        for j in range(1, nb + 1):
            cost = abs(ai - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j = na, nb
    path: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        steps = [(i - 1, j), (i, j - 1), (i - 1, j - 1)]
        i, j = min(steps, key=lambda x: D[x[0], x[1]])
    path.reverse()
    return path, float(D[na, nb])

# ----------------------------
# Per-frame metrics
# ----------------------------
def metrics_per_frame(P: np.ndarray) -> Dict[str, float]:
    m: Dict[str, float] = {}
    m["knee_L"] = math.degrees(joint_angle_3p(P[L_HIP], P[L_KNE], P[L_ANK]))
    m["knee_R"] = math.degrees(joint_angle_3p(P[R_HIP], P[R_KNE], P[R_ANK]))
    sc = shoulder_center(P)
    pc = pelvis_center(P)
    m["hip_L"] = math.degrees(joint_angle_3p(sc, P[L_HIP], P[L_KNE]))
    m["hip_R"] = math.degrees(joint_angle_3p(sc, P[R_HIP], P[R_KNE]))
    m["elbow_L"] = math.degrees(joint_angle_3p(P[L_SHO], P[L_ELB], P[L_WRI]))
    m["elbow_R"] = math.degrees(joint_angle_3p(P[R_SHO], P[R_ELB], P[R_WRI]))
    m["trunk_tilt"] = trunk_tilt_deg(P)
    wc = wrist_center(P)
    body_c = (pc + sc) / 2.0
    m["wrist_xy_dist"] = float(np.linalg.norm((wc - body_c)[:2]))
    return m

# ----------------------------
# Phase metrics aggregation
# ----------------------------
def phase_indices(phases: List[Tuple[str, int, int]], name: str) -> Optional[Tuple[int, int]]:
    for n, s, e in phases:
        if n == name:
            return s, e
    return None

def collect_pairs_in_range(path: List[Tuple[int, int]], s: int, e: int) -> List[Tuple[int, int]]:
    return [(i, j) for (i, j) in path if s <= i <= e]

def phase_metrics(
    path: List[Tuple[int, int]],
    poseA: np.ndarray,
    poseB: np.ndarray,
    phasesA: List[Tuple[str, int, int]],
    phasesB: List[Tuple[str, int, int]],
    name: str
) -> Optional[Tuple[Dict[str, Dict[str, float]], List[Tuple[int, int]]]]:
    rngA = phase_indices(phasesA, name)
    rngB = phase_indices(phasesB, name)
    if rngA is None or rngB is None:
        return None
    sA, eA = rngA
    pairs = collect_pairs_in_range(path, sA, eA)
    if len(pairs) == 0:
        return None
    keys = ["knee_L","knee_R","hip_L","hip_R","elbow_L","elbow_R","trunk_tilt","wrist_xy_dist"]
    diffs: Dict[str, List[float]] = {k: [] for k in keys}
    for (i, j) in pairs:
        mA = metrics_per_frame(poseA[i])
        mB = metrics_per_frame(poseB[j])
        for k in keys:
            diffs[k].append(float(mB[k] - mA[k]))  # B - A
    summary = {
        k: {
            "mean": float(np.mean(v)) if len(v) > 0 else 0.0,
            "abs_mean": float(np.mean(np.abs(v))) if len(v) > 0 else 0.0,
            "max_abs": float(np.max(np.abs(v))) if len(v) > 0 else 0.0,
            "count": int(len(v))
        }
        for k, v in diffs.items()
    }
    return summary, pairs

# ----------------------------
# Scoring
# ----------------------------
TOL = {
    "knee": 15.0,
    "hip": 15.0,
    "elbow": 15.0,
    "trunk_tilt": 10.0,
    "wrist_xy_dist": 0.20
}

W_METRIC = {
    "knee_L": 1.0, "knee_R": 1.0,
    "hip_L": 1.0, "hip_R": 1.0,
    "elbow_L": 0.5, "elbow_R": 0.5,
    "trunk_tilt": 1.0,
    "wrist_xy_dist": 1.0
}

W_PHASE = {
    "setup": 0.05,
    "first_pull": 0.25,
    "second_pull": 0.35,
    "turnover": 0.15,
    "catch_stand": 0.20
}

def score_from_abs_error(k: str, abs_err: float) -> float:
    if k in ["knee_L", "knee_R"]:
        tol = TOL["knee"]
    elif k in ["hip_L", "hip_R"]:
        tol = TOL["hip"]
    elif k in ["elbow_L", "elbow_R"]:
        tol = TOL["elbow"]
    elif k == "trunk_tilt":
        tol = TOL["trunk_tilt"]
    elif k == "wrist_xy_dist":
        tol = TOL["wrist_xy_dist"]
    else:
        tol = 10.0
    x = abs_err / tol
    return max(0.0, 100.0 * (1.0 - x))

def aggregate_score(results: Dict[str, Any], use_max_abs: bool = False) -> Tuple[float, Dict[str, float]]:
    total = 0.0
    details: Dict[str, float] = {}
    for ph, r in results.items():
        if (r is None) or (ph not in W_PHASE):
            continue
        summary, _pairs = r
        ph_score = 0.0
        wsum = 0.0
        for k, s in summary.items():
            if k not in W_METRIC:
                continue
            base_err = s["max_abs"] if use_max_abs else s["abs_mean"]
            sc = score_from_abs_error(k, base_err)
            ph_score += W_METRIC[k] * sc
            wsum += W_METRIC[k]
        ph_score = ph_score / wsum if wsum > 0 else 0.0
        total += W_PHASE[ph] * ph_score
        details[ph] = float(ph_score)
    return float(total), details