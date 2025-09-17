from typing import TypedDict, List, Tuple, Dict, Any, Optional
import time
import numpy as np

# compare_dtw 모듈의 함수들을 사용합니다.
from compare_dtw import (
    phase_segmentation_simple,
    wrist_center,
    dtw_path,
    phase_metrics,
    aggregate_score,
)

# LangGraph가 없다면 간단한 시퀀셜 래퍼로도 동작하도록 선택적 임포트
try:
    from langgraph.graph import StateGraph, END  # pip install langgraph
    _HAS_LANGGRAPH = True
except Exception:
    _HAS_LANGGRAPH = False


class LGState(TypedDict, total=False):
    pose3d_A: np.ndarray
    pose3d_B: np.ndarray
    seg_A: Dict[str, Any]
    seg_B: Dict[str, Any]
    path: List[Tuple[int, int]]
    dtw_cost: float
    results: Dict[str, Any]
    total_score: float
    detail_scores: Dict[str, float]
    meta: Dict[str, Any]


def _ensure_meta(state: LGState) -> Dict[str, Any]:
    m = state.get("meta", {})
    if not isinstance(m, dict):
        m = {}
    return m


def _merge(state: LGState, updates: Dict[str, Any]) -> LGState:
    new_state: LGState = LGState(**state)
    for k, v in updates.items():
        new_state[k] = v
    return new_state


def node_phase(state: LGState) -> LGState:
    A = state.get("pose3d_A")
    B = state.get("pose3d_B")
    if A is None or B is None:
        raise ValueError("LGState에 pose3d_A/pose3d_B가 필요합니다.")

    segA = phase_segmentation_simple(A)
    segB = phase_segmentation_simple(B)

    m = _ensure_meta(state)
    m["phase"] = {
        "A_phases": segA["phases"],
        "B_phases": segB["phases"],
        "A_marks": segA["marks"],
        "B_marks": segB["marks"],
        "ts": time.time(),
    }
    return _merge(state, {"seg_A": segA, "seg_B": segB, "meta": m})


def node_dtw(state: LGState) -> LGState:
    A = state["pose3d_A"]
    B = state["pose3d_B"]
    T_ = min(A.shape[0], B.shape[0])

    wcA = np.array([wrist_center(A[t])[2] for t in range(T_)], dtype=np.float32)
    wcB = np.array([wrist_center(B[t])[2] for t in range(T_)], dtype=np.float32)

    p, c = dtw_path(wcA, wcB)

    m = _ensure_meta(state)
    m["dtw"] = {
        "cost": float(c),
        "path_len": int(len(p)),
        "cost_norm": float(c / len(p)) if len(p) > 0 else None,
        "ts": time.time(),
    }
    return _merge(state, {"path": p, "dtw_cost": float(c), "meta": m})


def node_metrics(state: LGState) -> LGState:
    A = state["pose3d_A"]
    B = state["pose3d_B"]
    segA = state["seg_A"]
    segB = state["seg_B"]
    p = state["path"]

    phase_names = ["setup", "first_pull", "second_pull", "turnover", "catch_stand"]
    results: Dict[str, Any] = {}
    for name in phase_names:
        r = phase_metrics(p, A, B, segA["phases"], segB["phases"], name)
        results[name] = r

    m = _ensure_meta(state)
    m["metrics"] = {
        "phase_covered": [k for k, v in results.items() if v is not None],
        "ts": time.time(),
    }
    return _merge(state, {"results": results, "meta": m})


def node_scoring(state: LGState) -> LGState:
    results = state["results"]
    total_score, detail_scores = aggregate_score(results, use_max_abs=False)

    m = _ensure_meta(state)
    m["scoring"] = {
        "total": float(total_score),
        "details": {k: float(v) for k, v in detail_scores.items()},
        "ts": time.time(),
    }
    return _merge(
        state,
        {
            "total_score": float(total_score),
            "detail_scores": {k: float(v) for k, v in detail_scores.items()},
            "meta": m,
        },
    )


def build_app():
    """
    LangGraph가 설치되어 있으면 그래프를 컴파일해 반환하고,
    없으면 동일한 순서를 수행하는 간단한 래퍼 함수를 반환합니다.
    """
    if _HAS_LANGGRAPH:
        workflow = StateGraph(LGState)
        workflow.add_node("phase", node_phase)
        workflow.add_node("dtw", node_dtw)
        workflow.add_node("metrics", node_metrics)
        workflow.add_node("scoring", node_scoring)
        workflow.set_entry_point("phase")
        workflow.add_edge("phase", "dtw")
        workflow.add_edge("dtw", "metrics")
        workflow.add_edge("metrics", "scoring")
        workflow.add_edge("scoring", END)
        return workflow.compile()

    # LangGraph 미설치 시: 동일 순서를 수행하는 간단 래퍼
    class _SequentialApp:
        def invoke(self, state0: LGState) -> LGState:
            s = node_phase(state0)
            s = node_dtw(s)
            s = node_metrics(s)
            s = node_scoring(s)
            return s

    return _SequentialApp()