import os
import json
import time
from typing import Dict, Any, List, Tuple, Optional

import requests

# REST Chat Completions 엔드포인트 및 기본 모델명
CHAT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4.1-mini"


class LLMConfigError(RuntimeError):
    """LLM 호출 전 환경설정 오류를 명확히 구분하기 위한 예외 클래스."""
    pass


def _require_api_key(env_key: str = "OPENAI_API_KEY") -> str:
    """
    환경변수에서 OpenAI API 키를 읽습니다.
    깃 저장소에 키를 절대 하드코딩하지 말고, 실행환경에서 주입하세요.
    """
    api_key = os.environ.get(env_key, "").strip()
    if not api_key:
        raise LLMConfigError(
            f"환경변수 {env_key} 가 설정되지 않았습니다. "
            "Colab/서버에서 os.environ['OPENAI_API_KEY']로 주입하세요."
        )
    return api_key


def _extract_text(data: dict) -> str:
    """
    Chat Completions 응답 JSON에서 첫 번째 메시지 텍스트만 안전하게 추출합니다.
    """
    try:
        chs = data.get("choices") or []
        if chs:
            msg = chs[0].get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.strip()
    except Exception:
        pass
    return ""


def summarize_results_for_llm_topk(stateF: Dict[str, Any], topk: int = 2) -> str:
    """
    LLM 입력 크기를 줄이기 위해 페이즈별 abs_mean 상위 항목만 요약합니다.
    - mean(부호)도 함께 제공하여 방향성 코칭이 가능하도록 합니다.
    - DTW cost_norm, total_score를 함께 첨부합니다.
    """
    lines: List[str] = []
    results = stateF.get("results", {}) or {}

    for phase_name, r in results.items():
        if r is None:
            continue
        summary, _pairs = r
        # (metric, abs_mean, mean) 튜플 정렬
        items = [(k, summary[k]["abs_mean"], summary[k]["mean"]) for k in summary.keys()]
        items.sort(key=lambda x: x[1], reverse=True)
        sel = items[:max(1, int(topk))]
        comp = ", ".join([f"{k}=abs_mean:{abs_v:.2f},mean:{mean_v:.2f}" for (k, abs_v, mean_v) in sel])
        n = int(summary.get("knee_L", {}).get("count", 0))
        lines.append(f"{phase_name}: {comp}, n={n}")

    # DTW 메타
    cost = stateF.get("dtw_cost", None)
    path_len = len(stateF.get("path", []) or [])
    if cost is not None and path_len > 0:
        lines.append(f"dtw_cost={cost:.3f}, cost_norm={cost/path_len:.3f}, path_len={path_len}")

    # 총점
    total_score = stateF.get("total_score", 0.0)
    lines.append(f"total_score={float(total_score):.1f}")

    return "\n".join(lines)


def build_messages(stateF: Dict[str, Any], topk: int = 2) -> List[Dict[str, str]]:
    """
    gpt-4.1-mini에 전달할 system/user 메시지를 구성합니다.
    - 2~3문장 + 마지막 줄 '근거:' 형식
    - 초/중급자 친화: 용어 뒤 괄호 해설
    - 각 문장에 관련 페이즈 자연스럽게 포함
    """
    system = (
        "당신은 10년 이상의 크로스핏 코치입니다. "
        "전문 용어를 사용하되 초심자/중급자가 이해할 수 있도록 괄호로 쉬운 해설을 덧붙이세요. "
        "모르는 사항은 '모르겠습니다.'라고만 간결히 답하세요. "
        "출력은 2~3문장의 명확한 교정 문장으로 작성하고, 마지막 줄에 '근거:'를 붙이세요. "
        "각 지시는 관련 페이즈명을 자연스럽게 포함하세요(예: second_pull, catch_stand). "
        "한국어로 자연스럽고 매끄럽게, 실전 코칭 톤으로 작성하세요."
    )
    ctx = summarize_results_for_llm_topk(stateF, topk=topk)
    user = (
        "아래 데이터를 바탕으로 2~3문장으로 구체적인 교정 문장을 작성하고, 마지막 줄에 '근거:'를 한 줄로 덧붙이세요.\n"
        "- 각 문장은 하나의 명확한 지시만 담습니다(타이밍/각도/경로/체중이동 등).\n"
        "- 전문 용어 뒤에 괄호로 쉬운 해설을 덧붙이되 문장 흐름은 자연스럽게 유지하세요.\n"
        "- '근거:'에는 페이즈/핵심 항목/abs_mean/총점만 간결히 적으세요.\n\n"
        f"데이터:\n{ctx}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_chat(
    model: str,
    messages: List[Dict[str, str]],
    max_completion_tokens: int = 380,
    timeout: int = 60,
    retries: int = 2,
    backoff: float = 1.2,
) -> Tuple[str, dict]:
    """
    OpenAI Chat Completions REST API 호출.
    - 간단한 재시도(backoff) 지원
    - 성공 시 (텍스트, 원본응답) 반환
    """
    api_key = _require_api_key()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_completion_tokens": int(max_completion_tokens)}

    err: Optional[Exception] = None
    for t in range(max(1, retries)):
        try:
            resp = requests.post(CHAT_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                return _extract_text(data), data
            err = RuntimeError(f"OpenAI API 오류: {resp.status_code} - {resp.text}")
        except Exception as e:
            err = e
        time.sleep(backoff ** t)
    if err:
        raise err
    return "", {}


def generate_coaching(stateF: Dict[str, Any], model: str = DEFAULT_MODEL) -> str:
    """
    최종 코칭 문장 생성(2~3문장 + '근거:' 1줄).
    1차: topk=2, 실패 시 2차: topk=1로 축약.
    """
    # 1차 시도
    try:
        messages = build_messages(stateF, topk=2)
        text, _ = call_chat(model, messages, max_completion_tokens=380)
        if text:
            return text
    except Exception:
        # 2차로 진행
        pass

    # 2차 시도(더 축약)
    try:
        messages2 = build_messages(stateF, topk=1)
        text2, _ = call_chat(model, messages2, max_completion_tokens=360)
        if text2:
            return text2
    except Exception as e:
        return f"코칭 문장 생성 중 오류가 발생했습니다: {e}"

    return "코칭 문장을 생성하지 못했습니다."


# 사용 예시(참고):
# stateF는 pipeline_graph 등에서 계산된 최종 상태(dict)
# from coaching_llm import generate_coaching
# coaching_text = generate_coaching(stateF, model="gpt-4.1-mini")
#
# API 키 주입 방법(깃에 노출 금지):
#   - Colab/서버 셀에서:
#       import os
#       os.environ["OPENAI_API_KEY"] = "sk-..."  # 노트북 공유 시 제거 권장
#   - 또는 getpass로 입력:
#       import getpass, os
#       os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: ").strip()
#   - 또는 .env(깃에 올리지 말 것) + python-dotenv:
#       from dotenv import load_dotenv; load_dotenv()
#       # .env 파일 내 OPENAI_API_KEY=...