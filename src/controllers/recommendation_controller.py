#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recommendation Controller - 개인 맞춤 추천 관련 비즈니스 로직을 처리하는 컨트롤러
"""

import json
from typing import List, Tuple, Dict, Any
from datetime import datetime


class RecommendationController:
    """개인 맞춤 추천 관련 비즈니스 로직을 처리하는 컨트롤러"""
    
    def __init__(self, app_state: Dict[str, Any]):
        """
        RecommendationController 초기화
        
        Args:
            app_state (Dict[str, Any]): 애플리케이션 상태
        """
        self.app_state = app_state
        # vector_db_model이 app_state에 없으면 초기화 시 할당 시도
        if "vector_db_model" not in self.app_state or self.app_state["vector_db_model"] is None:
            try:
                from src.models.vector_db_model import VectorDBModel
                self.app_state["vector_db_model"] = VectorDBModel()
            except Exception as e:
                # 초기화 실패 시 None으로 둠
                self.app_state["vector_db_model"] = None

    def generate_recommendation(self, level: str, goal: str, freq: int, gear: List[str]) -> Tuple:
        """
        사용자의 운동 수준에 맞는 맞춤형 운동 계획을 LLM을 통해 생성합니다.
        """
        if not self.app_state["user_session"]["auth"]:
            return "로그인 후 이용해 주세요.", ""
        try:
            vector_db_model = self.app_state.get("vector_db_model")
            if not vector_db_model:
                return "(시스템 오류) VectorDB 모델이 초기화되지 않았습니다.", ""
            prompt = (
                f"경험 수준: {level}, 목표: {goal}, 주당 운동 횟수: {freq}, "
                f"사용 장비: {', '.join(gear) if gear else '없음'}인 사용자를 위한 1주일 크로스핏 맞춤 운동 계획을 JSON 형식(요약, wod, stretch, notes)으로 작성해줘."
            )
            result = vector_db_model.query(prompt)
            answer = result.get("result", "답변을 생성할 수 없습니다.")

            # markdown code block(```json ... ```) 제거 및 JSON 파싱 시도
            import re
            def extract_json(text):
                match = re.search(r"```json\\s*(.*?)```", text, re.DOTALL)
                if match:
                    return match.group(1)
                match = re.search(r"```\\s*(.*?)```", text, re.DOTALL)
                if match:
                    return match.group(1)
                return text

            pretty = None
            plan = None
            extracted = extract_json(answer)
            try:
                plan = json.loads(extracted)
                pretty = json.dumps(plan, ensure_ascii=False, indent=2)
            except Exception:
                plan = {"summary": answer}
                pretty = answer

            # 추천 히스토리 업데이트
            self.app_state.setdefault("recommend_history", []).append({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "inputs": [level, goal, int(freq), gear],
                "plan": plan
            })
            # 챗봇 히스토리에도 추가
            to_chat = pretty
            self.app_state.setdefault("chat_history", []).append({
                "role": "assistant",
                "content": f"[개인추천]\n{to_chat}"
            })
            return pretty, "챗봇 히스토리에 전송되었습니다."
        except Exception as e:
            return f"(오류) 맞춤 추천 생성 중 문제 발생: {str(e)}", ""