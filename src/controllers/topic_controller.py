#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topic Controller - 주제별 기능 관련 비즈니스 로직을 처리하는 컨트롤러
"""

from typing import Dict, Any, List
from ..models.qa_model import QAModel


class TopicController:
    """주제별 기능 관련 비즈니스 로직을 처리하는 컨트롤러"""
    
    def __init__(self, qa_model: QAModel, app_state: Dict[str, Any]):
        """
        TopicController 초기화
        
        Args:
            qa_model (QAModel): QA 모델 인스턴스
            app_state (Dict[str, Any]): 애플리케이션 상태
        """
        self.qa_model = qa_model
        self.app_state = app_state

    def search_glossary(self, query: str, category: str) -> str:
        """
        크로스핏 용어집에서 AI 기반 검색을 수행합니다.
        
        Args:
            query (str): 검색어
            category (str): 검색 카테고리
            
        Returns:
            str: AI가 생성한 검색 결과 및 답변
        """
        glossary = self.app_state.get("glossary", [])
        
        # AI 기반 검색 시도
        try:
            return self.qa_model.search_glossary_with_ai(glossary, query, category)
        except Exception as e:
            print(f"AI 검색 실패, 기본 검색으로 대체: {e}")
            # AI 검색 실패 시 기본 검색 사용
            return self.qa_model.search_glossary(glossary, query, category)

    def get_diet_recovery_guide(self, weight_band: str, pref: str, allergy: str) -> str:
        """
        사용자 정보에 기반한 식단 및 회복 가이드라인을 LLM을 통해 제공합니다.
        """
        if not self.app_state["user_session"]["auth"]:
            return "로그인 후 이용해 주세요."
        try:
            vector_db_model = self.app_state.get("vector_db_model")
            if not vector_db_model:
                return "(시스템 오류) VectorDB 모델이 초기화되지 않았습니다."
            prompt = f"체중대: {weight_band}, 선호: {pref}, 알레르기: {allergy or '없음'}인 사용자를 위한 운동 후 식단 및 회복 가이드라인을 구체적으로 알려줘."
            result = vector_db_model.query(prompt)
            return result.get("result", "답변을 생성할 수 없습니다.")
        except Exception as e:
            return f"(오류) 식단/회복 가이드 생성 중 문제 발생: {str(e)}"

    def get_certification_info(self) -> str:
        """
        크로스핏 관련 인증 정보를 LLM을 통해 제공합니다.
        """
        try:
            vector_db_model = self.app_state.get("vector_db_model")
            if not vector_db_model:
                return "(시스템 오류) VectorDB 모델이 초기화되지 않았습니다."
            prompt = "크로스핏 관련 자격증, 온램프 프로그램, 레벨 테스트 등 인증 제도에 대해 설명해줘."
            result = vector_db_model.query(prompt)
            return result.get("result", "답변을 생성할 수 없습니다.")
        except Exception as e:
            return f"(오류) 인증 정보 생성 중 문제 발생: {str(e)}"

    def get_mentoring_preset(self, topic: str) -> str:
        """
        선택된 주제에 대한 멘토링 메시지를 LLM을 통해 제공합니다.
        """
        try:
            vector_db_model = self.app_state.get("vector_db_model")
            if not vector_db_model:
                return "(시스템 오류) VectorDB 모델이 초기화되지 않았습니다."
            prompt = f"크로스핏 초보자를 위한 '{topic}' 상황에서의 멘토링 메시지를 구체적으로 작성해줘."
            result = vector_db_model.query(prompt)
            msg = result.get("result", "답변을 생성할 수 없습니다.")
            # 챗봇 히스토리에 추가
            self.app_state.setdefault("chat_history", []).append({
                "role": "assistant",
                "content": f"[멘토링] {msg}"
            })
            return f"챗봇에 전송됨: {msg}"
        except Exception as e:
            return f"(오류) 멘토링 메시지 생성 중 문제 발생: {str(e)}"

    def convert_weight(self, value: str, unit: str) -> str:
        """
        킬로그램과 파운드 간의 무게 단위를 변환합니다.
        
        Args:
            value (str): 변환할 값
            unit (str): 변환 방향
            
        Returns:
            str: 변환 결과
        """
        try:
            v = float(value)
        except ValueError:
            return "숫자를 입력해 주세요."

        if unit == "kg→lb":
            return f"{v:.2f} kg = {v * 2.2046226218:.2f} lb"
        elif unit == "lb→kg":
            return f"{v:.2f} lb = {v / 2.2046226218:.2f} kg"
        else:
            return "올바른 단위 변환 방향을 선택해 주세요."