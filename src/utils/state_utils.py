#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State Utils - 애플리케이션 상태 관리 유틸리티
"""

from typing import Dict, Any, List
from datetime import datetime


class StateUtils:
    """애플리케이션 상태 관리 유틸리티 기능을 제공하는 클래스"""
    
    @staticmethod
    def init_app_state() -> Dict[str, Any]:
        """애플리케이션의 초기 상태를 생성합니다."""
        return {
            "user_session": {
                "user_id": None,
                "name": None,
                "auth": False,
                "session_key": None
            },
            "chat_history": [],
            "source_bucket": {},
            "video_history": [],
            "recommend_history": [],
            "evidence_library": [],  # 초기값은 빈 리스트, 답변 생성 시만 추가
            "glossary": StateUtils.get_default_glossary(),
            "preset_sources": []
        }

    @staticmethod
    def load_evidence_from_data() -> List[Dict]:
        """
        data/raw/crossfit_guide/ 내 PDF 파일을 evidence_library로 자동 등록 (중복 없이, 파일명만 title)
        """
        import os
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw", "crossfit_guide")
        if not os.path.exists(base_dir):
            return []
        seen = set()
        evidence = []
        for fname in os.listdir(base_dir):
            if fname.lower().endswith(".pdf"):
                url = os.path.join("data", "raw", "crossfit_guide", fname)
                if url not in seen:
                    evidence.append({"title": fname, "url": url})
                    seen.add(url)
        return evidence

    @staticmethod
    def get_default_glossary() -> List[Dict]:
        """크로스핏 관련 용어집 초기 데이터를 정의합니다."""
        return [
            {"term": "박스", "category": "용어", "desc": "크로스핏 체육관을 의미"},
            {"term": "WOD", "category": "용어", "desc": "Workout of the Day, 하루 운동 프로그램"},
            {"term": "온램프", "category": "프로그램", "desc": "초보자 적응 과정"},
            
            # 올림픽 리프팅 용어
            {"term": "스내치", "category": "용어", "desc": "바벨을 땅에서 머리 위로 한 번에 들어올리는 올림픽 리프팅 동작"},
            {"term": "파워 스내치", "category": "용어", "desc": "스쿼트 자세 없이 서서 바벨을 머리 위로 들어올리는 스내치 변형"},
            {"term": "클린", "category": "용어", "desc": "바벨을 땅에서 어깨 위로 들어올리는 올림픽 리프팅 동작"},
            {"term": "파워 클린", "category": "용어", "desc": "스쿼트 자세 없이 서서 바벨을 어깨 위로 들어올리는 클린 변형"},
            {"term": "저크", "category": "용어", "desc": "어깨에서 머리 위로 바벨을 밀어올리는 동작"},
            {"term": "세컨드 풀", "category": "용어", "desc": "올림픽 리프팅에서 바벨을 가속시키는 두 번째 당기기 동작"},
            
            # 기본 운동 용어
            {"term": "쓰러스터", "category": "용어", "desc": "프론트 스쿼트에서 바로 푸시 프레스로 연결하는 복합 운동"},
            {"term": "버피", "category": "용어", "desc": "스쿼트 자세에서 플랭크, 푸시업, 점프로 연결되는 전신 운동"},
            {"term": "데드리프트", "category": "용어", "desc": "바닥의 바벨을 허벅지까지 들어올리는 기본 리프팅 동작"},
            {"term": "풀업", "category": "용어", "desc": "철봉에 매달려 턱이 바 위로 올라오도록 몸을 끌어올리는 운동"},
            {"term": "스쿼트", "category": "용어", "desc": "하체를 굽혔다 펴는 기본적인 하체 운동"},
            
            # WOD 관련 용어
            {"term": "AMRAP", "category": "용어", "desc": "As Many Rounds/Reps As Possible, 정해진 시간 내에 최대한 많은 라운드나 반복"},
            {"term": "EMOM", "category": "용어", "desc": "Every Minute On the Minute, 매분마다 정해진 운동 수행"},
            {"term": "타바타", "category": "프로그램", "desc": "20초 운동, 10초 휴식을 8라운드 반복하는 고강도 인터벌 프로그램"},
            {"term": "RX", "category": "용어", "desc": "처방된 대로, 스케일링 없이 원래 강도로 운동 수행"},
            {"term": "스케일링", "category": "용어", "desc": "개인의 능력에 맞게 운동 강도나 무게를 조정하는 것"},
            
            # 장비 용어
            {"term": "케틀벨", "category": "용어", "desc": "손잡이가 달린 구형 중량 기구"},
            {"term": "플라이오박스", "category": "용어", "desc": "박스 점프 등에 사용되는 사각형 점프대"},
            {"term": "로잉머신", "category": "용어", "desc": "조정 동작을 시뮬레이션하는 유산소 운동 기구"},
            {"term": "월볼", "category": "용어", "desc": "벽에 던지는 무거운 공, 스쿼트 투 월볼 샷에 사용"},
        ]

    @staticmethod
    def now_iso() -> str:
        """현재 시간을 ISO 8601 형식 문자열로 반환합니다."""
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def reset_user_session(app_state: Dict[str, Any]):
        """사용자 세션을 초기화합니다."""
        app_state["user_session"] = {
            "user_id": None,
            "name": None,
            "auth": False,
            "session_key": None
        }
        app_state["chat_history"] = []
        app_state["source_bucket"] = {}
        app_state["video_history"] = []
        app_state["recommend_history"] = []
        app_state["evidence_library"] = []

    @staticmethod
    def is_authenticated(app_state: Dict[str, Any]) -> bool:
        """사용자가 인증되었는지 확인합니다."""
        return app_state.get("user_session", {}).get("auth", False)