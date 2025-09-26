#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA Model - 질문답변 관련 데이터베이스 작업과 로직을 담당하는 모델
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

# OpenAI 관련 import
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain.prompts import ChatPromptTemplate
    from langchain_community.vectorstores import Chroma
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI/LangChain not available. Using fallback search.")

from ..config import config


class QAModel:
    """QA 관련 데이터베이스 작업을 처리하는 모델 클래스"""
    
    def __init__(self, db_path: str):
        """
        QAModel 초기화
        
        Args:
            db_path (str): SQLite 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.openai_api_key = config.openai_api_key
        self.llm = None
        self.embeddings = None
        self.init_tables()
        self.init_openai_models()
    
    def init_openai_models(self):
        """OpenAI 모델들을 초기화합니다."""
        if OPENAI_AVAILABLE and self.openai_api_key:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.3,
                    api_key=self.openai_api_key
                )
                self.embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
                print("✅ OpenAI 모델 초기화 완료")
            except Exception as e:
                print(f"⚠️ OpenAI 모델 초기화 실패: {e}")
                self.llm = None
                self.embeddings = None
        else:
            print("⚠️ OpenAI API 키가 없거나 라이브러리가 설치되지 않았습니다.")
    
    def init_tables(self):
        """QA 관련 테이블을 초기화합니다."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()

            # QA 로그
            cur.execute("""
            CREATE TABLE IF NOT EXISTS qa_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT,
                question TEXT,
                answer TEXT,
                ts TEXT
            )
            """)
            
            conn.commit()

    def log_qa(self, email: str, question: str, answer: str):
        """
        사용자 QA 로그를 기록합니다.
        
        Args:
            email (str): 사용자 이메일
            question (str): 질문
            answer (str): 답변
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO qa_logs (email, question, answer, ts) VALUES (?, ?, ?, ?)",
                (email, question, answer, datetime.now().isoformat())
            )
            conn.commit()

    def get_user_qa_history(self, email: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        사용자의 QA 히스토리를 조회합니다.
        
        Args:
            email (str): 사용자 이메일
            limit (int): 조회할 최대 개수
            
        Returns:
            List[Dict[str, Any]]: QA 히스토리 리스트
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT question, answer, ts FROM qa_logs WHERE email=? ORDER BY ts DESC LIMIT ?",
                (email, limit)
            )
            rows = cur.fetchall()
            
            return [
                {
                    "question": row[0],
                    "answer": row[1],
                    "timestamp": row[2]
                }
                for row in rows
            ]

    @staticmethod
    def generate_topic_query(topic: str) -> str:
        """
        주제에 따른 자동 쿼리를 생성합니다.
        
        Args:
            topic (str): 주제명
            
        Returns:
            str: 생성된 쿼리
        """
        import random
        
        topic_queries = {
            "용어/규칙/운동법": [
                "크로스핏의 기본 용어와 규칙에 대해 설명해주세요.",
                "크로스핏 운동법의 기본 원리와 주요 운동들을 알려주세요.",
                "WOD(Workout of the Day)의 구성과 스케일링 방법을 설명해주세요.",
                "크로스핏 박스에서 지켜야 할 안전 규칙과 에티켓을 알려주세요."
            ],
            "식단/회복": [
                "크로스핏 운동 후 효과적인 식단과 회복 방법을 알려주세요.",
                "운동 전후 영양 섭취 가이드라인을 제공해주세요.",
                "근육 회복을 위한 수면과 휴식의 중요성을 설명해주세요.",
                "크로스핏 선수들의 일반적인 식단 패턴과 보충제 사용법을 알려주세요."
            ],
            "인증/챌린지 안내": [
                "크로스핏 레벨 1 인증 과정과 자격요건을 설명해주세요.",
                "크로스핏 오픈(CrossFit Open) 참가 방법과 준비 과정을 알려주세요.",
                "크로스핏 박스에서 진행하는 챌린지와 대회 정보를 제공해주세요.",
                "크로스핏 코치 자격증 취득 과정과 커리어 패스를 설명해주세요."
            ],
            "멘토링(초보 심리/동기)": [
                "크로스핏 초보자가 겪는 심리적 어려움과 극복 방법을 알려주세요.",
                "운동 동기를 유지하고 지속적으로 발전하는 방법을 제공해주세요.",
                "크로스핏 커뮤니티에 적응하고 관계를 형성하는 팁을 알려주세요.",
                "운동 목표 설정과 달성을 위한 체계적인 접근법을 설명해주세요."
            ]
        }
        
        queries = topic_queries.get(topic, ["크로스핏에 대해 알려주세요."])
        return random.choice(queries)

    def generate_smart_answer(self, question: str, context_data: Optional[List[Dict]] = None) -> str:
        """
        OpenAI를 활용하여 스마트한 답변을 생성합니다.
        
        Args:
            question (str): 질문
            context_data (List[Dict], optional): 컨텍스트 데이터 (용어집 등)
            
        Returns:
            str: AI가 생성한 답변
        """
        if not self.llm:
            return self._fallback_answer(question, context_data)
        
        try:
            # 컨텍스트 준비
            context = ""
            if context_data:
                context = "\n".join([
                    f"- {item.get('term', '')}: {item.get('desc', '')}" 
                    for item in context_data[:10]  # 최대 10개만 사용
                ])
            
            # 프롬프트 템플릿 생성
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """당신은 크로스핏 전문 코치이자 트레이너입니다. 
사용자의 질문에 대해 정확하고 도움이 되는 답변을 제공해주세요.

답변 가이드라인:
- 운동 기법, 안전, 영양에 대한 정확한 정보 제공
- 초보자도 이해할 수 있도록 쉽게 설명
- 가능한 한 구체적인 예시와 팁 포함
- 안전을 최우선으로 고려한 조언
- 한국어로 답변

{context_info}"""),
                ("human", "{question}")
            ])
            
            context_info = f"참고 정보:\n{context}" if context else ""
            
            # LLM 체인 실행
            chain = prompt_template | self.llm
            response = chain.invoke({
                "context_info": context_info,
                "question": question
            })
            
            return response.content
            
        except Exception as e:
            print(f"OpenAI API 호출 실패: {e}")
            return self._fallback_answer(question, context_data)
    
    def _fallback_answer(self, question: str, context_data: Optional[List[Dict]] = None) -> str:
        """
        OpenAI API 사용이 불가능할 때의 대체 답변
        
        Args:
            question (str): 질문
            context_data (List[Dict], optional): 컨텍스트 데이터
            
        Returns:
            str: 대체 답변
        """
        if context_data:
            # 키워드 매칭으로 관련 정보 찾기
            question_lower = question.lower()
            relevant_items = []
            
            for item in context_data:
                term = item.get('term', '').lower()
                desc = item.get('desc', '').lower()
                
                if (any(word in term for word in question_lower.split()) or 
                    any(word in desc for word in question_lower.split())):
                    relevant_items.append(item)
            
            if relevant_items:
                result = "관련 정보를 찾았습니다:\n\n"
                for item in relevant_items[:5]:  # 최대 5개
                    result += f"• {item.get('term', '')}: {item.get('desc', '')}\n"
                return result
        
        return "죄송합니다. 현재 OpenAI API를 사용할 수 없어 상세한 답변을 제공하기 어렵습니다. 관리자에게 문의하시거나 나중에 다시 시도해주세요."

    def search_glossary_with_ai(self, glossary: List[Dict], query: str, category: str) -> str:
        """
        AI를 활용하여 크로스핏 용어집에서 검색을 수행합니다.
        
        Args:
            glossary (List[Dict]): 용어집 데이터
            query (str): 검색어
            category (str): 검색 카테고리
            
        Returns:
            str: AI가 생성한 검색 결과 및 답변
        """
        # 기존 검색 로직으로 관련 용어들 필터링
        filtered_glossary = self._filter_glossary(glossary, query, category)
        
        # AI를 활용한 스마트 답변 생성
        if self.llm:
            search_question = f"크로스핏에서 '{query}'에 대해 설명해주세요. 카테고리: {category}"
            # 매칭된 용어가 있으면 참고 정보로 제공, 없어도 AI가 답변 생성
            return self.generate_smart_answer(search_question, filtered_glossary if filtered_glossary else None)
        else:
            # AI를 사용할 수 없는 경우
            if not filtered_glossary:
                return "검색 결과가 없습니다."
            return self.search_glossary(glossary, query, category)
    
    def _filter_glossary(self, glossary: List[Dict], query: str, category: str) -> List[Dict]:
        """
        용어집을 필터링합니다.
        
        Args:
            glossary (List[Dict]): 용어집 데이터
            query (str): 검색어
            category (str): 검색 카테고리
            
        Returns:
            List[Dict]: 필터링된 용어집
        """
        q_norm = (query or "").strip().lower()
        cat_str = str(category or "전체")
        if cat_str not in {"전체", "용어", "프로그램"}:
            cat_str = "전체"

        filtered_items = []
        for item in glossary:
            ok_q = (not q_norm) or (q_norm in item["term"].lower()) or (q_norm in item["desc"].lower())
            ok_c = (cat_str == "전체") or (item["category"] == cat_str)

            if ok_q and ok_c:
                filtered_items.append(item)

        return filtered_items

    @staticmethod
    def search_glossary(glossary: List[Dict], query: str, category: str) -> str:
        """
        크로스핏 용어집에서 기본 텍스트 검색을 수행합니다. (AI 없이)
        
        Args:
            glossary (List[Dict]): 용어집 데이터
            query (str): 검색어
            category (str): 검색 카테고리
            
        Returns:
            str: 검색 결과 (단순 텍스트 매칭)
            
        Note:
            AI 기반 검색을 원한다면 search_glossary_with_ai() 메서드를 사용하세요.
        """
        q_norm = (query or "").strip().lower()
        cat_str = str(category or "전체")
        if cat_str not in {"전체", "용어", "프로그램"}:
            cat_str = "전체"

        rows = []
        for item in glossary:
            ok_q = (not q_norm) or (q_norm in item["term"].lower()) or (q_norm in item["desc"].lower())
            ok_c = (cat_str == "전체") or (item["category"] == cat_str)

            if ok_q and ok_c:
                rows.append(f"- {item['term']} ({item['category']}): {item['desc']}")

        return "\n".join(rows) if rows else "결과가 없습니다."