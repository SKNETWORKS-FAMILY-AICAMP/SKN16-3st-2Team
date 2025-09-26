#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chatbot Controller - 챗봇 관련 비즈니스 로직을 처리하는 컨트롤러
"""

import json
import uuid
import gradio as gr
from typing import Tuple, Dict, Any, Optional
from ..models.qa_model import QAModel
from ..models.vector_db_model import VectorDBModel


class ChatbotController:
    """챗봇 관련 비즈니스 로직을 처리하는 컨트롤러"""
    
    def __init__(self, qa_model: QAModel, vector_db_model: VectorDBModel, app_state: Dict[str, Any]):
        """
        ChatbotController 초기화
        
        Args:
            qa_model (QAModel): QA 모델 인스턴스
            vector_db_model (VectorDBModel): VectorDB 모델 인스턴스
            app_state (Dict[str, Any]): 애플리케이션 상태
        """
        self.qa_model = qa_model
        self.vector_db_model = vector_db_model
        self.app_state = app_state

    def send_chat(self, user_msg: str) -> Tuple:
        """
        사용자의 질문을 받아 챗봇 응답을 생성합니다.
        
        Args:
            user_msg (str): 사용자 메시지
            
        Returns:
            Tuple: (채팅 히스토리, 상태 메시지, 링크 패널, 근거 자료)
        """
        if not self.app_state["user_session"]["auth"]:
            return self.app_state["chat_history"], "로그인 후 이용해 주세요.", gr.update(value=self._sources_md()), gr.update(value=self._sources_md())
        
        if not user_msg or not user_msg.strip():
            return self.app_state["chat_history"], "질문을 입력해 주세요.", gr.update(value=self._sources_md()), gr.update(value=self._sources_md())

        try:
            # VectorDB가 사용 가능한 경우 우선 사용
            if self.vector_db_model.vectordb and self.vector_db_model.qa_chain:
                result = self.vector_db_model.query(user_msg)
                answer = result.get("result", "답변을 생성할 수 없습니다.")
                
                # 소스 문서 처리 및 evidence_library에 추가
                source_documents = result.get("source_documents", [])
                if source_documents:
                    self._add_sources(source_documents)
            else:
                # VectorDB가 없는 경우 QA 모델의 AI 기능 사용
                answer = self.qa_model.generate_smart_answer(user_msg)
            
            # QA 로그 기록
            if self.app_state["user_session"]["user_id"]:
                self.qa_model.log_qa(
                    self.app_state["user_session"]["user_id"], 
                    user_msg, 
                    answer
                )
            
            # 채팅 히스토리 업데이트
            self.app_state.setdefault("chat_history", [])
            self.app_state["chat_history"].append([user_msg, self._safe_str(answer)])
            
        except Exception as e:
            print(f"Error in chatbot: {e}")
            answer = "죄송합니다. 현재 질문 처리에 문제가 있습니다. 잠시 후 다시 시도해주세요."
            self.app_state["chat_history"].append([user_msg, answer])

        updated_sources_md = self._sources_md()
        return self.app_state["chat_history"], "", gr.update(value=updated_sources_md), gr.update(value=updated_sources_md)

    def send_topic_query(self, topic: str) -> Tuple:
        """
        선택된 주제에 대한 자동 쿼리를 생성하고 챗봇에 전송합니다.
        
        Args:
            topic (str): 주제명
            
        Returns:
            Tuple: (채팅 히스토리, 상태 메시지, 링크 패널, 근거 자료)
        """
        if not self.app_state["user_session"]["auth"]:
            return self._format_chat_history(), "로그인 후 이용해 주세요.", "", self._sources_md()
        
        query = self.qa_model.generate_topic_query(topic)
        
        # 사용자 메시지 추가
        self.app_state["chat_history"].append({"role": "user", "content": f"[{topic}] {query}"})
        
        try:
            # VectorDB를 통한 답변 생성
            result = self.vector_db_model.query(query)
            answer = result.get("result", "답변을 생성할 수 없습니다.")
            
            # 소스 문서 처리
            source_documents = result.get("source_documents", [])
            if source_documents:
                self._add_sources(source_documents[:3])  # 상위 3개만
            
            # 챗봇 응답 추가
            self.app_state["chat_history"].append({"role": "assistant", "content": answer})
            
            # QA 로그 기록
            if self.app_state["user_session"]["user_id"]:
                self.qa_model.log_qa(self.app_state["user_session"]["user_id"], query, answer)
            
            status = f"✅ {topic} 관련 정보가 챗봇에 전송되었습니다."
            
        except Exception as e:
            error_msg = f"죄송합니다. {topic} 정보를 가져오는 중 오류가 발생했습니다: {str(e)}"
            self.app_state["chat_history"].append({"role": "assistant", "content": error_msg})
            status = f"❌ 오류 발생: {str(e)}"
        
        return self._format_chat_history(), status, "", self._sources_md()

    def _format_chat_history(self):
        """
        chat_history를 Gradio Chatbot이 요구하는 [[user, bot], ...] 형식으로 변환
        """
        history = self.app_state.get("chat_history", [])
        formatted = []
        last_user = None
        for item in history:
            if isinstance(item, dict):
                if item.get("role") == "user":
                    last_user = item.get("content", "")
                elif item.get("role") == "assistant":
                    formatted.append([last_user or "", item.get("content", "")])
                    last_user = None
            elif isinstance(item, list) and len(item) == 2:
                formatted.append(item)
        return formatted

    def download_history(self) -> gr.update:
        """
        현재까지의 챗봇 대화 기록을 JSON 파일로 저장합니다.
        
        Returns:
            gr.update: 파일 다운로드 업데이트
        """
        try:
            data = self._export_chat_json()
            fname = self._tmp_filename()
            with open(fname, "w", encoding="utf-8") as f:
                f.write(data)
            return gr.update(value=fname, visible=True)
        except Exception as e:
            print(f"Error downloading history: {e}")
            return gr.update(value=None, visible=False)

    def _safe_str(self, text) -> str:
        """안전한 문자열 변환을 수행합니다."""
        if text is None:
            return ""
        if not isinstance(text, str):
            return str(text)
        return text

    def _add_sources(self, source_documents):
        """챗봇 응답에서 받은 근거 자료들을 evidence_library에 파일명 기준으로 중복 없이 추가합니다."""
        import os
        for doc in source_documents:
            metadata = getattr(doc, 'metadata', {})
            # 파일 경로 또는 url 추출
            src = metadata.get("source") or metadata.get("file_path") or metadata.get("url")
            if not src:
                continue
            fname = os.path.basename(src)
            url = src
            # evidence_library에 중복 없이 추가
            evidence_lib = self.app_state.setdefault("evidence_library", [])
            if not any(item["url"] == url for item in evidence_lib):
                evidence_lib.append({"title": fname, "url": url})

    def _sources_md(self) -> str:
        """현재 적립된 모든 근거 자료들을 Markdown 형식으로 렌더링합니다."""
        evidence_library = self.app_state.get("evidence_library", [])
        if not evidence_library:
            return "근거 자료가 없습니다."
        return "\n".join(
            f"- [{item.get('title') or item['url']}]({item['url']})"
            for item in evidence_library
        )

    def _export_chat_json(self) -> str:
        """챗봇 대화 히스토리를 JSON 문자열로 직렬화합니다."""
        payload = {
            "exported_at": self._now_iso(),
            "user_session_info": self.app_state.get("user_session", {}),
            "chat_turns": self.app_state.get("chat_history", [])
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _tmp_filename(self, prefix: str = "chat_history", ext: str = "json") -> str:
        """고유한 임시 파일 이름을 생성합니다."""
        return f"{prefix}_{uuid.uuid4().hex}.{ext}"

    def _now_iso(self) -> str:
        """현재 시간을 ISO 8601 형식 문자열로 반환합니다."""
        from datetime import datetime
        return datetime.now().isoformat(timespec="seconds")