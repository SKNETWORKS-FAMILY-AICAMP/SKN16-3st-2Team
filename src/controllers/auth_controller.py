#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auth Controller - 인증 관련 비즈니스 로직을 처리하는 컨트롤러
"""

import uuid
import gradio as gr
from typing import Tuple, Optional, Dict, Any
from ..models.user_model import UserModel


class AuthController:
    """인증 관련 비즈니스 로직을 처리하는 컨트롤러"""
    
    def __init__(self, user_model: UserModel, app_state: Dict[str, Any]):
        """
        AuthController 초기화
        
        Args:
            user_model (UserModel): 사용자 모델 인스턴스
            app_state (Dict[str, Any]): 애플리케이션 상태
        """
        self.user_model = user_model
        self.app_state = app_state

    def do_login(self, email: str, name: str, pwd: str) -> Tuple:
        """
        사용자 로그인 처리를 수행합니다.
        
        Args:
            email (str): 사용자 이메일
            name (str): 사용자 닉네임
            pwd (str): 비밀번호
            
        Returns:
            Tuple: Gradio 업데이트 객체들
        """
        if not email or not pwd:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value="이메일/비밀번호를 입력해 주세요."),
                gr.update(visible=False),
            )

        session_val, msg = self.user_model.login_user(email, pwd)
        vis_admin = self.user_model.is_admin(session_val)
        user = self.user_model.get_user(email)

        if not user:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value="가입되어 있지 않습니다. 회원가입 후 시도해주세요."),
                gr.update(visible=False),
            )
        elif self.user_model._hash_password(pwd) != user[2]:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value="비밀번호가 일치하지 않습니다."),
                gr.update(visible=False),
            )

        # 앱 상태 업데이트
        self.app_state["user_session"] = {
            "user_id": user[0],
            "name": user[1] or "데모",
            "auth": True,
            "session_key": f"sk-{uuid.uuid4().hex}"
        }
        
        # 기본 근거 자료 설정
        if "preset_sources" in self.app_state:
            self.app_state["evidence_library"] = list(self.app_state["preset_sources"])

        self.user_model.log_session(email, "login")

        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=f"{msg}"),
            gr.update(visible=vis_admin)
        )

    def do_signup(self, email: str, name: str, pwd: str) -> Tuple:
        """
        사용자 회원가입 처리를 수행합니다.
        
        Args:
            email (str): 사용자 이메일
            name (str): 사용자 닉네임
            pwd (str): 비밀번호
            
        Returns:
            Tuple: Gradio 업데이트 객체들
        """
        success = self.user_model.register_user(email, name or email, pwd)
        if success:
            self.user_model.log_session(email, "signup")
            return self.do_login(email, name, pwd)
        else:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value="회원가입에 실패했습니다. 이미 존재하는 이메일일 수 있습니다."),
                gr.update(visible=False),
            )

    def do_logout(self) -> Tuple:
        """
        사용자 로그아웃 처리를 수행합니다.
        
        Returns:
            Tuple: Gradio 업데이트 객체들
        """
        if self.app_state["user_session"]["user_id"]:
            self.user_model.log_session(self.app_state["user_session"]["user_id"], "logout")
        
        # 앱 상태 초기화
        self._reset_app_state()
        
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "로그아웃 되었습니다. 다시 로그인해 주세요.",
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value="")
        )

    def _reset_app_state(self):
        """애플리케이션 상태를 초기화합니다."""
        self.app_state["user_session"] = {
            "user_id": None,
            "name": None,
            "auth": False,
            "session_key": None
        }
        self.app_state["chat_history"] = []
        self.app_state["source_bucket"] = {}
        self.app_state["video_history"] = []
        self.app_state["recommend_history"] = []
        self.app_state["evidence_library"] = []