#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auth View - 로그인/회원가입 UI 컴포넌트
"""

import gradio as gr


class AuthView:
    """인증 관련 UI 컴포넌트를 관리하는 클래스"""
    
    @staticmethod
    def create_auth_page():
        """로그인/회원가입 페이지를 생성합니다."""
        with gr.Column(visible=True) as entry_page:
            gr.Markdown("### 로그인 / 회원가입")
            with gr.Row():
                email = gr.Textbox(label="이메일", placeholder="you@example.com", scale=2, type="email")
                name = gr.Textbox(label="닉네임", placeholder="닉네임(선택)", scale=1)
                pwd = gr.Textbox(label="비밀번호", type="password", placeholder="비밀번호", scale=2)
            with gr.Row():
                login_btn = gr.Button("로그인", variant="primary", scale=2)
                signup_btn = gr.Button("회원가입", scale=1)
            entry_status = gr.Markdown("")
        
        return {
            'entry_page': entry_page,
            'email': email,
            'name': name,
            'pwd': pwd,
            'login_btn': login_btn,
            'signup_btn': signup_btn,
            'entry_status': entry_status
        }