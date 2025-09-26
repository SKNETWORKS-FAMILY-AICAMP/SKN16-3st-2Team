#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chatbot View - 챗봇 UI 컴포넌트
"""

import gradio as gr


class ChatbotView:
    """챗봇 관련 UI 컴포넌트를 관리하는 클래스"""
    
    @staticmethod
    def create_chatbot_tab():
        """챗봇 탭을 생성합니다."""
        with gr.Tab("챗봇(Q&A)", elem_id="chatbot_tab"):
            with gr.Row():
                user_input = gr.Textbox(label="질문", placeholder="예) 스쿼트 클린 무릎 각도는?", scale=4)
                send_btn = gr.Button("전송", variant="primary", scale=1)
            chat_view = gr.Chatbot(label="답변 및 채팅 히스토리", height=420, render_markdown=True)
            links_panel = gr.Markdown("링크 모아보기", label="연관 자료 링크", elem_id="links_panel_chatbot")
            with gr.Row():
                download_btn = gr.Button("히스토리 내려받기", elem_id="download_button")
                download_file = gr.File(label="다운로드 파일", visible=False, elem_id="download_file_output")
        
        return {
            'user_input': user_input,
            'send_btn': send_btn,
            'chat_view': chat_view,
            'links_panel': links_panel,
            'download_btn': download_btn,
            'download_file': download_file
        }