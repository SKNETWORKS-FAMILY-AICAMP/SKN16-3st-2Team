#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recommendation View - 개인 맞춤 추천 UI 컴포넌트
"""

import gradio as gr

class RecommendationView:
    """개인 맞춤 추천 관련 UI 컴포넌트를 관리하는 클래스"""
    
    @staticmethod
    def create_recommendation_tab():
        """개인 맞춤 추천 탭을 생성합니다."""
        with gr.Tab("개인 맞춤 추천", elem_id="recommendation_tab"):
            level = gr.Radio(
                ["초보", "중급", "상급"],
                value="초보",
                label="경험 수준",
                info="현재 운동 경험 수준을 선택해주세요."
            )
            goal = gr.Radio(
                ["체지방 감량", "근력 향상", "기술 습득"],
                value="기술 습득",
                label="주요 운동 목표",
                info="가장 중요한 운동 목표를 선택해주세요."
            )
            freq = gr.Slider(
                1, 6, step=1, value=3,
                label="주당 운동 횟수",
                info="일주일에 몇 번 운동하고 싶으신가요?"
            )
            gear = gr.CheckboxGroup(
                ["덤벨", "케틀벨", "바벨", "로워링 밴드"],
                label="사용 가능한 장비 선택",
                info="현재 활용 가능한 운동 장비를 모두 선택해주세요."
            )
            rec_btn = gr.Button("맞춤 추천 생성", variant="primary")
            rec_out = gr.Code(label="추천 결과(JSON)", language="json", interactive=False)
            rec_status = gr.Markdown("", elem_id="recommendation_status")
        
        return {
            'level': level,
            'goal': goal,
            'freq': freq,
            'gear': gear,
            'rec_btn': rec_btn,
            'rec_out': rec_out,
            'rec_status': rec_status
        }