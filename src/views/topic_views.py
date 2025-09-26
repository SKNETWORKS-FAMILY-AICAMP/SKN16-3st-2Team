#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topic Views - 주제별 탭 UI 컴포넌트들
"""

import gradio as gr

class TopicViews:
    """주제별 탭 UI 컴포넌트들을 관리하는 클래스"""
    
    @staticmethod
    def create_glossary_tab():
        """용어/규칙/운동법 탭을 생성합니다."""
        with gr.Tab("용어/규칙/운동법", elem_id="glossary_tab"):
            gr.Markdown("### 📚 크로스핏 용어, 규칙, 운동법 정보")
            
            terms_status = gr.Markdown("", elem_id="terms_status")
            
            gr.Markdown("---")
            gr.Markdown("### 🔍 용어집 검색")
            
            q = gr.Textbox(label="검색어", placeholder="예) WOD, 스내치")
            cat = gr.Dropdown(
                ["전체", "용어", "프로그램"],
                value="전체",
                label="검색 카테고리",
                info="검색할 용어의 카테고리를 선택해주세요."
            )
            search_btn = gr.Button("용어/규칙/운동법 검색", variant="primary", size="lg")
            glo_out = gr.Markdown("용어집 검색 결과", elem_id="glossary_output")
        
        return {
            'terms_status': terms_status,
            'q': q,
            'cat': cat,
            'search_btn': search_btn,
            'glo_out': glo_out
        }

    @staticmethod
    def create_diet_tab():
        """식단/회복 탭을 생성합니다."""
        with gr.Tab("식단/회복", elem_id="diet_recovery_tab"):
            gr.Markdown("### 🥗 식단 및 회복 가이드")
            diet_query_status = gr.Markdown("", elem_id="diet_query_status")
            gr.Markdown("---")
            gr.Markdown("### 🎯 개인 맞춤 가이드")
            weight_band = gr.Dropdown(
                ["<60kg", "60~80kg", ">80kg"],
                value="60~80kg",
                label="현재 체중대",
                info="현재 체중이 속하는 구간을 선택해주세요."
            )
            pref = gr.Dropdown(
                ["선호 없음", "고단백", "채식"],
                value="고단백",
                label="식성 선호도",
                info="평소 식단 선호도를 선택해주세요."
            )
            allergy = gr.Textbox(
                label="알레르기 정보 (선택 사항)",
                placeholder="예) 유제품, 견과류"
            )
            diet_btn = gr.Button("회복 가이드 생성", variant="primary")
            diet_out = gr.Markdown("식단 및 회복 가이드", elem_id="diet_output")
        
        return {
            'diet_query_status': diet_query_status,
            'weight_band': weight_band,
            'pref': pref,
            'allergy': allergy,
            'diet_btn': diet_btn,
            'diet_out': diet_out
        }

    @staticmethod
    def create_certification_tab():
        """인증/챌린지 안내 탭을 생성합니다."""
        with gr.Tab("인증/챌린지 안내", elem_id="certification_tab"):
            gr.Markdown("### 🏆 크로스핏 인증 및 챌린지")
            cert_query_status = gr.Markdown("", elem_id="cert_query_status")
            gr.Markdown("---")
            gr.Markdown("### 📋 기본 정보")
            cert_btn = gr.Button("정보 요약 보기")
            cert_out = gr.Markdown("크로스핏 인증 및 챌린지 정보", elem_id="certification_output")
        
        return {
            'cert_query_status': cert_query_status,
            'cert_btn': cert_btn,
            'cert_out': cert_out
        }

    @staticmethod
    def create_mentoring_tab():
        """멘토링(초보 심리/동기) 탭을 생성합니다."""
        with gr.Tab("멘토링(초보 심리/동기)", elem_id="mentoring_tab"):
            gr.Markdown("### 💪 초보자 멘토링 및 동기부여")
            mentoring_query_status = gr.Markdown("", elem_id="mentoring_query_status")
            gr.Markdown("---")
            gr.Markdown("### 🎯 맞춤 멘토링")
            topic = gr.Radio(
                ["첫 수업 긴장", "페이스 조절", "목표 설정"],
                value="첫 수업 긴장",
                label="멘토링 주제 선택",
                info="도움이 필요한 주제를 선택해주세요."
            )
            mt_btn = gr.Button("멘토링 메시지 챗봇에 전송", variant="secondary")
            mt_out = gr.Markdown("", elem_id="mentoring_status_output")
        
        return {
            'mentoring_query_status': mentoring_query_status,
            'topic': topic,
            'mt_btn': mt_btn,
            'mt_out': mt_out
        }

    @staticmethod
    def create_evidence_tab():
        """근거 자료 허브 탭을 생성합니다."""
        with gr.Tab("근거 자료 허브", elem_id="evidence_hub_tab"):
            evidence_full = gr.Markdown("수집된 근거 자료", elem_id="evidence_full_output")
        
        return {
            'evidence_full': evidence_full
        }

    @staticmethod
    def create_weight_converter_tab():
        """무게 변환 탭을 생성합니다."""
        with gr.Tab("무게 변환", elem_id="weight_converter_tab"):
            gr.Markdown("### ⚖️ 무게 단위 변환")
            gr.Markdown("kg과 lb 간의 무게 변환을 쉽게 할 수 있습니다.")
            
            with gr.Row():
                w_val = gr.Number(
                    label="변환할 무게",
                    value=100,
                    info="변환하려는 무게 값을 입력하세요."
                )
                w_dir = gr.Radio(
                    ["kg → lb", "lb → kg"],
                    value="kg → lb",
                    label="변환 방향",
                    info="변환할 방향을 선택하세요."
                )
            
            w_btn = gr.Button("변환하기", variant="primary", size="lg")
            w_out = gr.Markdown("변환 결과가 여기에 표시됩니다.", elem_id="weight_output")
        
        return {
            'w_val': w_val,
            'w_dir': w_dir,
            'w_btn': w_btn,
            'w_out': w_out
        }