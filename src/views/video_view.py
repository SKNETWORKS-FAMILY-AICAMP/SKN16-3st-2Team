#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video View - 영상 코칭 UI 컴포넌트
"""
import gradio as gr

class VideoView:
    """영상 코칭 관련 UI 컴포넌트를 관리하는 클래스"""
    
    @staticmethod
    def create_video_tab():
        """영상 코칭 탭을 생성합니다."""
        with gr.Tab("영상 코칭", elem_id="video_coaching_tab"):
            pose = gr.Dropdown(
                ["파워 스내치", "스쿼트 클린", "파워 클린"],
                label="분석할 자세 선택",
                info="분석하고자 하는 운동 자세를 선택해주세요.",
                interactive=True
            )
            vfile = gr.Video(label="사용자 영상 업로드(mp4)", sources=["upload"])
            analyze_btn = gr.Button("영상 분석 실행", variant="primary")
            with gr.Row():
                user_v = gr.Video(label="사용자 영상 분석 결과")
                ref_v = gr.Video(label="참조(레퍼런스) 영상")
            metrics = gr.Markdown("지표", label="분석 지표", elem_id="metrics_output")
            coaching = gr.Textbox("코칭", label="맞춤 코칭 피드백", lines=5, interactive=False)
            history = gr.JSON(label="영상 분석 히스토리(현재 세션)", elem_id="video_history_json_output")
        
        return {
            'pose': pose,
            'vfile': vfile,
            'analyze_btn': analyze_btn,
            'user_v': user_v,
            'ref_v': ref_v,
            'metrics': metrics,
            'coaching': coaching,
            'history': history
        }