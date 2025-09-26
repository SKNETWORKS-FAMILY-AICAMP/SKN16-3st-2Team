#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main View - 메인 페이지 레이아웃 및 공통 UI 컴포넌트
"""

import gradio as gr


class MainView:
    """메인 페이지 레이아웃을 관리하는 클래스"""
    
    @staticmethod
    def create_main_layout():
        """메인 페이지 레이아웃을 생성합니다."""
        with gr.Column(visible=False) as main_page:
            # 헤더 영역 - 환영 메시지와 로그아웃 버튼
            with gr.Row(elem_id="welcome_logout_row", elem_classes="header-row"):
                with gr.Column(scale=4):
                    welcome = gr.Markdown("", elem_id="welcome_message_area")
                with gr.Column(scale=1):
                    logout_btn = gr.Button("로그아웃", size="sm", elem_id="logout_button")
            
            # 구분선 추가
            gr.HTML("<hr style='margin: 10px 0; border: none; border-bottom: 1px solid #e0e0e0;'>")

            with gr.Row(elem_id="main_content_row"):
                # 좌측 메뉴 영역
                left_menu_components = MainView._create_left_menu()
                
                # 우측 탭 콘텐츠 영역
                with gr.Column(scale=3, elem_id="right_tabs_column"):
                    main_tabs = gr.Tabs(elem_id="main_tabs")
        
        result = {
            'main_page': main_page,
            'welcome': welcome,
            'logout_btn': logout_btn,
            'main_tabs': main_tabs
        }
        
        # 좌측 메뉴 컴포넌트들을 포함
        result.update(left_menu_components)
        
        return result
    
    @staticmethod
    def _create_left_menu():
        """좌측 메뉴를 생성합니다."""
        with gr.Column(scale=1, elem_id="left_menu_column"):
            gr.Markdown("#### 메뉴")
            gr.Markdown("""
- 챗봇(Q&A)
- 영상 코칭
- 개인 맞춤 추천
- 용어/규칙
- 식단/회복
- 인증 안내
- 멘토링
- 근거 자료 허브
            """)

            gr.Markdown("#### KG↔LB 계산기")
            w_val = gr.Textbox(label="무게 값", placeholder="예: 60")
            w_dir = gr.Radio(choices=["kg→lb", "lb→kg"], value="kg→lb", label="변환 방향")
            w_btn = gr.Button("변환")
            w_out = gr.Markdown()

            gr.Markdown("#### 근거 자료 허브(요약)")
            evidence_brief = gr.Markdown(value="", elem_id="evidence_brief_summary")
        
        return {
            'w_val': w_val,
            'w_dir': w_dir,
            'w_btn': w_btn,
            'w_out': w_out,
            'evidence_brief': evidence_brief
        }

    @staticmethod
    def create_admin_tab():
        """관리자 탭을 생성합니다."""
        with gr.Tab("관리자 VectorDB 관리", visible=False) as admin_tab:
            gr.Markdown("## 🗄️ **VectorDB 관리자 대시보드**")
            backup_desc = gr.Textbox(label="백업 설명(선택)", placeholder="ex) major-update, 실험 등")
            import pandas as pd
            
            # 초기 빈 데이터프레임 생성
            initial_data = pd.DataFrame(
                columns=["행 번호", "구분(버전명)", "파일명", "크기", "최종수정", "롤백", "삭제"]
            )
            
            db_table = gr.Dataframe(
                value=initial_data,
                headers=["행 번호", "구분(버전명)", "파일명", "크기", "최종수정", "롤백", "삭제"],
                interactive=False,
                label="DB 관리",
                row_count=(0, "dynamic"),
                col_count=(7, "fixed"),
            )
            op_result = gr.Textbox(label="실행 결과", lines=3, interactive=False)

            row_select = gr.Number(
                label="행 번호 선택 (0=현재, 1~=백업행)",
                minimum=0, value=0, precision=0
            )

            action_type = gr.Radio(choices=["백업", "롤백", "삭제"], value="백업", label="실행 작업 선택")
            action_btn = gr.Button("작업 실행")
        
        return {
            'admin_tab': admin_tab,
            'backup_desc': backup_desc,
            'db_table': db_table,
            'op_result': op_result,
            'row_select': row_select,
            'action_type': action_type,
            'action_btn': action_btn
        }