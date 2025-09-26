#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CrossFit 코칭 애플리케이션 메인 파일 (MVC 구조)

이 파일은 크로스핏 코칭 데모 애플리케이션 메인 파일입니다.
"""

import gradio as gr
from typing import Dict, Any

# Config 및 Utils
from .config import config, DB_PATH, CHROMA_DIR, CHROMA_BACKUP_DIR, PDF_GUIDE_DIR
from .utils import NetworkUtils, StateUtils

# Models
from .models import UserModel, QAModel, VectorDBModel

# Views
from .views import AuthView, MainView, ChatbotView, VideoView, RecommendationView, TopicViews

# Controllers
from .controllers import (
    AuthController, ChatbotController, VideoController, 
    RecommendationController, TopicController, AdminController
)


class CrossFitApp:
    """CrossFit 코칭 애플리케이션 메인 클래스"""
    
    def __init__(self):
        """애플리케이션 초기화"""
        self.app_state = StateUtils.init_app_state()
        
        # Models 초기화
        self.user_model = UserModel(DB_PATH)
        self.qa_model = QAModel(DB_PATH)
        self.vector_db_model = VectorDBModel(CHROMA_DIR, PDF_GUIDE_DIR, config.openai_api_key)
        # app_state에 vector_db_model 명시적으로 할당
        self.app_state["vector_db_model"] = self.vector_db_model
        
        # Controllers 초기화
        self.auth_controller = AuthController(self.user_model, self.app_state)
        self.chatbot_controller = ChatbotController(self.qa_model, self.vector_db_model, self.app_state)
        self.video_controller = VideoController(self.app_state)
        self.recommendation_controller = RecommendationController(self.app_state)
        self.topic_controller = TopicController(self.qa_model, self.app_state)
        self.admin_controller = AdminController(self.vector_db_model, CHROMA_BACKUP_DIR)
        
        # VectorDB 및 QA Chain 초기화
        self._initialize_vector_db()

    def _initialize_vector_db(self):
        """VectorDB와 QA Chain을 초기화합니다."""
        print("VectorDB 초기화 중...")
        vectordb = self.vector_db_model.initialize_vectordb()
        
        if vectordb:
            print("QA Chain 초기화 중...")
            qa_chain = self.vector_db_model.initialize_qa_chain()
            if not qa_chain:
                print("⚠️  QA Chain 초기화에 실패했습니다.")
        else:
            print("⚠️  VectorDB 초기화에 실패했습니다.")

    def build_gradio_app(self) -> gr.Blocks:
        """Gradio 애플리케이션을 구축합니다."""
        
        # CSS 스타일링
        custom_css = """
        .topic-query-btn {
            font-size: 18px !important;
            font-weight: bold !important;
            padding: 15px 30px !important;
            border-radius: 10px !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.3) !important;
            transition: all 0.3s ease !important;
        }
        
        .topic-query-btn:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.4) !important;
        }
        
        .status-message {
            background: #f8f9fa !important;
            border-left: 4px solid #007bff !important;
            padding: 12px !important;
            border-radius: 5px !important;
            margin: 10px 0 !important;
        }
        
        .tab-nav button {
            font-size: 14px !important;
            font-weight: 600 !important;
            padding: 10px 16px !important;
            border-radius: 8px !important;
            margin: 2px !important;
        }
        """
        
        with gr.Blocks(title="CrossFit 코칭 데모 (MVC 구조)", css=custom_css) as demo:
            # Views 생성
            auth_components = AuthView.create_auth_page()
            main_components = MainView.create_main_layout()
            
            # 탭 생성
            with main_components['main_tabs']:
                chatbot_components = ChatbotView.create_chatbot_tab()
                video_components = VideoView.create_video_tab()
                recommendation_components = RecommendationView.create_recommendation_tab()
                glossary_components = TopicViews.create_glossary_tab()
                diet_components = TopicViews.create_diet_tab()
                cert_components = TopicViews.create_certification_tab()
                mentoring_components = TopicViews.create_mentoring_tab()
                weight_components = TopicViews.create_weight_converter_tab()
                evidence_components = TopicViews.create_evidence_tab()
                admin_components = MainView.create_admin_tab()
            
            # 이벤트 바인딩
            self._bind_events(demo, {
                **auth_components,
                **main_components,
                **chatbot_components,
                **video_components,
                **recommendation_components,
                **glossary_components,
                **diet_components,
                **cert_components,
                **mentoring_components,
                **weight_components,
                **evidence_components,
                **admin_components
            })
        
        return demo

    def _bind_events(self, demo: gr.Blocks, components: Dict[str, Any]):
        """이벤트 바인딩을 설정합니다."""
        
        # 인증 이벤트
        components['pwd'].submit(
            self.auth_controller.do_login,
            [components['email'], components['name'], components['pwd']],
            [components['entry_page'], components['main_page'], components['welcome'], components['admin_tab']]
        )
        components['login_btn'].click(
            self.auth_controller.do_login,
            [components['email'], components['name'], components['pwd']],
            [components['entry_page'], components['main_page'], components['welcome'], components['admin_tab']]
        )
        
        components['signup_btn'].click(
            self.auth_controller.do_signup,
            [components['email'], components['name'], components['pwd']],
            [components['entry_page'], components['main_page'], components['welcome']]
        )
        
        components['logout_btn'].click(
            self.auth_controller.do_logout,
            None,
            [components['entry_page'], components['main_page'], components['entry_status'], 
             components['email'], components['name'], components['pwd']]
        )
        
        # 챗봇 이벤트
        components['send_btn'].click(
            self.chatbot_controller.send_chat,
            [components['user_input']],
            [components['chat_view'], components['entry_status'], components['links_panel'], 
             components['evidence_full']]
        )
        
        components['download_btn'].click(
            self.chatbot_controller.download_history,
            None,
            [components['download_file']]
        )
        # 주제별 자동 쿼리 버튼 이벤트 제거됨
        
        # 기타 기능 이벤트
        components['analyze_btn'].click(
            self.video_controller.analyze_video,
            [components['pose'], components['vfile']],
            [components['user_v'], components['ref_v'], components['metrics'], 
             components['coaching'], components['history']]
        )
        
        components['rec_btn'].click(
            self.recommendation_controller.generate_recommendation,
            [components['level'], components['goal'], components['freq'], components['gear']],
            [components['rec_out'], components['rec_status']]
        )
        
        components['search_btn'].click(
            self.topic_controller.search_glossary,
            [components['q'], components['cat']],
            [components['glo_out']]
        )
        
        components['diet_btn'].click(
            self.topic_controller.get_diet_recovery_guide,
            [components['weight_band'], components['pref'], components['allergy']],
            [components['diet_out']]
        )
        
        components['cert_btn'].click(
            lambda: self.topic_controller.get_certification_info(),
            None,
            [components['cert_out']]
        )
        
        components['mt_btn'].click(
            self.topic_controller.get_mentoring_preset,
            [components['topic']],
            [components['mt_out']]
        )
        
        components['w_btn'].click(
            self.topic_controller.convert_weight,
            [components['w_val'], components['w_dir']],
            [components['w_out']]
        )
        
        # 관리자 이벤트
        components['action_btn'].click(
            self.admin_controller.do_action,
            [components['row_select'], components['backup_desc'], components['action_type']],
            [components['db_table'], components['op_result']]
        )
        
        # 초기 로드 이벤트
        def refresh_evidence_on_load():
            evidence_md = self.chatbot_controller._sources_md()
            return evidence_md, evidence_md
        
        def table_reset():
            return self.admin_controller.get_all_db_rows(), ""
        
        demo.load(refresh_evidence_on_load, None, [components['evidence_full'], components['evidence_brief']])
        demo.load(table_reset, None, [components['db_table'], components['op_result']])

    def run(self):
        """애플리케이션을 실행합니다."""
        print("CrossFit 코칭 데모 애플리케이션을 시작합니다...")
        
        # Gradio 앱 빌드
        print("Gradio UI 구성 중...")
        demo = self.build_gradio_app()
        
        # 포트 탐색
        port = NetworkUtils.find_free_port() or 7861
        
        # 앱 실행
        print(f"애플리케이션을 포트 {port}에서 시작합니다...")
        print(f"브라우저에서 http://localhost:{port} 로 접속하세요.")
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            debug=True,
            show_error=True,
            share=False,
            prevent_thread_lock=True
        )


def main():
    """메인 실행 함수"""
    app = CrossFitApp()
    app.run()


if __name__ == "__main__":
    main()