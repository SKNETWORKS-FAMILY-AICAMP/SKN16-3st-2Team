#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration - 애플리케이션 설정 및 환경 변수 관리
"""

import os
from typing import Optional

# 프로젝트 루트 디렉토리 (src의 상위 디렉토리)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 데이터 디렉토리 경로들
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
ASSETS_DIR = os.path.join(DATA_DIR, "assets")
DB_DIR = os.path.join(DATA_DIR, "sqlite_db")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
CHROMA_BACKUP_DIR = os.path.join(DATA_DIR, "chroma_db_backups")
PDF_GUIDE_DIR = os.path.join(DATA_DIR, "raw", "crossfit_guide")

# 데이터베이스 경로
DB_PATH = os.path.join(DB_DIR, "users.db")

# 환경 변수 설정
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_PROJECT = "ai_camp_3rd_project"

# 포트 설정
DEFAULT_PORT_RANGE = (7860, 7865)
DEFAULT_PORT = 7861

class Config:
    """애플리케이션 설정을 관리하는 클래스"""
    
    def __init__(self):
        """Config 초기화"""
        self._load_env_file()
        self._setup_environment()
        self._ensure_directories()

    def _load_env_file(self):
        """프로젝트 루트의 .env 파일을 로드합니다."""
        env_path = os.path.join(PROJECT_ROOT, ".env")
        
        try:
            from dotenv import load_dotenv
            if os.path.exists(env_path):
                load_dotenv(env_path)
                print("✅ .env file loaded successfully using python-dotenv")
            else:
                print("⚠️  .env file not found. Please create one from .env.example")
        except ImportError:
            # 수동으로 .env 파일 로드
            if os.path.exists(env_path):
                print("Loading environment variables from .env file manually...")
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
                print("✅ .env file loaded successfully")
            else:
                print("⚠️  .env file not found. Please create one from .env.example")

    def _setup_environment(self):
        """환경 변수를 설정합니다."""
        os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
        os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

    def _ensure_directories(self):
        """필요한 디렉토리들을 생성합니다."""
        directories = [DB_DIR, CHROMA_DIR, CHROMA_BACKUP_DIR, PDF_GUIDE_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def get_api_key(self, key_name: str) -> str:
        """
        환경 변수에서 API 키를 가져옵니다.
        
        Args:
            key_name (str): API 키 이름
            
        Returns:
            str: API 키 값
        """
        api_key = os.environ.get(key_name)
        if not api_key:
            print(f"Warning: {key_name} environment variable not set!")
            print(f"Please set it in .env file or as environment variable")
        return api_key or ""

    @property
    def openai_api_key(self) -> str:
        """OpenAI API 키를 반환합니다."""
        return self.get_api_key("OPENAI_API_KEY")

    @property
    def langchain_api_key(self) -> str:
        """LangChain API 키를 반환합니다."""
        return self.get_api_key("LANGCHAIN_API_KEY")

    @property
    def langchain_project(self) -> str:
        """LangChain 프로젝트 이름을 반환합니다."""
        return self.get_api_key("LANGCHAIN_PROJECT") or LANGCHAIN_PROJECT


# 전역 설정 인스턴스
config = Config()