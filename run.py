#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CrossFit 코칭 애플리케이션 실행 파일 (MVC 구조)
"""

import sys
import os

def main():
    """메인 실행 함수"""    
    # src 디렉토리를 Python 경로에 추가
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from src.main import main as app_main
        app_main()
    except ImportError as e:
        print(f"❌ 모듈 import 오류: {e}")
        print("필요한 패키지를 설치해주세요:")
        print("pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()