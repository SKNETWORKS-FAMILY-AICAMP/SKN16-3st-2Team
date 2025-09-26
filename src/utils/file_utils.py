#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Utils - 파일 관련 유틸리티 함수들
"""

import os
import shutil
from typing import List, Optional


class FileUtils:
    """파일 관련 유틸리티 기능을 제공하는 클래스"""
    
    @staticmethod
    def ensure_directory(directory_path: str):
        """디렉토리가 존재하지 않으면 생성합니다."""
        os.makedirs(directory_path, exist_ok=True)

    @staticmethod
    def get_pdf_files(directory: str) -> List[str]:
        """
        지정된 디렉토리에서 PDF 파일 목록을 반환합니다.
        
        Args:
            directory (str): 검색할 디렉토리 경로
            
        Returns:
            List[str]: PDF 파일 경로 리스트
        """
        if not os.path.exists(directory):
            return []
        
        return [
            os.path.join(directory, f) 
            for f in os.listdir(directory) 
            if f.lower().endswith('.pdf')
        ]

    @staticmethod
    def copy_directory(src: str, dst: str) -> bool:
        """
        디렉토리를 복사합니다.
        
        Args:
            src (str): 소스 디렉토리
            dst (str): 대상 디렉토리
            
        Returns:
            bool: 복사 성공 여부
        """
        try:
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            return True
        except Exception as e:
            print(f"Error copying directory: {e}")
            return False

    @staticmethod
    def remove_directory(directory: str) -> bool:
        """
        디렉토리를 삭제합니다.
        
        Args:
            directory (str): 삭제할 디렉토리
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            return True
        except Exception as e:
            print(f"Error removing directory: {e}")
            return False

    @staticmethod
    def file_exists(file_path: str) -> bool:
        """
        파일이 존재하는지 확인합니다.
        
        Args:
            file_path (str): 파일 경로
            
        Returns:
            bool: 파일 존재 여부
        """
        return os.path.isfile(file_path)

    @staticmethod
    def get_file_size(file_path: str) -> Optional[int]:
        """
        파일 크기를 반환합니다.
        
        Args:
            file_path (str): 파일 경로
            
        Returns:
            Optional[int]: 파일 크기 (바이트), 파일이 없으면 None
        """
        try:
            return os.path.getsize(file_path)
        except OSError:
            return None