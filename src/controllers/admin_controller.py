#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Admin Controller - 관리자 기능 관련 비즈니스 로직을 처리하는 컨트롤러
"""

import os
from typing import List, Tuple, Dict, Any
from ..models.vector_db_model import VectorDBModel


class AdminController:
    """관리자 기능 관련 비즈니스 로직을 처리하는 컨트롤러"""
    
    def __init__(self, vector_db_model: VectorDBModel, backup_dir: str):
        """
        AdminController 초기화
        
        Args:
            vector_db_model (VectorDBModel): VectorDB 모델 인스턴스
            backup_dir (str): 백업 디렉토리 경로
        """
        self.vector_db_model = vector_db_model
        self.backup_dir = backup_dir

    def get_all_db_rows(self) -> List[List[str]]:
        """
        모든 DB 행 정보를 가져옵니다.
        
        Returns:
            List[List[str]]: DB 행 정보 리스트
        """
        try:
            curr_p = os.path.join(self.vector_db_model.chroma_dir, "chroma.sqlite3")
            curr_row = ["0", "현재 DB", *self._sqlite_file_info(curr_p), "백업", ""]
            backup_rows = []
            
            if os.path.exists(self.backup_dir):
                for idx, ver in enumerate(sorted(os.listdir(self.backup_dir), reverse=True), 1):
                    sqlite_path = os.path.join(self.backup_dir, ver, "chroma.sqlite3")
                    fn, sz, dt = self._sqlite_file_info(sqlite_path)
                    backup_rows.append([str(idx), ver, fn, sz, dt, "롤백", "삭제"])
            
            result = [curr_row] + backup_rows
            # 최소한 빈 행이라도 반환하여 Table 오류 방지
            return result if result else [["", "", "", "", "", "", ""]]
        except Exception as e:
            # 오류 발생 시 빈 행 반환
            return [["오류", f"DB 정보를 불러올 수 없습니다: {str(e)}", "", "", "", "", ""]]

    def backup_db(self, description: str) -> Tuple[List[List[str]], str]:
        """
        데이터베이스를 백업합니다.
        
        Args:
            description (str): 백업 설명
            
        Returns:
            Tuple[List[List[str]], str]: (업데이트된 DB 행 정보, 결과 메시지)
        """
        success, message = self.vector_db_model.backup_db(self.backup_dir, description)
        return self.get_all_db_rows(), message

    def handle_rollback(self, row_idx: int) -> Tuple[List[List[str]], str]:
        """
        데이터베이스를 롤백합니다.
        
        Args:
            row_idx (int): 롤백할 행 인덱스
            
        Returns:
            Tuple[List[List[str]], str]: (업데이트된 DB 행 정보, 결과 메시지)
        """
        all_rows = self.get_all_db_rows()
        if row_idx == 0:
            return all_rows, "⚠️ 현재 DB는 롤백 대상 아님"
        
        if row_idx >= len(all_rows):
            return all_rows, "❌ 올바른 행 번호를 선택하세요."
        
        ver = all_rows[int(row_idx)][1]
        src = os.path.join(self.backup_dir, ver)
        
        success, message = self.vector_db_model.restore_db(src)
        if success:
            message = f"✅ 복구 완료: {ver}"
        
        return self.get_all_db_rows(), message

    def handle_delete(self, row_idx: int) -> Tuple[List[List[str]], str]:
        """
        백업을 삭제합니다.
        
        Args:
            row_idx (int): 삭제할 행 인덱스
            
        Returns:
            Tuple[List[List[str]], str]: (업데이트된 DB 행 정보, 결과 메시지)
        """
        all_rows = self.get_all_db_rows()
        if row_idx == 0:
            return all_rows, "⚠️ 현재 DB는 삭제 불가"
        
        if row_idx >= len(all_rows):
            return all_rows, "❌ 올바른 행 번호를 선택하세요."
        
        ver = all_rows[int(row_idx)][1]
        path = os.path.join(self.backup_dir, ver)
        
        try:
            if not os.path.exists(path):
                return all_rows, "❌ 이미 삭제된 백업본입니다!"
            
            import shutil
            shutil.rmtree(path)
            return self.get_all_db_rows(), f"🗑️ 백업본 삭제 완료: {ver}"
        except Exception as e:
            return all_rows, f"❌ 삭제 실패: {str(e)}"

    def do_action(self, idx: int, desc: str, act_type: str) -> Tuple[List[List[str]], str]:
        """
        관리자 액션을 수행합니다.
        
        Args:
            idx (int): 행 인덱스
            desc (str): 설명
            act_type (str): 액션 타입
            
        Returns:
            Tuple[List[List[str]], str]: (업데이트된 DB 행 정보, 결과 메시지)
        """
        rows = self.get_all_db_rows()
        if idx < 0 or idx >= len(rows):
            return rows, "❌ 올바른 행 번호를 선택하세요."
        
        if act_type == "백업":
            return self.backup_db(desc)
        elif act_type == "롤백":
            return self.handle_rollback(int(idx))
        elif act_type == "삭제":
            return self.handle_delete(int(idx))
        else:
            return rows, "❓ 지원하지 않는 작업"

    def _sqlite_file_info(self, path: str) -> Tuple[str, str, str]:
        """
        SQLite 파일 정보를 반환합니다.
        
        Args:
            path (str): 파일 경로
            
        Returns:
            Tuple[str, str, str]: (파일명, 크기, 수정일)
        """
        if not os.path.isfile(path):
            return "-", "-", "-"
        
        size = os.path.getsize(path) // 1024
        from datetime import datetime
        date = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
        return os.path.basename(path), f"{size} KB", date