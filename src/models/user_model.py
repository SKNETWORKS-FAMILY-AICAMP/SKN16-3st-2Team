#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User Model - 사용자 관련 데이터베이스 작업을 담당하는 모델
"""

import sqlite3
import hashlib
from datetime import datetime
from typing import Optional, Tuple


class UserModel:
    """사용자 관련 데이터베이스 작업을 처리하는 모델 클래스"""
    
    def __init__(self, db_path: str):
        """
        UserModel 초기화
        
        Args:
            db_path (str): SQLite 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.init_tables()
    
    def init_tables(self):
        """사용자 관련 테이블들을 초기화합니다."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()

            # 유저 테이블
            cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
              email TEXT PRIMARY KEY,
              nickname TEXT,
              password_hash TEXT,
              created_at TEXT,
              role TEXT DEFAULT "user"
            );
            """)

            # 로그인/로그아웃 로그
            cur.execute("""
            CREATE TABLE IF NOT EXISTS session_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT,
                action TEXT,
                ts TEXT
            )
            """)

            # 기본 관리자 계정 생성
            admin_email, admin_nickname, admin_pw = "admin@crossfit.com", "관리자", "admin123"
            pw_hash = self._hash_password(admin_pw)
            cur.execute("INSERT OR IGNORE INTO users(email, nickname, password_hash, created_at, role) VALUES (?, ?, ?, datetime('now'), 'admin')", 
                       (admin_email, admin_nickname, pw_hash))
            
            # 기본 데모 계정 생성
            demo_email, demo_nickname, demo_pw = "demo@demo.com", "데모", "x"
            pw_hash = self._hash_password(demo_pw)
            cur.execute("INSERT OR IGNORE INTO users(email, nickname, password_hash, created_at) VALUES (?, ?, ?, datetime('now'))", 
                       (demo_email, demo_nickname, pw_hash))
            
            conn.commit()

    def _hash_password(self, password: str) -> str:
        """비밀번호를 해시화합니다."""
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def get_user(self, email: str) -> Optional[Tuple]:
        """
        사용자 정보를 조회합니다.
        
        Args:
            email (str): 사용자 이메일
            
        Returns:
            Optional[Tuple]: (email, nickname, password_hash, role) 또는 None
        """
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT email, nickname, password_hash, role FROM users WHERE email=?", (email,))
            return c.fetchone()

    def register_user(self, email: str, nickname: str, password: str) -> bool:
        """
        새 사용자를 등록합니다.
        
        Args:
            email (str): 사용자 이메일
            nickname (str): 사용자 닉네임
            password (str): 사용자 비밀번호
            
        Returns:
            bool: 등록 성공 여부
        """
        try:
            pw_hash = self._hash_password(password)
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute("INSERT OR IGNORE INTO users(email, nickname, password_hash, created_at) VALUES (?, ?, ?, datetime('now'))", 
                         (email, nickname, pw_hash))
                conn.commit()
                return c.rowcount > 0
        except Exception as e:
            print(f"Error registering user: {e}")
            return False

    def login_user(self, email: str, password: str) -> Tuple[Optional[dict], str]:
        """
        사용자 로그인을 처리합니다.
        
        Args:
            email (str): 사용자 이메일
            password (str): 사용자 비밀번호
            
        Returns:
            Tuple[Optional[dict], str]: (세션 정보, 메시지)
        """
        user = self.get_user(email)
        if not user: 
            return None, "존재하지 않는 계정"
        
        if self._hash_password(password) != user[2]:
            return None, "비밀번호 오류"
        
        session = {"user_id": user[0], "name": user[1], "role": user[3]}
        return session, f"{user[1]}님, 환영합니다."

    def is_admin(self, session: Optional[dict]) -> bool:
        """
        관리자 권한을 확인합니다.
        
        Args:
            session (Optional[dict]): 사용자 세션 정보
            
        Returns:
            bool: 관리자 여부
        """
        return session and session.get("role") == "admin"

    def log_session(self, email: str, action: str):
        """
        사용자 세션 로그를 기록합니다.
        
        Args:
            email (str): 사용자 이메일
            action (str): 액션 ('login', 'logout', 'signup')
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO session_logs (email, action, ts) VALUES (?, ?, ?)",
                        (email, action, datetime.now().isoformat()))
            conn.commit()