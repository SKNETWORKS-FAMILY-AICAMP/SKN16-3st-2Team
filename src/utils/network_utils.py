#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network Utils - 네트워크 관련 유틸리티 함수들
"""

import socket
import random
from typing import Optional


class NetworkUtils:
    """네트워크 관련 유틸리티 기능을 제공하는 클래스"""
    
    @staticmethod
    def find_free_port(start: int = 7860, end: int = 7865) -> Optional[int]:
        """
        지정된 범위 내에서 사용 가능한 네트워크 포트를 찾아 반환합니다.
        
        Args:
            start (int): 시작 포트 번호
            end (int): 종료 포트 번호
            
        Returns:
            Optional[int]: 사용 가능한 포트 번호, 없으면 None
        """
        ports = list(range(start, end))
        random.shuffle(ports)

        for port in ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
        return None

    @staticmethod
    def is_port_available(port: int) -> bool:
        """
        지정된 포트가 사용 가능한지 확인합니다.
        
        Args:
            port (int): 확인할 포트 번호
            
        Returns:
            bool: 포트 사용 가능 여부
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
                return True
            except OSError:
                return False