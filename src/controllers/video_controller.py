#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Controller - 영상 코칭 관련 비즈니스 로직을 처리하는 컨트롤러
"""

from typing import Tuple, Dict, Any
from datetime import datetime


class VideoController:
    """영상 코칭 관련 비즈니스 로직을 처리하는 컨트롤러"""
    
    def __init__(self, app_state: Dict[str, Any]):
        """
        VideoController 초기화
        
        Args:
            app_state (Dict[str, Any]): 애플리케이션 상태
        """
        self.app_state = app_state

    def analyze_video(self, pose_type: str, video: str) -> Tuple:
        """
        사용자가 업로드한 운동 영상을 분석합니다.
        
        Args:
            pose_type (str): 운동 자세 타입
            video (str): 업로드된 비디오 파일 경로
            
        Returns:
            Tuple: 분석 결과
        """
        if not self.app_state["user_session"]["auth"]:
            return None, None, "로그인 후 이용해 주세요.", "", ""
        
        if not pose_type:
            return None, None, "자세를 선택해 주세요.", "", ""
        
        if video is None:
            return None, None, "영상(mp4)을 업로드해 주세요.", "", ""

        # 데모용 모킹 결과
        result = {
            "ref_video_url": "https://cdn.example.com/ref_snatch.mp4",
            "user_overlay_video_url": video,
            "metrics": {
                "score": 88,
                "bar_path_deviation": 3.0,
                "tempo": 1.05,
                "labels": ["주의"]
            },
            "coaching_text": "바벨이 몸에서 멀어집니다. 가슴을 열고 바를 몸 가까이 끌어올리세요."
        }
        
        # 비디오 히스토리 업데이트
        self.app_state.setdefault("video_history", []).append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "pose": pose_type,
            "metrics": result["metrics"],
            "coaching": result["coaching_text"]
        })

        metrics_text = (
            f"점수: {result['metrics']['score']} | "
            f"궤적편차: {result['metrics']['bar_path_deviation']} | "
            f"템포: {result['metrics']['tempo']}"
        )
        
        return (
            result["user_overlay_video_url"], 
            result["ref_video_url"], 
            metrics_text, 
            result["coaching_text"], 
            self.app_state["video_history"]
        )