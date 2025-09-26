"""
Views package for CrossFit Coaching App

This package contains all UI components and layouts.
"""

from .auth_view import AuthView
from .main_view import MainView
from .chatbot_view import ChatbotView
from .video_view import VideoView
from .recommendation_view import RecommendationView
from .topic_views import TopicViews

__all__ = [
    'AuthView',
    'MainView',
    'ChatbotView',
    'VideoView',
    'RecommendationView',
    'TopicViews'
]