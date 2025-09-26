# API 참조 문서

이 문서는 CrossFit 코칭 애플리케이션의 주요 클래스와 메서드에 대한 API 참조를 제공합니다.

## 🏗️ MVC 아키텍처 개요

### 주요 컴포넌트
- **Models**: 데이터 관리 및 비즈니스 로직
- **Views**: UI 컴포넌트 (Gradio 인터페이스)
- **Controllers**: 사용자 입력 처리 및 모델-뷰 연결
- **Utils**: 공통 유틸리티 기능

## 📊 Models

### QAModel 클래스
질문답변 관련 데이터베이스 작업과 AI 모델 연동을 담당합니다.

#### `__init__(db_path: str)`
QAModel을 초기화합니다.
- **매개변수**: 
  - `db_path (str)`: SQLite 데이터베이스 파일 경로
- **설명**: OpenAI 모델 초기화 및 데이터베이스 테이블 생성

#### `search_glossary_with_ai(query: str, glossary: List[Dict]) -> str`
AI를 활용하여 용어사전을 검색합니다.
- **매개변수**:
  - `query (str)`: 검색 쿼리
  - `glossary (List[Dict])`: 용어사전 리스트
- **반환값**: `str` (AI가 생성한 검색 결과)
- **설명**: OpenAI GPT-4o-mini 모델을 사용한 스마트 검색

#### `generate_topic_query(topic: str) -> str`
주제에 따른 자동 쿼리를 생성합니다.
- **매개변수**: 
  - `topic (str)`: 주제명 ("용어/규칙/운동법", "식단/회복" 등)
- **반환값**: `str` (생성된 쿼리)
- **설명**: 주제별 미리 정의된 질문 중 랜덤 선택

### UserModel 클래스
사용자 인증 및 관리를 담당합니다.

#### `register_user(email: str, password: str) -> Tuple[bool, str]`
새 사용자를 등록합니다.
- **매개변수**:
  - `email (str)`: 사용자 이메일
  - `password (str)`: 비밀번호
- **반환값**: `Tuple[bool, str]` (성공 여부, 메시지)

#### `authenticate_user(email: str, password: str) -> Optional[Dict]`
사용자 인증을 처리합니다.
- **매개변수**:
  - `email (str)`: 이메일
  - `password (str)`: 비밀번호
- **반환값**: `Optional[Dict]` (사용자 정보 또는 None)

### VectorDBModel 클래스
ChromaDB 벡터 데이터베이스 관리를 담당합니다.

#### `__init__(chroma_dir: str, pdf_dir: str, api_key: str)`
VectorDB 모델을 초기화합니다.
- **매개변수**:
  - `chroma_dir (str)`: ChromaDB 디렉토리 경로
  - `pdf_dir (str)`: PDF 파일 디렉토리 경로
  - `api_key (str)`: OpenAI API 키

#### `query(text: str, k: int = 3) -> str`
벡터 데이터베이스에서 관련 문서를 검색합니다.
- **매개변수**:
  - `text (str)`: 검색 쿼리
  - `k (int)`: 반환할 문서 수
- **반환값**: `str` (검색 결과)

## 🎮 Controllers

### AuthController 클래스
사용자 인증 로직을 처리합니다.

#### `login(email: str, password: str) -> str`
로그인 처리
- **반환값**: `str` (상태 메시지)

#### `register(email: str, password: str, confirm_password: str) -> str`  
회원가입 처리
- **반환값**: `str` (상태 메시지)

### ChatbotController 클래스
챗봇 대화 로직을 처리합니다.

#### `chat_response(message: str, history: List, app_state: Dict) -> Tuple`
챗봇 응답 생성
- **매개변수**:
  - `message (str)`: 사용자 메시지
  - `history (List)`: 대화 기록
  - `app_state (Dict)`: 애플리케이션 상태
- **반환값**: `Tuple` (업데이트된 기록, 응답)

### RecommendationController 클래스
개인 맞춤 추천 로직을 처리합니다.

#### `generate_workout_plan(level: str, goal: str, freq: str, gear: List[str]) -> Tuple`
운동 계획 생성
- **매개변수**:
  - `level (str)`: 운동 수준
  - `goal (str)`: 운동 목표
  - `freq (str)`: 주당 빈도
  - `gear (List[str])`: 사용 장비
- **반환값**: `Tuple` (계획 텍스트, 데이터)

### VideoController 클래스
비디오 분석 로직을 처리합니다 (현재 모킹).

#### `analyze_exercise_video(video_path: str, exercise_type: str) -> Dict`
운동 비디오 분석 (데모)
- **반환값**: `Dict` (분석 결과)

### AdminController 클래스
관리자 기능을 처리합니다.

#### `create_backup(description: str) -> Tuple`
VectorDB 백업 생성
- **매개변수**: 
  - `description (str)`: 백업 설명
- **반환값**: `Tuple` (테이블 데이터, 상태 메시지)

## 👁️ Views

### MainView 클래스
메인 UI 레이아웃을 생성합니다.

#### `create_main_layout() -> gr.Row`
메인 헤더 레이아웃 생성
- **반환값**: `gr.Row` (Gradio Row 컴포넌트)

#### `create_admin_tab() -> gr.Tab`
관리자 탭 생성
- **반환값**: `gr.Tab` (Gradio Tab 컴포넌트)

### ChatbotView 클래스
챗봇 UI를 생성합니다.

#### `create_chatbot_interface() -> gr.Tab`
챗봇 인터페이스 생성
- **반환값**: `gr.Tab`

## 🛠️ Utils

### StateUtils 클래스
애플리케이션 상태 관리를 담당합니다.

#### `get_initial_state() -> Dict[str, Any]`
초기 애플리케이션 상태를 반환합니다.
- **반환값**: `Dict[str, Any]` (초기 상태)

#### `get_default_glossary() -> List[Dict[str, str]]`
기본 용어 사전을 반환합니다 (23개 CrossFit 용어).
- **반환값**: `List[Dict[str, str]]` (용어 사전)

### NetworkUtils 클래스
네트워크 관련 유틸리티를 제공합니다.

#### `find_free_port(start: int = 7860, end: int = 7865) -> Optional[int]`
사용 가능한 포트를 찾습니다.
- **매개변수**:
  - `start (int)`: 시작 포트
  - `end (int)`: 종료 포트
- **반환값**: `Optional[int]` (사용 가능한 포트 또는 None)

## � 주요 애플리케이션 클래스

### CrossFitApp 클래스
메인 애플리케이션 클래스로 모든 컴포넌트를 통합합니다.

#### `__init__()`
애플리케이션 초기화
- **설명**: 모든 Models, Controllers, 상태 초기화

#### `create_interface() -> gr.Interface`
전체 Gradio 인터페이스 생성
- **반환값**: `gr.Interface` (완전한 웹 애플리케이션)

## 📱 애플리케이션 상태 구조

```python
app_state = {
    "user_session": {
        "auth": bool,           # 인증 상태
        "user_id": str,         # 사용자 ID
        "email": str,           # 이메일
        "is_admin": bool        # 관리자 권한
    },
    "chat_history": List,       # 챗봇 대화 기록
    "evidence_library": List,   # 증거 자료 라이브러리
    "glossary": List,          # 용어 사전 (23개 용어)
    "recommendation_history": List  # 추천 기록
}
```

## 🔒 보안 고려사항

- 비밀번호는 SHA-256 해시로 저장
- 세션 키는 UUID를 사용하여 고유성 보장
- API 키는 환경 변수로 관리
- SQLite 데이터베이스는 로컬 파일 시스템에 저장

## 🚀 확장 가능성

- 영상 분석 기능은 실제 AI 모델로 교체 가능
- 추천 시스템은 머신러닝 모델로 고도화 가능
- 멀티 사용자 지원을 위한 세션 관리 강화 가능
- RESTful API로 백엔드 분리 가능