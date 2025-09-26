# CrossFit 코칭 애플리케이션 - 설치 및 실행 가이드

이 프로젝트는 CrossFit 코칭 애플리케이션입니다.

## 🎯 프로젝트 특징

✅ **프로젝트**: 모든 데이터, 설정, 소스코드 포함  
✅ **즉시 실행 가능**: API 키만 설정하면 바로 실행  
✅ **샘플 데이터 포함**: CrossFit 가이드 PDF, 데이터베이스  
✅ **MVC 아키텍처**: 체계적으로 구성된 코드 구조

## 🚀 빠른 시작

### 1. 저장소 클론
```bash
git clone https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN16-3st-2Team.git
cd SKN16-3st-2Team
```

### 2. Python 가상환경 생성 및 활성화

#### Option A: venv 사용 (권장)
```bash
# 가상환경 생성
python -m venv crossfit_env

# 가상환경 활성화
# Windows
crossfit_env\Scripts\activate
# macOS/Linux
source crossfit_env/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정 (필수)
프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가:

```env
# OpenAI API 설정 (필수 - VectorDB 기능용)
OPENAI_API_KEY=your_openai_api_key_here

# LangChain 설정 (선택사항 - 디버깅용)
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_PROJECT=crossfit_coaching_app
LANGCHAIN_TRACING_V2=true

# 애플리케이션 설정
APP_NAME=CrossFit_Coaching_App
DEBUG=False
```

> 💡 **참고**: OpenAI API 키가 없어도 기본 UI 및 사용자 관리 기능은 정상 작동합니다.

### 5. 애플리케이션 실행
```bash
python run.py
```

브라우저에서 표시되는 URL(`http://localhost:{PORT}`)로 접속하세요.

## 🔧 문제 해결

### API 키 관련
- OpenAI API 키가 없으면 VectorDB 기능이 제한됩니다.
- 기본 UI 및 사용자 관리 기능은 정상 작동합니다.

### 포트 충돌
- 기본 포트(7861)가 사용 중이면 자동으로 다른 포트를 찾습니다.
- 수동 포트 지정: `python run.py --port 8080`

### 의존성 오류
```bash
# 패키지 업그레이드
pip install --upgrade -r requirements.txt

# 개별 패키지 설치
pip install gradio langchain openai chromadb
```


## 📁 주요 폴더 구조 (요약)

```
SKN16-3st-2Team/
├── src/         # 소스 코드 (MVC)
├── data/        # 데이터셋, PDF, DB
├── docs/        # 문서
├── notebooks/   # Jupyter 노트북
├── run.py       # 실행 파일
└── README.md    # 프로젝트 개요
```


## ✨ 주요 기능 (요약)
- AI 챗봇 (PDF 기반 Q&A)
- 영상 분석 (데모)
- 개인 맞춤 추천
- 지식 베이스/관리자 도구/사용자 관리

> 로그인 정보, 상세 기능 설명 등은 [SETUP_GUIDE.md](SETUP_GUIDE.md) 참고

## 🔒 보안 및 설정

- ✅ **모든 데이터 포함**: PDF, 데이터베이스, 설정 파일
- ⚠️ **API 키만 별도 설정**: `.env` 파일로 개인 API 키 관리
- 🔐 **안전한 구조**: 민감한 정보는 환경 변수로 분리

## 📞 지원

- 📖 **상세 문서**: [README.md](README.md) 참조
- 🐛 **버그 리포트**: GitHub Issues 활용
- 💡 **기능 제안**: Discussions 탭 활용