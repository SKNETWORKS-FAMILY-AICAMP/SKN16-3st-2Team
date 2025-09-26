# CrossFit 코칭 애플리케이션 설정 가이드

## 📋 개요

이 가이드는 CrossFit 코칭 데모 애플리케이션의 설치 및 실행 방법을 안내합니다.

## 🔧 환경 변수 설정

#### 방법 1: .env 파일 사용 (권장)

1. `.env.example` 파일을 `.env`로 복사:
```bash
cp config/.env.example .env
```

2. `.env` 파일을 열어 실제 API 키로 수정:
```bash
# .env 파일 내용
OPENAI_API_KEY=your_actual_openai_api_key_here
LANGCHAIN_API_KEY=your_actual_langchain_api_key_here
LANGCHAIN_PROJECT=ai_camp_3rd_project
```

> 💡 **빠른 시작**: 기본 설치법은 [SETUP_GITHUB.md](SETUP_GITHUB.md)를 참조하세요.

## 🚀 설치 및 실행 요약

설치/실행법, 환경 변수, 폴더 구조 등은 [SETUP_GITHUB.md](SETUP_GITHUB.md) 참고

### 주요 변경사항 (Colab → 로컬 환경)
- 메인 애플리케이션: 루트 → `run.py` (MVC 구조)
- MVC 소스 코드: `src/` 디렉토리 (models, views, controllers, utils)
- 환경 변수: 루트 → `.env` 파일 
- 문서: 루트 → `docs/` 디렉토리
- 데이터베이스: `/content/drive/MyDrive/...` → `./data/sqlite_db/`
- ChromaDB: `/content/drive/MyDrive/...` → `./data/chroma_db/`
- PDF 가이드: `/content/drive/MyDrive/...` → `./data/raw/crossfit_guide/`

### 추가된 기능들
- `.env` 파일 자동 로드 기능 추가
- 체계적인 폴더 구조 적용
- 실행 스크립트 분리 (`run.py`)
- 설정 파일 분리 (`config/settings.py`)
- API 참조 문서 추가

## 🔧 기능 설명

### 1. 챗봇 Q&A
- **작동 조건**: OpenAI API 키 + PDF 파일들
- **없을 경우**: 기본 응답 메시지 표시
- **기능**: PDF 문서 기반 질의응답, 대화 히스토리 저장

### 2. 영상 코칭 (데모)
- **상태**: 모킹 데이터 사용 (실제 영상 분석 X)
- **기능**: 영상 업로드, 가상의 분석 결과 표시

### 3. 개인 맞춤 추천
- **기능**: 사용자 입력 기반 운동 계획 생성
- **출력**: JSON 형태의 상세 추천

### 4. 기타 기능들
- 용어/규칙 검색
- 식단/회복 가이드
- 단위 변환기 (kg ↔ lb)
- 관리자 VectorDB 관리

## 🔑 로그인 정보

### 관리자 계정
- **이메일**: `admin@crossfit.com`
- **비밀번호**: `admin123`
- **권한**: VectorDB 관리 탭 접근 가능

### 데모 계정
- **이메일**: `demo@demo.com`
- **비밀번호**: `x`
- **권한**: 일반 사용자

## ⚠️ 문제 해결

### API 키 관련
```
Warning: OPENAI_API_KEY environment variable not set!
```
→ `.env` 파일에 API 키를 설정하거나 환경 변수를 직접 설정하고 애플리케이션을 재시작하세요.

#### .env 파일 확인 방법:
1. 프로젝트 루트에 `.env` 파일이 있는지 확인
2. `.env` 파일에 `OPENAI_API_KEY=실제키값` 형태로 설정되어 있는지 확인
3. API 키에 따옴표나 불필요한 공백이 없는지 확인

### PDF 파일 관련
```
Warning: No PDF files found in data/raw/crossfit_guide/
```
→ PDF 파일을 해당 디렉토리에 배치하고 재시작하세요.

### 포트 충돌
→ 애플리케이션이 자동으로 사용 가능한 포트를 찾습니다 (7860-7960 범위).

### 데이터베이스 문제
→ `data/` 디렉토리의 하위 폴더들이 자동으로 생성됩니다.


## 📁 폴더 구조, 설치법 등은 [SETUP_GITHUB.md](SETUP_GITHUB.md) 참고

## 🔄 개발 워크플로우

1. **로컬 개발**: `python run.py`
2. **의존성 추가**: `requirements.txt` 업데이트
3. **설정 변경**: `config/settings.py` 수정
4. **문서 업데이트**: `docs/` 디렉토리에서 문서 관리
5. **데이터 백업**: 관리자 탭에서 DB 백업
6. **버전 관리**: Git으로 코드 변경사항 추적

## 📚 추가 문서

- **[API 참조](API_REFERENCE.md)**: 함수 및 클래스 상세 문서
- **[프로젝트 개요](../README.md)**: 전체 프로젝트 설명
- **[설치 가이드](SETUP_GITHUB.md)**: GitHub에서 프로젝트 설치법

---

💡 추가 질문이나 문제가 있으면 GitHub Issues를 생성해주세요.