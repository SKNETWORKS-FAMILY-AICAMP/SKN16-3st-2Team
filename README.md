# SKN16-3rd-2Team
SKN 16기 3차 단위프로젝트

## 프로젝트 구조

```
SKN16-3st-2Team/
├── README.md                   # 프로젝트 설명 파일
├── .gitignore                  # Git 무시 파일 설정
├── data/                       # 데이터 디렉토리
│   ├── assets/                 # 이미지 등 미디어 파일
│   └── raw/                    # 원본 데이터
│       ├── crossfit_guide/     # CrossFit 가이드 PDF 문서들
│       └── questions/          # 질문 관련 데이터
├── notebooks/                  # Jupyter 파일들
└── vectordb/                   # VectorDB 디렉토리
```

### 디렉토리 설명

- **data/**: 프로젝트에서 사용하는 모든 데이터를 포함
  - **assets/**: 프로젝트에서 사용하는 미디어 파일들
  - **raw/**: 가공되지 않은 원본 데이터
    - **crossfit_guide/**: CrossFit 관련 PDF 가이드 문서들
    - **questions/**: CrossFit 관련 질문 데이터

- **notebooks/**: 데이터 분석 및 처리를 위한 Jupyter 노트북 파일들

- **vectordb/**: 벡터 데이터베이스 관련 파일들 (임베딩, 검색 등)
