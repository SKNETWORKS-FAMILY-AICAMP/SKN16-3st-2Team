#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VectorDB Model - 벡터 데이터베이스 관련 작업을 담당하는 모델
"""

import os
import shutil
from datetime import datetime
from typing import Optional, List, Dict, Any
import tiktoken

try:
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain.vectorstores import Chroma
    from langchain.prompts import ChatPromptTemplate
    from langchain.chains import RetrievalQA
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain not installed. VectorDB features will be limited.")


class VectorDBModel:
    """VectorDB 관련 작업을 처리하는 모델 클래스"""
    
    def __init__(self, chroma_dir: str, pdf_guide_dir: str, openai_api_key: str):
        """
        VectorDBModel 초기화
        
        Args:
            chroma_dir (str): ChromaDB 저장 디렉토리
            pdf_guide_dir (str): PDF 가이드 파일 디렉토리
            openai_api_key (str): OpenAI API 키
        """
        self.chroma_dir = chroma_dir
        self.pdf_guide_dir = pdf_guide_dir
        self.openai_api_key = openai_api_key
        self.vectordb = None
        self.qa_chain = None

    def chroma_db_exists(self) -> bool:
        """ChromaDB가 존재하는지 확인합니다."""
        return os.path.isfile(os.path.join(self.chroma_dir, "chroma.sqlite3"))

    def initialize_vectordb(self):
        """VectorDB를 초기화합니다."""
        if not LANGCHAIN_AVAILABLE or not self.openai_api_key:
            print("Warning: VectorDB initialization skipped - missing dependencies or API key")
            return None
            
        if self.chroma_db_exists():
            print("⚡ 기존 VectorDB가 이미 존재합니다. batch/bulk 임베딩 건너뜀.")
            return self.load_existing_vectordb()
        
        # PDF 파일들이 있는지 확인
        if not os.path.exists(self.pdf_guide_dir):
            print(f"Warning: PDF guide directory not found: {self.pdf_guide_dir}")
            return None
        
        pdf_files = [f for f in os.listdir(self.pdf_guide_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print(f"Warning: No PDF files found in {self.pdf_guide_dir}")
            return None
        
        print(f"Found {len(pdf_files)} PDF files. Initializing VectorDB...")
        
        embeddings = OpenAIEmbeddings()
        
        # PDF 파일 로드
        docs = []
        for pdf_file in pdf_files:
            file_path = os.path.join(self.pdf_guide_dir, pdf_file)
            try:
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
                print(f"Loaded: {pdf_file}")
            except Exception as e:
                print(f"Error loading {pdf_file}: {e}")

        if not docs:
            print("No documents were loaded successfully.")
            return None

        # 문서 분할
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        all_chunks = splitter.split_documents(docs)

        # 배치별 임베딩 및 저장
        for i, batch_chunks in enumerate(self._batch_by_token_limit(all_chunks, 290000)):
            db = Chroma.from_documents(
                documents=batch_chunks,
                embedding=embeddings,
                persist_directory=self.chroma_dir
            )
            db.persist()
            del db
            print(f"Batch {i+1} 저장 완료 (chunks: {len(batch_chunks)})")
        
        print("✅ 신규 임베딩 및 VectorDB 생성 완료")
        return self.load_existing_vectordb()

    def load_existing_vectordb(self):
        """기존 VectorDB를 로드합니다."""
        if not LANGCHAIN_AVAILABLE or not self.openai_api_key:
            return None
            
        try:
            embeddings = OpenAIEmbeddings()
            self.vectordb = Chroma(
                persist_directory=self.chroma_dir,
                embedding_function=embeddings
            )
            print("VectorDB 로딩 완료!")
            return self.vectordb
        except Exception as e:
            print(f"Error loading VectorDB: {e}")
            return None

    def initialize_qa_chain(self):
        """QA Chain을 초기화합니다."""
        if not LANGCHAIN_AVAILABLE or not self.vectordb or not self.openai_api_key:
            return None
            
        try:
            llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.3
            )

            prompt = ChatPromptTemplate.from_template(
            """
            아래 문서 참고(context)해서 반드시 한국어로만, 구체적으로 답해줘.
            <context>
            {context}
            </context>
            질문: {question}
            """
            )

            retriever = self.vectordb.as_retriever(search_kwargs={"k": 4})

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": prompt
                }
            )
            
            print("QA Chain 초기화 완료!")
            return self.qa_chain
        except Exception as e:
            print(f"Error initializing QA chain: {e}")
            return None

    def query(self, question: str) -> Dict[str, Any]:
        """
        질문에 대한 답변을 생성합니다.
        
        Args:
            question (str): 질문
            
        Returns:
            Dict[str, Any]: 답변 및 소스 문서 정보
        """
        if not self.qa_chain:
            return {
                "result": "현재 문서 검색 기능이 비활성화되어 있습니다.",
                "source_documents": []
            }
        
        try:
            return self.qa_chain.invoke({"query": question})
        except Exception as e:
            print(f"Error in QA query: {e}")
            return {
                "result": "죄송합니다. 현재 질문 처리에 문제가 있습니다.",
                "source_documents": []
            }

    def _batch_by_token_limit(self, chunks, max_tokens=290000, model_name="text-embedding-3-small"):
        """토큰 수 기반으로 청크를 배치로 나눕니다."""
        if not LANGCHAIN_AVAILABLE:
            yield chunks
            return
            
        tokenizer = tiktoken.encoding_for_model(model_name)
        curr_batch = []
        curr_tokens = 0
        for chunk in chunks:
            text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
            tokens = len(tokenizer.encode(text))
            if curr_tokens + tokens > max_tokens and curr_batch:
                yield curr_batch
                curr_batch = []
                curr_tokens = 0
            curr_batch.append(chunk)
            curr_tokens += tokens
        if curr_batch:
            yield curr_batch

    def backup_db(self, backup_dir: str, description: str = "") -> tuple:
        """
        데이터베이스를 백업합니다.
        
        Args:
            backup_dir (str): 백업 디렉토리
            description (str): 백업 설명
            
        Returns:
            tuple: (성공 여부, 메시지)
        """
        try:
            now = datetime.now().strftime("%Y%m%d-%H%M%S")
            ver_folder = f"{now}-{description.strip()}" if description.strip() else now
            target_path = os.path.join(backup_dir, ver_folder)
            shutil.copytree(self.chroma_dir, target_path)
            return True, f"📦 백업 완료: {ver_folder}"
        except Exception as e:
            return False, f"❌ 백업 실패: {str(e)}"

    def restore_db(self, backup_path: str) -> tuple:
        """
        백업에서 데이터베이스를 복원합니다.
        
        Args:
            backup_path (str): 백업 파일 경로
            
        Returns:
            tuple: (성공 여부, 메시지)
        """
        try:
            if not os.path.exists(backup_path):
                return False, "❌ 백업본이 존재하지 않습니다!"
            
            if os.path.exists(self.chroma_dir):
                shutil.rmtree(self.chroma_dir)
            shutil.copytree(backup_path, self.chroma_dir)
            
            # VectorDB 다시 로드
            self.load_existing_vectordb()
            self.initialize_qa_chain()
            
            return True, f"✅ 복구 완료"
        except Exception as e:
            return False, f"❌ 복구 실패: {str(e)}"