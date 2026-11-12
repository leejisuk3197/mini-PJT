"""옷차림 문서 검색용 RAG 파이프라인을 정의하는 모듈."""

import os
import re
import threading
from pathlib import Path

from dotenv import load_dotenv
from kiwipiepy import Kiwi
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "data" / "docs"
CHROMA_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "weatherfit_docs"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
TOP_K = 3
HYBRID_WEIGHTS = [0.3, 0.7]  # [BM25 비중, 벡터 비중]
KEEP_POS_TAGS = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "SL", "SN"}

kiwi = Kiwi()
_hybrid_retriever = None
_build_lock = threading.Lock()  # 도구가 병렬 호출돼도 검색기를 한 번만 만들도록 막는다

FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def load_documents() -> list[Document]:
    """data/docs의 마크다운 문서를 읽어 front matter의 doc_id·title을 메타데이터로 붙인다."""
    documents = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        meta = {"source": path.name}
        match = FRONT_MATTER.match(text)
        if match:
            for line in match.group(1).splitlines():
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
            text = text[match.end():]
        documents.append(Document(page_content=text, metadata=meta))
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """문서를 '##' 소제목 단위 청크로 자르고, 청크 앞에 문서 제목과 소제목을 붙인다."""
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "title_h1"), ("##", "section")])
    chunks = []
    for doc in documents:
        for i, piece in enumerate(splitter.split_text(doc.page_content)):
            section = piece.metadata.get("section", "개요")
            title = doc.metadata.get("title", piece.metadata.get("title_h1", ""))
            chunks.append(Document(
                page_content=f"[{title}] {section}\n{piece.page_content}",
                metadata={
                    "doc_id": doc.metadata["doc_id"],
                    "title": title,
                    "section": section,
                    "chunk_id": f"{doc.metadata['doc_id']}#{i}",
                },
            ))
    return chunks


def build_embeddings() -> BedrockEmbeddings:
    """Titan v2 임베딩 모델을 만든다."""
    return BedrockEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def build_vectorstore(chunks: list[Document], rebuild: bool = False) -> Chroma:
    """청크를 Chroma에 저장한다. 같은 청크 id로 이미 저장돼 있으면 다시 임베딩하지 않는다."""
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=build_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    ids = [c.metadata["chunk_id"] for c in chunks]
    stored = set(store.get(include=[])["ids"])
    if rebuild or stored != set(ids):
        if stored:
            store.delete(ids=list(stored))
        store.add_documents(chunks, ids=ids)
    return store


def build_vector_retriever(chunks: list[Document], k: int = TOP_K):
    """Chroma 벡터 검색기를 만든다."""
    return build_vectorstore(chunks).as_retriever(search_kwargs={"k": k})


def korean_tokenizer(text: str) -> list[str]:
    """kiwi 형태소 분석으로 명사·어간·숫자·외래어만 남기는 BM25용 토크나이저."""
    return [t.form for t in kiwi.tokenize(text) if t.tag in KEEP_POS_TAGS]


def build_bm25_retriever(chunks: list[Document], k: int = TOP_K) -> BM25Retriever:
    """청크 목록으로 한국어 BM25 키워드 검색기를 만든다."""
    bm25 = BM25Retriever.from_documents(chunks, preprocess_func=korean_tokenizer)
    bm25.k = k
    return bm25


def build_hybrid_retriever(k: int = TOP_K) -> EnsembleRetriever:
    """BM25와 Chroma 벡터 검색을 가중치로 합친 하이브리드 검색기를 만든다."""
    chunks = split_documents(load_documents())
    return EnsembleRetriever(
        retrievers=[build_bm25_retriever(chunks, k), build_vector_retriever(chunks, k)],
        weights=HYBRID_WEIGHTS,
    )


def get_retriever() -> EnsembleRetriever:
    """하이브리드 검색기를 한 번만 만들어 재사용한다. 여러 스레드가 동시에 불러도 안전하다."""
    global _hybrid_retriever
    with _build_lock:
        if _hybrid_retriever is None:
            _hybrid_retriever = build_hybrid_retriever()
    return _hybrid_retriever


def search(query: str, k: int = TOP_K) -> list[Document]:
    """질의로 옷차림 문서를 하이브리드 검색해 상위 k개 청크를 돌려준다."""
    return get_retriever().invoke(query)[:k]
