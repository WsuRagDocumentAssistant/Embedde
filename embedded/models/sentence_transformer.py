import logging
import time
from typing import List, Optional

from ..base import BaseEmbeddedModel, DenseCapable
from ..hf_utils import resolve_model_path

logger = logging.getLogger(__name__)


class SentenceTransformerModel(BaseEmbeddedModel, DenseCapable):
    """sentence-transformers로 로드 가능한 모든 HF 모델을 커버하는 범용 백엔드.

    e5, bge, ko-sroberta 등 대부분의 dense 임베딩 모델은
    이 클래스 하나 + 모델별 인자(model_name, 접두사)로 처리한다.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        local_dir: Optional[str] = None,
        query_prefix: str = "",
        passage_prefix: str = "",
        normalize: bool = True,
        batch_size: int = 12,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        ignore_patterns: Optional[List[str]] = None,
        max_workers: int = 4,
        local_files_only: bool = False,
    ):
        super().__init__(batch_size=batch_size)
        from sentence_transformers import SentenceTransformer  # 지연 import: 미설치 환경에서도 패키지 로드는 가능해야 함

        self._name = model_name
        self.normalize = normalize            # DenseCapable 클래스 기본값을 인스턴스 값으로 덮어씀
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        logger.info("모델 로딩 시작: %s", model_name)
        start = time.time()
        path = resolve_model_path(
            model_name, local_dir,
            revision=revision, token=token, ignore_patterns=ignore_patterns,
            max_workers=max_workers, local_files_only=local_files_only,
        )
        self._model = SentenceTransformer(path, device=device)
        logger.info("모델 로딩 완료: %s (%.1fs)", model_name, time.time() - start)

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        # sentence-transformers 최신 버전에서 메서드명이 변경됨
        getter = getattr(self._model, "get_embedding_dimension", None)
        if getter is None:
            getter = self._model.get_sentence_embedding_dimension
        return getter()

    def _encode_raw(self, texts: List[str]):
        return self._model.encode(
            texts,
            batch_size=len(texts),        # 배치 분할은 base.encode()가 담당
            convert_to_tensor=True,       # GPU 텐서 그대로 반환 (CPU 변환은 base가 마지막에 한 번)
            normalize_embeddings=False,   # 정규화도 base가 담당
            show_progress_bar=False,
        )


# ---- 사용법 ---------------------------------------------------------------
# 구체 모델(어떤 HF 레포를 쓸지)은 패키지에 하드코딩하지 않는다.
# 클래스를 직접 생성하며 model_name 등을 넘긴다 — IDE 자동완성/타입 힌트를 그대로 받는다.
#
#   from embedded.models import SentenceTransformerModel
#
#   model = SentenceTransformerModel(
#       model_name="sentence-transformers/all-MiniLM-L6-v2",
#   )
#
# ⚠️ 접두사 주의: 일부 모델은 query/passage 접두사가 있어야 검색 성능이 나온다.
#    빠뜨려도 에러 없이 조용히 품질만 떨어지므로 생성 시 반드시 함께 지정할 것.
#      - intfloat/multilingual-e5-*  → query_prefix="query: ", passage_prefix="passage: "
#      - BAAI/bge 계열(영문 instruct) → 모델 카드의 지시 프롬프트 확인
#      - jhgan/ko-sroberta-multitask, all-MiniLM-* → 접두사 불필요
