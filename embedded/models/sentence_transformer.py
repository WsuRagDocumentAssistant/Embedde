import logging
import time
from typing import List, Optional

from ..base import BaseEmbeddedModel, DEFAULT_BATCH_SIZE, DenseCapable
from ..hf_utils import ensure_model_dir

logger = logging.getLogger(__name__)


class SentenceTransformerModel(BaseEmbeddedModel, DenseCapable):
    """sentence-transformers로 로드 가능한 모든 HF 모델을 커버하는 범용 백엔드.

    e5, bge, ko-sroberta 등 대부분의 dense 임베딩 모델은
    이 클래스 하나 + 모델별 인자(model_path, 접두사)로 처리한다.

    이미 받아둔 로컬 폴더만 로드한다 — 다운로드는 하지 않는다. 경로가 없거나
    불완전하면 FileNotFoundError 로 즉시 실패한다.
    개발 PC 에서 가중치를 받으려면 embedded.hf_utils.download_model() 을 쓴다.

        model = SentenceTransformerModel(
            "/srv/models/e5-large",
            query_prefix="query: ", passage_prefix="passage: ",
        )

    Args:
        model_path: 이미 받아둔 모델 폴더 경로. (위치 인자로 줄 수 있다)
        device: 연산 장치("cuda", "cuda:1", "cpu"). None이면 라이브러리가
            자동 선택(GPU 있으면 GPU).
        query_prefix: encode_queries()가 각 텍스트 앞에 붙이는 접두사.
            e5 계열은 "query: " 필요. 아래 접두사 주의 참고.
        passage_prefix: encode_documents()가 각 텍스트 앞에 붙이는 접두사.
            e5 계열은 "passage: " 필요.
        normalize: dense 벡터 L2 정규화. True면 내적이 곧 코사인 유사도.
        batch_size: 백엔드에 전달할 배치 크기. GPU 메모리 여유에 맞춰 조절.

    Note:
        model_name 속성은 로드한 폴더 경로를 돌려준다.

    Raises:
        FileNotFoundError: 경로가 없거나 모델 폴더로서 불완전할 때.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: Optional[str] = None,
        query_prefix: str = "",
        passage_prefix: str = "",
        normalize: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        super().__init__(batch_size=batch_size)
        from sentence_transformers import SentenceTransformer  # 지연 import: 미설치 환경에서도 패키지 로드는 가능해야 함

        path = ensure_model_dir(model_path)
        self._name = path
        self.normalize = normalize            # DenseCapable 클래스 기본값을 인스턴스 값으로 덮어씀
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        logger.info("모델 로딩 시작: %s", path)
        start = time.time()
        self._model = SentenceTransformer(path, device=device)
        logger.info("모델 로딩 완료: %s (%.1fs)", path, time.time() - start)

    def _require_model(self) -> None:
        if self._model is None:
            raise RuntimeError(
                f"모델이 이미 unload 되었습니다: {self._name!r}. "
                "다시 사용하려면 인스턴스를 새로 생성하세요."
            )

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def device(self) -> str:
        """가중치가 실제로 올라간 연산 장치("cpu", "cuda:0" 등).

        생성 시 넘긴 device 인자를 그대로 돌려주지 않고 실측한다 — 라이브러리가
        인자를 무시하거나 지연 이동하는 경우가 있어(예: FlagEmbedding, bge_m3.py
        참고) 입력값과 실제 위치가 다를 수 있기 때문이다.
        """
        self._require_model()
        return str(self._model.device)

    @property
    def dimension(self) -> int:
        self._require_model()
        # sentence-transformers 최신 버전에서 메서드명이 변경됨
        getter = getattr(self._model, "get_embedding_dimension", None)
        if getter is None:
            getter = self._model.get_sentence_embedding_dimension
        return getter()

    def _encode_raw(self, texts: List[str], batch_size: int):
        return self._model.encode(
            texts,
            # 배치 분할은 sentence-transformers 에 맡긴다
            # (길이순 정렬로 패딩 낭비를 줄이므로 미리 자르면 손해)
            batch_size=batch_size,
            convert_to_tensor=True,       # GPU 텐서 그대로 반환 (CPU 변환은 base가 한 번)
            normalize_embeddings=False,   # 정규화도 base가 담당
            show_progress_bar=False,
        )


# ---- 사용법 ---------------------------------------------------------------
# 어떤 HF 모델을 쓸지는 패키지에 하드코딩하지 않는다.
# 클래스를 직접 생성하며 폴더 경로를 넘긴다 — IDE 자동완성/타입 힌트를 그대로 받는다.
#
#   from embedded.models import SentenceTransformerModel
#
#   model = SentenceTransformerModel("models/minilm-l6")
#
# 개발 PC 에서 가중치를 새로 받으려면:
#
#   from embedded.hf_utils import download_model
#   download_model("sentence-transformers/all-MiniLM-L6-v2", "models/minilm-l6")
#
# ⚠️ 접두사 주의: 일부 모델은 query/passage 접두사가 있어야 검색 성능이 나온다.
#    빠뜨려도 에러 없이 조용히 품질만 떨어지므로 생성 시 반드시 함께 지정할 것.
#      - intfloat/multilingual-e5-*  → query_prefix="query: ", passage_prefix="passage: "
#      - BAAI/bge 계열(영문 instruct) → 모델 카드의 지시 프롬프트 확인
#      - jhgan/ko-sroberta-multitask, all-MiniLM-* → 접두사 불필요
