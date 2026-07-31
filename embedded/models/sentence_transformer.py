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

    기본은 다운로드 금지다. 이미 받아둔 로컬 폴더 경로를 넘겨서 쓰고,
    경로가 없거나 불완전하면 FileNotFoundError 로 즉시 실패한다(Hub로
    폴백하지 않는다). 개발 PC에서 새로 받을 때만 allow_download=True 와
    local_dir 을 함께 지정한다.

        # 운영: 반입한 폴더로 로드 (local_dir 은 쓰지 않는다)
        SentenceTransformerModel(model_name="/srv/models/e5-large", ...)
        # 개발: Hub에서 받아 폴더에 저장
        SentenceTransformerModel(model_name="intfloat/multilingual-e5-large",
                                 local_dir="models/e5", allow_download=True, ...)

    Args:
        model_name: allow_download=False(기본)면 로드할 로컬 폴더 경로,
            True면 Hub 레포 ID(예: "intfloat/multilingual-e5-large").
            범용 백엔드이므로 필수.
        device: 연산 장치("cuda", "cuda:1", "cpu"). None이면 라이브러리가
            자동 선택(GPU 있으면 GPU).
        local_dir: 다운로드 대상 폴더. allow_download=True 일 때만 유효하며
            그때는 필수다. 기본 모드에서 지정하면 ValueError(로드 경로는
            model_name 으로 준다 — 두 인자가 어긋나는 상황을 막기 위함).
        allow_download: True면 Hub 접근/다운로드를 허용. 운영 서버에서는 끈다.
        query_prefix: encode_queries()가 각 텍스트 앞에 붙이는 접두사.
            e5 계열은 "query: " 필요. 아래 접두사 주의 참고.
        passage_prefix: encode_documents()가 각 텍스트 앞에 붙이는 접두사.
            e5 계열은 "passage: " 필요.
        normalize: dense 벡터 L2 정규화. True면 내적이 곧 코사인 유사도.
        batch_size: encode() 한 번에 처리할 문장 수. GPU 메모리 여유에 맞춰 조절.
        revision: 브랜치/태그/커밋 해시. None이면 main 브랜치 최신 커밋.
        token: private/gated 레포 인증 토큰.
        ignore_patterns: 다운로드에서 제외할 파일 글롭(예: ["onnx/*"]).
        max_workers: 병렬 다운로드 스레드 수.
        local_files_only: True면 네트워크 요청 없이 로컬 파일만 사용.

    Note:
        revision 이후 5개는 실제 다운로드가 일어날 때만 쓰인다.
        model_name 속성은 인자가 아니라 **실제 로드된 경로**를 돌려준다.

    Raises:
        FileNotFoundError: 로컬 경로가 없거나 모델 폴더로서 불완전할 때.
        ValueError: 모드에 맞지 않는 인자 조합일 때(위 local_dir 설명 참고).
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        local_dir: Optional[str] = None,
        allow_download: bool = False,
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

        self.normalize = normalize            # DenseCapable 클래스 기본값을 인스턴스 값으로 덮어씀
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        logger.info("모델 로딩 시작: %s", model_name)
        start = time.time()
        path = resolve_model_path(
            model_name, local_dir, allow_download=allow_download,
            revision=revision, token=token, ignore_patterns=ignore_patterns,
            max_workers=max_workers, local_files_only=local_files_only,
        )
        # 실제 로드한 경로를 기록한다 — 다운로드 모드면 local_dir 이 되므로
        # 인자로 받은 model_name(레포 ID)과 다를 수 있다.
        self._name = path
        self._model = SentenceTransformer(path, device=device)
        logger.info("모델 로딩 완료: %s (%.1fs)", path, time.time() - start)

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        if self._model is None:
            raise RuntimeError(
                f"모델이 이미 unload 되었습니다: {self._name!r}. "
                "다시 사용하려면 인스턴스를 새로 생성하세요."
            )
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
# 구체 모델(어떤 HF 레포를 쓸지)은 패키지에 하드코딩하지 않는다.
# 클래스를 직접 생성하며 model_name 등을 넘긴다 — IDE 자동완성/타입 힌트를 그대로 받는다.
#
#   from embedded.models import SentenceTransformerModel
#
#   # 이미 받아둔 폴더로 로드 (기본: 다운로드 금지)
#   model = SentenceTransformerModel(model_name="models/minilm-l6")
#
#   # 개발 PC에서 Hub에서 새로 받기
#   model = SentenceTransformerModel(
#       model_name="sentence-transformers/all-MiniLM-L6-v2",
#       local_dir="models/minilm-l6", allow_download=True,
#   )
#
# ⚠️ 접두사 주의: 일부 모델은 query/passage 접두사가 있어야 검색 성능이 나온다.
#    빠뜨려도 에러 없이 조용히 품질만 떨어지므로 생성 시 반드시 함께 지정할 것.
#      - intfloat/multilingual-e5-*  → query_prefix="query: ", passage_prefix="passage: "
#      - BAAI/bge 계열(영문 instruct) → 모델 카드의 지시 프롬프트 확인
#      - jhgan/ko-sroberta-multitask, all-MiniLM-* → 접두사 불필요
