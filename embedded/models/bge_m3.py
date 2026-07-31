import logging
import time
from typing import Dict, List, Optional

from ..base import BaseEmbeddedModel, DenseCapable, SparseCapable
from ..hf_utils import resolve_model_path

logger = logging.getLogger(__name__)


class BGEM3Model(BaseEmbeddedModel, DenseCapable, SparseCapable):
    """BGE-M3: dense + sparse(lexical) 동시 지원. FlagEmbedding 필요.

    기본은 다운로드 금지다. 이미 받아둔 로컬 폴더 경로를 넘겨서 쓰고,
    경로가 없거나 불완전하면 FileNotFoundError 로 즉시 실패한다(Hub로
    폴백하지 않는다). 개발 PC에서 새로 받을 때만 allow_download=True 와
    local_dir 을 함께 지정한다.

        # 운영: 반입한 폴더로 로드 (local_dir 은 쓰지 않는다)
        BGEM3Model(model_name="/srv/models/bge-m3")
        # 개발: Hub에서 받아 폴더에 저장
        BGEM3Model(model_name="BAAI/bge-m3", local_dir="models/bge-m3",
                   allow_download=True)

    Args:
        model_name: allow_download=False(기본)면 로드할 로컬 폴더 경로,
            True면 Hub 레포 ID(예: "BAAI/bge-m3").
        device: 연산 장치("cuda", "cuda:1", "cpu"). None이면 라이브러리가
            자동 선택(GPU 있으면 GPU).
        local_dir: 다운로드 대상 폴더. allow_download=True 일 때만 유효하며
            그때는 필수다. 기본 모드에서 지정하면 ValueError(로드 경로는
            model_name 으로 준다 — 두 인자가 어긋나는 상황을 막기 위함).
        allow_download: True면 Hub 접근/다운로드를 허용. 운영 서버에서는 끈다.
        use_fp16: 16비트로 로드. GPU에서 메모리 절반 + 속도 향상.
            CPU에서는 이점이 없으므로 False 권장.
        normalize: dense 벡터 L2 정규화. True면 내적이 곧 코사인 유사도.
            (FlagEmbedding 자체 정규화는 끄고 base 에서 일괄 처리하므로
            False 를 주면 실제로 정규화되지 않은 벡터가 나온다.)
        batch_size: encode() 한 번에 처리할 문장 수. GPU 메모리 여유에 맞춰 조절.
        passage_max_length: 문서 토큰 상한. 짧은 문서만 다루면 줄여서 VRAM 절약.
        query_max_length: 쿼리 토큰 상한.
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
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        local_dir: Optional[str] = None,
        allow_download: bool = False,
        use_fp16: bool = True,
        normalize: bool = True,
        batch_size: int = 12,
        passage_max_length: int = 512,
        query_max_length: int = 512,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        ignore_patterns: Optional[List[str]] = None,
        max_workers: int = 4,
        local_files_only: bool = False,
    ):
        super().__init__(batch_size=batch_size)
        from FlagEmbedding import BGEM3FlagModel  # 지연 import

        self.normalize = normalize            # DenseCapable 클래스 기본값을 인스턴스 값으로 덮어씀
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
        self._model = BGEM3FlagModel(
            path,
            use_fp16=use_fp16,
            # FlagEmbedding 1.3+ 는 인자명이 devices(복수)다. device= 로 주면
            # **kwargs 로 흘러 조용히 무시되므로(GPU 지정이 안 먹음) 주의.
            devices=device,
            # 정규화는 base(DenseCapable)가 담당하므로 여기서는 끈다.
            # 켜두면 normalize=False 를 줘도 정규화된 벡터가 나온다.
            normalize_embeddings=False,
            passage_max_length=passage_max_length,
            query_max_length=query_max_length,
        )
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
        try:
            return self._model.model.config.hidden_size
        except AttributeError:
            # FlagEmbedding 내부 구조가 바뀐 경우의 폴백 (bge-m3 dense 기본 차원).
            # 조용히 넘기면 잘못된 차원으로 벡터 DB 컬렉션을 만들 수 있어 경고를 남긴다.
            logger.warning(
                "FlagEmbedding 내부 구조에서 hidden_size 를 읽지 못해 기본값 1024 를 "
                "사용합니다. 라이브러리 버전 변경으로 실제 차원과 다를 수 있습니다: %s",
                self._name,
            )
            return 1024

    def _encode_raw(self, texts: List[str], batch_size: int):
        # 배치 분할은 FlagEmbedding 에 맡긴다(길이순 정렬로 패딩 낭비를 줄임).
        out = self._model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return out["dense_vecs"]  # numpy (FlagEmbedding이 내부에서 CPU로 반환)

    def _encode_sparse_raw(
        self, texts: List[str], batch_size: int
    ) -> List[Dict[int, float]]:
        # 빈 입력 방어는 SparseCapable.encode_sparse()가 이미 처리 (여기는 항상 non-empty)
        out = self._model.encode(
            texts,
            batch_size=batch_size,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        # lexical_weights: 문장마다 {token_id(str): weight} — 키를 int로 통일
        return [
            {int(tok): float(w) for tok, w in weights.items()}
            for weights in out["lexical_weights"]
        ]
