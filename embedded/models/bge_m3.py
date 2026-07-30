import logging
import time
from typing import Dict, List, Optional

from ..base import BaseEmbeddedModel, DenseCapable, SparseCapable
from ..hf_utils import resolve_model_path

logger = logging.getLogger(__name__)


class BGEM3Model(BaseEmbeddedModel, DenseCapable, SparseCapable):
    """BGE-M3: dense + sparse(lexical) 동시 지원. FlagEmbedding 필요.

    기본은 다운로드 금지다. 이미 받아둔 로컬 폴더 경로를 넘겨서 쓰고,
    경로가 없으면 FileNotFoundError 로 즉시 실패한다(전역 HF 캐시를 조용히
    채우지 않는다). 개발 PC에서 새로 받을 때만 allow_download=True 와
    local_dir 을 함께 지정한다.

        # 운영: 반입한 폴더로 로드
        BGEM3Model(model_name="/srv/models/bge-m3")
        # 개발: Hub에서 받아 폴더에 저장
        BGEM3Model(model_name="BAAI/bge-m3", local_dir="models/bge-m3",
                   allow_download=True)

    Args:
        model_name: 로컬 폴더 경로, 또는 allow_download=True 일 때 Hub 레포 ID
            (예: "BAAI/bge-m3").
        device: 연산 장치("cuda", "cuda:1", "cpu"). None이면 라이브러리가
            자동 선택(GPU 있으면 GPU).
        local_dir: 다운로드 대상 폴더. allow_download=True 일 때만 의미가 있고,
            그때는 필수다. 지정한 폴더에 파일을 그대로 받는다(전역 캐시 미사용).
        allow_download: True면 Hub 접근/다운로드를 허용. 운영 서버에서는 끈다.
        use_fp16: 16비트로 로드. GPU에서 메모리 절반 + 속도 향상.
            CPU에서는 이점이 없으므로 False 권장.
        normalize: dense 벡터 L2 정규화. True면 내적이 곧 코사인 유사도.
        batch_size: encode() 한 번에 처리할 문장 수. GPU 메모리 여유에 맞춰 조절.
        revision: 브랜치/태그/커밋 해시. None이면 main 브랜치 최신 커밋.
        token: private/gated 레포 인증 토큰.
        ignore_patterns: 다운로드에서 제외할 파일 글롭(예: ["onnx/*"]).
        max_workers: 병렬 다운로드 스레드 수.
        local_files_only: True면 네트워크 요청 없이 로컬 파일만 사용.

    Note:
        revision 이후 5개는 실제 다운로드가 일어날 때만 쓰인다.

    Raises:
        FileNotFoundError: allow_download=False 인데 경로가 없을 때.
        ValueError: allow_download=True 인데 local_dir 을 지정하지 않았을 때.
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
        revision: Optional[str] = None,
        token: Optional[str] = None,
        ignore_patterns: Optional[List[str]] = None,
        max_workers: int = 4,
        local_files_only: bool = False,
    ):
        super().__init__(batch_size=batch_size)
        from FlagEmbedding import BGEM3FlagModel  # 지연 import

        self._name = model_name
        self.normalize = normalize            # DenseCapable 클래스 기본값을 인스턴스 값으로 덮어씀
        logger.info("모델 로딩 시작: %s", model_name)
        start = time.time()
        path = resolve_model_path(
            model_name, local_dir, allow_download=allow_download,
            revision=revision, token=token, ignore_patterns=ignore_patterns,
            max_workers=max_workers, local_files_only=local_files_only,
        )
        self._model = BGEM3FlagModel(path, use_fp16=use_fp16, device=device)
        logger.info("모델 로딩 완료: %s (%.1fs)", model_name, time.time() - start)

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        try:
            return self._model.model.config.hidden_size
        except AttributeError:
            return 1024  # bge-m3 dense 기본 차원

    def _encode_raw(self, texts: List[str]):
        out = self._model.encode(
            texts,
            batch_size=len(texts),  # 배치 분할은 base.encode()가 담당
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return out["dense_vecs"]  # numpy (FlagEmbedding이 내부에서 CPU로 반환)

    def _encode_sparse_raw(self, texts: List[str]) -> List[Dict[int, float]]:
        # 빈 입력 방어는 SparseCapable.encode_sparse()가 이미 처리 (여기는 항상 non-empty)
        out = self._model.encode(
            texts,
            batch_size=self.batch_size,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        # lexical_weights: 문장마다 {token_id(str): weight} — 키를 int로 통일
        return [
            {int(tok): float(w) for tok, w in weights.items()}
            for weights in out["lexical_weights"]
        ]
